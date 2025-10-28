import os
import csv
import glob
import torch
import argparse
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from ggtp_modules import GNNEncoder, GGTP_Decoder, cvae_loss_function
from torch.utils.data import DataLoader
from train_utils import *


def train_epoch(data_loader, encoder, decoder, optimizer, args):
    epoch_loss = []
    epoch_recon_loss = []
    epoch_kld_loss = []
    epoch_metrics = []
    encoder.train()
    decoder.train()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            # Prepare data
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

            # Encode scene
            optimizer.zero_grad()
            encoder_outputs = encoder(inputs)

            # First stage prediction (3 seconds)
            first_stage_trajectory = batch[7].to(args.device)
            agents_traj, scores, weights, mu, logvar = decoder(
                encoder_outputs, 
                first_stage_trajectory, 
                inputs['neighbor_agents_past'], 
                30,  # 3 seconds × 10 Hz
                ground_truth_futures=neighbors_gt_future[:, :, :30]  # Only first 30 steps
            )
            
            # CVAE loss
            loss, recon_loss, kld_loss = cvae_loss_function(
                agents_traj, 
                neighbors_gt_future[:, :, :30],  # Only first 30 steps
                mu, 
                logvar,
                beta=args.beta
            )
            
            # Add planning loss (optional, from original scorer)
            planning_loss = calc_loss(
                agents_traj,  # Already 30 timesteps from decoder [B, M, N, 30, 3]
                first_stage_trajectory[:, :, :30],  # First 30 steps [B, M, 30, 6]
                ego_gt_future[:, :30],  # First 30 steps [B, 30, 3]
                scores, 
                weights,
                ego_gt_future[:, :30],  # First 30 steps
                neighbors_gt_future[:, :, :30],  # First 30 steps
                neighbors_future_valid[:, :, :30]  # First 30 steps
            )
            
            total_loss = loss + 0.1 * planning_loss

            # Second stage prediction (8 seconds)
            second_stage_trajectory = batch[8].to(args.device)
            agents_traj_full, scores_full, weights_full, mu_full, logvar_full = decoder(
                encoder_outputs, 
                second_stage_trajectory, 
                inputs['neighbor_agents_past'], 
                80,  # 8 seconds × 10 Hz
                ground_truth_futures=neighbors_gt_future
            )
            
            loss_full, recon_loss_full, kld_loss_full = cvae_loss_function(
                agents_traj_full,
                neighbors_gt_future,
                mu_full,
                logvar_full,
                beta=args.beta
            )
            
            planning_loss_full = calc_loss(
                agents_traj_full,  # Already 80 timesteps from decoder
                second_stage_trajectory,
                ego_gt_future,
                scores_full,
                weights_full,
                ego_gt_future,
                neighbors_gt_future,
                neighbors_future_valid
            )
            
            total_loss += 0.2 * (loss_full + 0.1 * planning_loss_full)

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

            # Compute metrics
            metrics = calc_metrics(
                second_stage_trajectory, 
                agents_traj_full.mean(1),  # Average over ego trajectory candidates
                scores_full,
                ego_gt_future, 
                neighbors_gt_future, 
                neighbors_future_valid
            )
            
            epoch_metrics.append(metrics)
            epoch_loss.append(total_loss.item())
            epoch_recon_loss.append((recon_loss + recon_loss_full).item() / 2)
            epoch_kld_loss.append((kld_loss + kld_loss_full).item() / 2)
            
            data_epoch.set_postfix(
                loss=f'{np.mean(epoch_loss):.4f}',
                recon=f'{np.mean(epoch_recon_loss):.4f}',
                kld=f'{np.mean(epoch_kld_loss):.4f}'
            )

    # Show metrics
    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    
    logging.info(
        f"Loss: {np.mean(epoch_loss):.4f}, Recon: {np.mean(epoch_recon_loss):.4f}, "
        f"KLD: {np.mean(epoch_kld_loss):.4f}"
    )
    logging.info(
        f"plannerADE: {planningADE:.4f}, plannerFDE: {planningFDE:.4f}, "
        f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n"
    )
        
    return np.mean(epoch_loss), [planningADE, planningFDE, predictionADE, predictionFDE]


def valid_epoch(data_loader, encoder, decoder, args):
    epoch_loss = []
    epoch_metrics = []
    encoder.eval()
    decoder.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

            with torch.no_grad():
                encoder_outputs = encoder(inputs)

                # Full prediction
                second_stage_trajectory = batch[8].to(args.device)
                agents_traj, scores, weights = decoder(
                    encoder_outputs, 
                    second_stage_trajectory, 
                    inputs['neighbor_agents_past'], 
                    80
                )
                
                # Use prediction for validation
                # agents_traj: [B, M, N, T, 3], select first ego candidate
                pred_traj = agents_traj[:, 0]  # [B, N, T, 3]
                loss = F.mse_loss(pred_traj, neighbors_gt_future)

            # Compute metrics
            metrics = calc_metrics(
                second_stage_trajectory, 
                pred_traj,  # [B, N, T, 3]
                scores,
                ego_gt_future, 
                neighbors_gt_future, 
                neighbors_future_valid
            )
            
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss=f'{np.mean(epoch_loss):.4f}')

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    
    logging.info(
        f"val-plannerADE: {planningADE:.4f}, val-plannerFDE: {planningFDE:.4f}, "
        f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n"
    )

    return np.mean(epoch_loss), [planningADE, planningFDE, predictionADE, predictionFDE]


def model_training(args):
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Model: GGTP (GNN + CVAE)")
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Beta (KLD weight): {}".format(args.beta))
    logging.info("Use device: {}".format(args.device))

    # Set seed
    set_seed(args.seed)

    # Set up model
    encoder = GNNEncoder(
        node_dim=11,
        edge_dim=4,
        dim=256,
        heads=4,
        layers=2
    ).to(args.device)
    logging.info("GNN Encoder Params: {}".format(sum(p.numel() for p in encoder.parameters())))
    
    decoder = GGTP_Decoder(
        neighbors=args.num_neighbors,
        max_time=8,
        max_branch=args.num_candidates,
        agent_dim=256,
        ego_plan_dim=256,
        latent_dim=args.latent_dim,
        horizon=80,
        traj_dim=3
    ).to(args.device)
    logging.info("CVAE Decoder Params: {}".format(sum(p.numel() for p in decoder.parameters())))

    # Set up optimizer
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=args.learning_rate
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Set up data loaders
    train_set = DrivingData(
        glob.glob(os.path.join(args.train_set, '*.npz')), 
        args.num_neighbors, 
        args.num_candidates
    )
    valid_set = DrivingData(
        glob.glob(os.path.join(args.valid_set, '*.npz')), 
        args.num_neighbors, 
        args.num_candidates
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=4)
    logging.info(f"Dataset Prepared: {len(train_set)} train, {len(valid_set)} validation\n")
    
    # Begin training
    for epoch in range(args.train_epochs):
        logging.info(f"Epoch {epoch+1}/{args.train_epochs}")
        train_loss, train_metrics = train_epoch(train_loader, encoder, decoder, optimizer, args)
        val_loss, val_metrics = valid_epoch(valid_loader, encoder, decoder, args)

        # Save to training log
        log = {
            'epoch': epoch+1, 
            'loss': train_loss, 
            'lr': optimizer.param_groups[0]['lr'], 
            'val-loss': val_loss,
            'train-planningADE': train_metrics[0], 
            'train-planningFDE': train_metrics[1],
            'train-predictionADE': train_metrics[2], 
            'train-predictionFDE': train_metrics[3],
            'val-planningADE': val_metrics[0], 
            'val-planningFDE': val_metrics[1],
            'val-predictionADE': val_metrics[2], 
            'val-predictionFDE': val_metrics[3]
        }

        if epoch == 0:
            with open(f'{log_path}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'{log_path}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # Reduce learning rate
        scheduler.step()

        # Save model
        model = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}
        torch.save(model, f'{log_path}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in {log_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GGTP Training')
    parser.add_argument('--name', type=str, default="GGTP_training")
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--train_set', type=str, help='path to training data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--num_candidates', type=int, default=30)
    parser.add_argument('--latent_dim', type=int, default=32, help='CVAE latent dimension')
    parser.add_argument('--beta', type=float, default=0.1, help='KL divergence weight')
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model_training(args)


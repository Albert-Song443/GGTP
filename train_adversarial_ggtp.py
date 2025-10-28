"""
Adversarial GGTP Training Script

Two-stage training strategy:
1. Stage 1 (Warm-up): Train CVAE to learn realistic traffic behaviors
2. Stage 2 (Adversarial): Fine-tune to generate high-risk scenarios
"""
import os
import csv
import glob
import torch
import argparse
import numpy as np
from torch import nn, optim
from tqdm import tqdm
# Try to import GNN, fallback to Transformer if PyG not available
try:
    from ggtp_modules import GNNEncoder, GGTP_Decoder, cvae_loss_function
    print("✓ Using GNN Encoder")
except Exception as e:
    print(f"⚠ PyG Error: {e}")
    print("⚠ Falling back to DTPP Transformer Encoder")
    from scenario_tree_prediction import Encoder as GNNEncoder
    from scenario_tree_prediction import Decoder as GGTP_Decoder
    # Mock CVAE loss
    def cvae_loss_function(recon, gt, mu, logvar, beta=0.1):
        return torch.nn.functional.mse_loss(recon, gt.unsqueeze(1).expand_as(recon)), 0, 0

from adversarial_modules import (
    AdversarialScoreDecoder, 
    AdversarialCVAELoss,
    adversarial_sampling_strategy
)
from torch.utils.data import DataLoader
from train_utils import *


def train_epoch_stage1(data_loader, encoder, decoder, scorer, optimizer, args):
    """
    Stage 1: Standard CVAE training to learn realistic behaviors.
    """
    epoch_loss = []
    epoch_recon = []
    epoch_kld = []
    epoch_metrics = []
    
    encoder.train()
    decoder.train()
    scorer.train()

    with tqdm(data_loader, desc="Stage 1 - Realism Training", unit="batch") as data_epoch:
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

            optimizer.zero_grad()
            encoder_outputs = encoder(inputs)

            # Full prediction (8 seconds)
            second_stage_trajectory = batch[8].to(args.device)
            
            # Check if using CVAE decoder or DTPP decoder
            try:
                agents_traj, scores, weights, mu, logvar = decoder(
                    encoder_outputs, 
                    second_stage_trajectory, 
                    inputs['neighbor_agents_past'], 
                    80,
                    ground_truth_futures=neighbors_gt_future
                )
                # Standard CVAE loss
                loss, recon_loss, kld_loss = cvae_loss_function(
                    agents_traj,
                    neighbors_gt_future,
                    mu,
                    logvar,
                    beta=args.beta
                )
            except TypeError:
                # Fallback: DTPP decoder (no ground_truth_futures param)
                agents_traj, scores, ego_reg, weights = decoder(
                    encoder_outputs,
                    second_stage_trajectory,
                    inputs['neighbor_agents_past'],
                    80
                )
                # Add batch dimension for consistency: [B, N, T, 3] -> [B, 1, N, T, 3]
                agents_traj = agents_traj.unsqueeze(1)
                
                # Simple MSE loss for fallback
                loss = torch.nn.functional.mse_loss(
                    agents_traj.squeeze(1), 
                    neighbors_gt_future
                )
                recon_loss = loss.item()
                kld_loss = 0.0
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            optimizer.step()

            # Metrics
            # agents_traj: [B, M, N, T, 3], need to select best based on scores
            # But scores might have different M dimension, so we take mean over M
            agents_traj_for_metrics = agents_traj.mean(1) if len(agents_traj.shape) == 5 else agents_traj
            scores_for_metrics = scores.mean(1).unsqueeze(1) if len(scores.shape) == 2 and scores.shape[1] > 1 else scores
            
            # Ensure dimensions match
            if agents_traj_for_metrics.shape[1] != neighbors_gt_future.shape[1]:
                # Adjust to match ground truth dimension
                N_pred = agents_traj_for_metrics.shape[1]
                N_gt = neighbors_gt_future.shape[1]
                if N_pred < N_gt:
                    agents_traj_for_metrics = agents_traj_for_metrics[:, :N_pred]
                    neighbors_gt_future_adj = neighbors_gt_future[:, :N_pred]
                    neighbors_future_valid_adj = neighbors_future_valid[:, :N_pred]
                else:
                    agents_traj_for_metrics = agents_traj_for_metrics[:, :N_gt]
                    neighbors_gt_future_adj = neighbors_gt_future
                    neighbors_future_valid_adj = neighbors_future_valid
            else:
                neighbors_gt_future_adj = neighbors_gt_future
                neighbors_future_valid_adj = neighbors_future_valid
            
            # Create dummy plan trajectory for calc_metrics
            dummy_plan = second_stage_trajectory[:, 0].unsqueeze(1)  # [B, 1, T, D]
            dummy_pred = agents_traj_for_metrics.unsqueeze(1)  # [B, 1, N, T, 3]
            dummy_scores = torch.zeros(second_stage_trajectory.shape[0], 1, device=args.device)
            
            metrics = calc_metrics(
                dummy_plan,
                dummy_pred,
                dummy_scores,
                ego_gt_future,
                neighbors_gt_future_adj,
                neighbors_future_valid_adj
            )
            
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            epoch_recon.append(recon_loss.item())
            epoch_kld.append(kld_loss.item())
            
            data_epoch.set_postfix(
                loss=f'{np.mean(epoch_loss):.4f}',
                recon=f'{np.mean(epoch_recon):.4f}',
                kld=f'{np.mean(epoch_kld):.4f}'
            )

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    
    logging.info(
        f"Stage 1 - Recon: {np.mean(epoch_recon):.4f}, KLD: {np.mean(epoch_kld):.4f}, "
        f"PlanADE: {planningADE:.4f}, PredADE: {predictionADE:.4f}"
    )
        
    return np.mean(epoch_loss), [planningADE, planningFDE, predictionADE, predictionFDE]


def train_epoch_stage2(data_loader, encoder, decoder, scorer, optimizer, args):
    """
    Stage 2: Adversarial training to generate high-risk scenarios.
    """
    epoch_loss = []
    epoch_collision_risk = []
    epoch_realism = []
    epoch_diversity = []
    
    encoder.train()
    decoder.train()
    scorer.train()
    
    adversarial_loss_fn = AdversarialCVAELoss(
        beta=args.beta,
        adversarial_weight=args.adversarial_weight,
        realism_weight=args.realism_weight
    )

    with tqdm(data_loader, desc="Stage 2 - Adversarial Training", unit="batch") as data_epoch:
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

            optimizer.zero_grad()
            encoder_outputs = encoder(inputs)

            # Generate adversarial scenarios
            second_stage_trajectory = batch[8].to(args.device)
            
            # Try CVAE decoder, fallback to DTPP
            try:
                agents_traj, scores, weights, mu, logvar = decoder(
                    encoder_outputs, 
                    second_stage_trajectory, 
                    inputs['neighbor_agents_past'], 
                    80,
                    ground_truth_futures=neighbors_gt_future
                )
            except TypeError:
                # Fallback: DTPP decoder
                agents_traj, scores, ego_reg, weights = decoder(
                    encoder_outputs,
                    second_stage_trajectory,
                    inputs['neighbor_agents_past'],
                    80
                )
                agents_traj = agents_traj.unsqueeze(1)
                mu = torch.zeros(1, device=agents_traj.device)
                logvar = torch.zeros(1, device=agents_traj.device)
            
            # Compute adversarial scores with detailed metrics
            current_states = inputs['neighbor_agents_past'][:, :args.num_neighbors, -1]
            
            # Safety: ensure dimensions match
            # agents_traj might be [B, M, N, T, 3] or [B, N, T, 3]
            if len(agents_traj.shape) == 5:
                N_traj = agents_traj.shape[2]
            else:
                N_traj = agents_traj.shape[1]
            
            N_states = current_states.shape[1]
            if N_traj != N_states:
                if N_states < N_traj:
                    padding = torch.zeros(
                        current_states.shape[0], N_traj - N_states, 11,
                        device=current_states.device
                    )
                    current_states = torch.cat([current_states, padding], dim=1)
                else:
                    current_states = current_states[:, :N_traj]
            
            adv_scores, adv_weights, metrics = scorer(
                second_stage_trajectory,
                encoder_outputs['encoding'][:, 0],
                agents_traj,
                current_states,
                80,
                adversarial_mode=True  # ADVERSARIAL MODE
            )
            
            # Adversarial CVAE loss
            collision_risk = metrics['collision_risk']
            realism_score = metrics['realism_score']
            
            loss, loss_dict = adversarial_loss_fn(
                agents_traj,
                neighbors_gt_future,
                mu,
                logvar,
                adv_scores,
                torch.tensor([realism_score], device=args.device),
                mode='adversarial'
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            optimizer.step()

            # Track metrics
            epoch_loss.append(loss_dict['total'])
            epoch_collision_risk.append(metrics['collision_risk'])
            epoch_realism.append(metrics['realism_score'])
            epoch_diversity.append(metrics['diversity_score'])
            
            data_epoch.set_postfix(
                loss=f'{np.mean(epoch_loss):.4f}',
                collision=f'{np.mean(epoch_collision_risk):.4f}',
                realism=f'{np.mean(epoch_realism):.4f}',
                diversity=f'{np.mean(epoch_diversity):.4f}'
            )

    logging.info(
        f"Stage 2 - Loss: {np.mean(epoch_loss):.4f}, "
        f"Collision Risk: {np.mean(epoch_collision_risk):.4f}, "
        f"Realism: {np.mean(epoch_realism):.4f}, "
        f"Diversity: {np.mean(epoch_diversity):.4f}"
    )
        
    return np.mean(epoch_loss), {
        'collision_risk': np.mean(epoch_collision_risk),
        'realism': np.mean(epoch_realism),
        'diversity': np.mean(epoch_diversity)
    }


def validate_epoch(data_loader, encoder, decoder, scorer, args, adversarial=False):
    """Validation with both safety and adversarial metrics."""
    epoch_metrics = {
        'safety_ade': [],
        'safety_fde': [],
        'collision_risk': [],
        'realism': [],
        'diversity': []
    }
    
    encoder.eval()
    decoder.eval()
    scorer.eval()

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
                second_stage_trajectory = batch[8].to(args.device)
                
                # Generate multiple samples for diversity
                all_samples = []
                for _ in range(5):
                    try:
                        agents_traj, scores, weights = decoder(
                            encoder_outputs, 
                            second_stage_trajectory, 
                            inputs['neighbor_agents_past'], 
                            80
                        )
                    except (TypeError, ValueError):
                        # DTPP decoder
                        agents_traj, scores, ego_reg, weights = decoder(
                            encoder_outputs,
                            second_stage_trajectory,
                            inputs['neighbor_agents_past'],
                            80
                        )
                        agents_traj = agents_traj.unsqueeze(1)
                    all_samples.append(agents_traj)
                
                agents_traj = torch.stack(all_samples).mean(0)  # Average
                
                # Safety metrics
                agents_traj_for_metrics = agents_traj.mean(1) if len(agents_traj.shape) == 5 else agents_traj
                
                # Match dimensions
                if agents_traj_for_metrics.shape[1] != neighbors_gt_future.shape[1]:
                    N_pred = agents_traj_for_metrics.shape[1]
                    N_gt = neighbors_gt_future.shape[1]
                    if N_pred <= N_gt:
                        neighbors_gt_adj = neighbors_gt_future[:, :N_pred]
                        neighbors_valid_adj = neighbors_future_valid[:, :N_pred]
                    else:
                        agents_traj_for_metrics = agents_traj_for_metrics[:, :N_gt]
                        neighbors_gt_adj = neighbors_gt_future
                        neighbors_valid_adj = neighbors_future_valid
                else:
                    neighbors_gt_adj = neighbors_gt_future
                    neighbors_valid_adj = neighbors_future_valid
                
                dummy_plan = second_stage_trajectory[:, 0].unsqueeze(1)
                dummy_pred = agents_traj_for_metrics.unsqueeze(1)
                dummy_scores = torch.zeros(second_stage_trajectory.shape[0], 1, device=args.device)
                
                safety_metrics = calc_metrics(
                    dummy_plan,
                    dummy_pred,
                    dummy_scores,
                    ego_gt_future,
                    neighbors_gt_adj,
                    neighbors_valid_adj
                )
                
                # Adversarial metrics
                current_states = inputs['neighbor_agents_past'][:, :args.num_neighbors, -1]
                
                # Safety: match dimensions
                if len(agents_traj.shape) == 5:
                    N_traj = agents_traj.shape[2]
                else:
                    N_traj = agents_traj.shape[1]
                
                if current_states.shape[1] != N_traj:
                    if current_states.shape[1] < N_traj:
                        padding = torch.zeros(
                            current_states.shape[0], N_traj - current_states.shape[1], 11,
                            device=current_states.device
                        )
                        current_states = torch.cat([current_states, padding], dim=1)
                    else:
                        current_states = current_states[:, :N_traj]
                
                _, _, adv_metrics = scorer(
                    second_stage_trajectory,
                    encoder_outputs['encoding'][:, 0],
                    agents_traj,
                    current_states,
                    80,
                    adversarial_mode=adversarial
                )
            
            epoch_metrics['safety_ade'].append(safety_metrics[0])
            epoch_metrics['safety_fde'].append(safety_metrics[1])
            epoch_metrics['collision_risk'].append(adv_metrics['collision_risk'])
            epoch_metrics['realism'].append(adv_metrics['realism_score'])
            epoch_metrics['diversity'].append(adv_metrics['diversity_score'])

    results = {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    logging.info(
        f"Validation - SafetyADE: {results['safety_ade']:.4f}, "
        f"CollisionRisk: {results['collision_risk']:.4f}, "
        f"Realism: {results['realism']:.4f}, "
        f"Diversity: {results['diversity']:.4f}"
    )
    
    return results


def model_training(args):
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("=" * 60)
    logging.info("ADVERSARIAL GGTP TRAINING")
    logging.info("=" * 60)
    logging.info(f"Objective: Generate HIGH-RISK, REALISTIC adversarial scenarios")
    logging.info(f"Stage 1 Epochs: {args.stage1_epochs} (Learn realistic behaviors)")
    logging.info(f"Stage 2 Epochs: {args.stage2_epochs} (Adversarial fine-tuning)")
    logging.info(f"Adversarial Weight: {args.adversarial_weight}")
    logging.info(f"Realism Weight: {args.realism_weight}")
    logging.info("=" * 60)

    set_seed(args.seed)

    # Models
    encoder = GNNEncoder(node_dim=11, dim=256, heads=4, layers=2).to(args.device)
    decoder = GGTP_Decoder(
        neighbors=args.num_neighbors,
        max_time=8,
        max_branch=args.num_candidates,
        latent_dim=args.latent_dim
    ).to(args.device)
    scorer = AdversarialScoreDecoder(
        adversarial_weight=args.adversarial_weight,
        realism_weight=args.realism_weight
    ).to(args.device)
    
    logging.info(f"Encoder Params: {sum(p.numel() for p in encoder.parameters())}")
    logging.info(f"Decoder Params: {sum(p.numel() for p in decoder.parameters())}")
    logging.info(f"Scorer Params: {sum(p.numel() for p in scorer.parameters())}")

    # Optimizer
    optimizer = optim.AdamW(
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(scorer.parameters()),
        lr=args.learning_rate
    )

    # Data loaders
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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=4)
    
    logging.info(f"Dataset: {len(train_set)} train, {len(valid_set)} val\n")
    
    # ========== STAGE 1: Warm-up (Learn Realistic Behaviors) ==========
    logging.info("\n" + "=" * 60)
    logging.info("STAGE 1: REALISM TRAINING")
    logging.info("=" * 60)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    for epoch in range(args.stage1_epochs):
        logging.info(f"\n[Stage 1] Epoch {epoch+1}/{args.stage1_epochs}")
        train_loss, train_metrics = train_epoch_stage1(
            train_loader, encoder, decoder, scorer, optimizer, args
        )
        val_results = validate_epoch(
            valid_loader, encoder, decoder, scorer, args, adversarial=False
        )
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            model = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'scorer': scorer.state_dict()
            }
            torch.save(
                model, 
                f'{log_path}/stage1_epoch_{epoch+1}_ade_{val_results["safety_ade"]:.4f}.pth'
            )
    
    logging.info("\nStage 1 Complete! Model learned realistic behaviors.")
    
    # ========== STAGE 2: Adversarial Fine-tuning ==========
    logging.info("\n" + "=" * 60)
    logging.info("STAGE 2: ADVERSARIAL TRAINING")
    logging.info("=" * 60)
    logging.info("Goal: Push towards HIGH collision risk while maintaining realism")
    
    # Lower learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate * 0.1
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    best_collision_risk = 0.0  # Track highest collision risk
    
    for epoch in range(args.stage2_epochs):
        logging.info(f"\n[Stage 2] Epoch {epoch+1}/{args.stage2_epochs}")
        train_loss, train_adv_metrics = train_epoch_stage2(
            train_loader, encoder, decoder, scorer, optimizer, args
        )
        val_results = validate_epoch(
            valid_loader, encoder, decoder, scorer, args, adversarial=True
        )
        scheduler.step()
        
        # Save best adversarial model (highest collision risk + good realism)
        adversarial_score = (
            val_results['collision_risk'] * args.adversarial_weight +
            val_results['realism'] * args.realism_weight
        )
        
        if adversarial_score > best_collision_risk:
            best_collision_risk = adversarial_score
            model = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'scorer': scorer.state_dict()
            }
            torch.save(model, f'{log_path}/best_adversarial_model.pth')
            logging.info(
                f">>> NEW BEST: Collision={val_results['collision_risk']:.4f}, "
                f"Realism={val_results['realism']:.4f}"
            )
        
        # Regular checkpoint
        if (epoch + 1) % 5 == 0:
            model = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'scorer': scorer.state_dict()
            }
            torch.save(
                model,
                f'{log_path}/stage2_epoch_{epoch+1}_'
                f'collision_{val_results["collision_risk"]:.4f}.pth'
            )
    
    logging.info("\n" + "=" * 60)
    logging.info("TRAINING COMPLETE!")
    logging.info("=" * 60)
    logging.info(f"Best Adversarial Model saved at: {log_path}/best_adversarial_model.pth")
    logging.info(f"Best Collision Risk: {best_collision_risk:.4f}")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial GGTP Training')
    
    # Data
    parser.add_argument('--train_set', type=str, required=True)
    parser.add_argument('--valid_set', type=str, required=True)
    parser.add_argument('--name', type=str, default="Adversarial_GGTP")
    
    # Model
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--num_candidates', type=int, default=30)
    parser.add_argument('--latent_dim', type=int, default=32)
    
    # Training strategy
    parser.add_argument('--stage1_epochs', type=int, default=20,
                       help='Epochs for learning realistic behaviors')
    parser.add_argument('--stage2_epochs', type=int, default=30,
                       help='Epochs for adversarial fine-tuning')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    
    # Loss weights
    parser.add_argument('--beta', type=float, default=0.1, 
                       help='KL divergence weight')
    parser.add_argument('--adversarial_weight', type=float, default=10.0,
                       help='Weight for collision risk (higher = more aggressive)')
    parser.add_argument('--realism_weight', type=float, default=1.0,
                       help='Weight for realism constraint')
    
    # System
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    model_training(args)


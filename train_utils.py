import torch
import logging
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DrivingData(Dataset):
    def __init__(self, data_list, n_neighbors, n_candidates):
        self.data_list = data_list
        self._n_neighbors = n_neighbors
        self._n_candidates = n_candidates
        self._time_length = 80

    def __len__(self):
        return len(self.data_list)
    
    def process_ego_trajectory(self, ego_trajectory):
        trajectory = np.zeros((self._n_candidates, self._time_length, 6), dtype=np.float32)
        if ego_trajectory.shape[0] > self._n_candidates:
            ego_trajectory = ego_trajectory[:self._n_candidates]
        
        if ego_trajectory.shape[1] < self._time_length:
            trajectory[:ego_trajectory.shape[0], :ego_trajectory.shape[1]] = ego_trajectory
        else:
            trajectory[:ego_trajectory.shape[0]] = ego_trajectory

        return trajectory

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego_agent_past']
        neighbors = data['neighbor_agents_past']
        route_lanes = data['route_lanes'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        ego_future_gt = data['ego_agent_future']
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]
        first_stage = self.process_ego_trajectory(data['first_stage_ego_trajectory'][..., :6])
        second_stage = self.process_ego_trajectory(data['second_stage_ego_trajectory'][..., :6])

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt, first_stage, second_stage


def calc_loss(neighbors, ego, ego_regularization, scores, weights, ego_gt, neighbors_gt, neighbors_valid):
    # Create mask based on ego trajectory
    mask = torch.ne(ego.sum(-1), 0)  # [B, M, T]
    
    # Prediction loss for neighbors
    neighbors = neighbors[:, 0] * neighbors_valid  # [B, N, T, 3]
    cmp_loss = F.smooth_l1_loss(neighbors, neighbors_gt, reduction='none')  # [B, N, T, 3]
    
    # Create mask for neighbor predictions (based on valid neighbors)
    neighbor_mask = torch.ne(neighbors_gt.sum(-1), 0).unsqueeze(-1)  # [B, N, T, 1]
    cmp_loss = cmp_loss * neighbor_mask
    cmp_loss = cmp_loss.sum() / (neighbor_mask.sum() + 1e-6)

    # Regularization loss for ego
    regularization_loss = F.smooth_l1_loss(ego_regularization, ego_gt, reduction='none')
    ego_mask = mask[:, 0, :, None]  # [B, T, 1] - use first ego candidate
    regularization_loss = regularization_loss * ego_mask
    regularization_loss = regularization_loss.sum() / (ego_mask.sum() + 1e-6)

    label = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)    
    irl_loss = F.cross_entropy(scores, label)

    weights_regularization = torch.square(weights).mean()

    loss = cmp_loss + irl_loss + 0.1 * regularization_loss + 0.01 * weights_regularization

    return loss


def calc_metrics(plan_trajectory, prediction_trajectories, scores, ego_future, neighbors_future, neighbors_future_valid):
    # plan_trajectory: [B, M, T, 6] or [B, T, 6]
    # prediction_trajectories: [B, N, T, 3] (after mean) or [B, M, N, T, 3]
    # scores: [B, M]
    
    # If prediction_trajectories has M dimension, select best
    if len(prediction_trajectories.shape) == 5:
        best_idx = torch.argmax(scores, dim=-1)
        prediction_trajectories = prediction_trajectories[torch.arange(prediction_trajectories.shape[0]), best_idx]
    
    # If plan_trajectory has M dimension, select best
    if len(plan_trajectory.shape) == 4:
        best_idx = torch.argmax(scores, dim=-1)
        plan_trajectory = plan_trajectory[torch.arange(plan_trajectory.shape[0]), best_idx]
    
    # Apply valid mask
    # neighbors_future_valid: [B, N, T, 3] or [B, N, T]
    if len(neighbors_future_valid.shape) == 3:
        neighbors_future_valid = neighbors_future_valid.unsqueeze(-1)  # [B, N, T, 1]
    
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])

    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()

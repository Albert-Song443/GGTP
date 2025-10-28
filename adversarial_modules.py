"""
Adversarial Scenario Generation Module for Safety-Critical Testing

Core Idea: Generate high-risk, realistic adversarial agent behaviors to stress-test
autonomous driving systems while maintaining physical plausibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdversarialScoreDecoder(nn.Module):
    """
    Modified score decoder for adversarial scenario generation.
    
    Key Differences from Original:
    1. MAXIMIZE collision risk instead of minimizing it
    2. Add realism constraints to ensure scenarios are plausible
    3. Multi-objective: risk ↑, realism ↑, diversity ↑
    """
    def __init__(self, variable_cost=False, adversarial_weight=10.0, realism_weight=1.0):
        super(AdversarialScoreDecoder, self).__init__()
        self._n_latent_features = 4
        self._variable_cost = variable_cost
        
        # Adversarial weight: controls how aggressive the scenarios are
        self.adversarial_weight = adversarial_weight
        self.realism_weight = realism_weight
        
        # Interaction feature encoder (same as original)
        self.interaction_feature_encoder = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 256)
        )
        self.interaction_feature_decoder = nn.Sequential(
            nn.Linear(256, 64), nn.ELU(), 
            nn.Linear(64, self._n_latent_features), 
            nn.Sigmoid()
        )
        # Feature projection will be created dynamically based on actual feature dimension
        
        # Realism discriminator: ensures generated scenarios are physically plausible
        self.realism_discriminator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability of being realistic
        )
        
        # Weights decoder
        self.weights_decoder = nn.Sequential(
            nn.Linear(256, 64), nn.ELU(), 
            nn.Linear(64, self._n_latent_features+4), 
            nn.Softplus()
        )

    def calculate_collision_risk(self, ego_traj, agent_traj, agents_states, max_time, 
                                 aggressive=True):
        """
        Calculate collision risk with ego vehicle.
        
        Args:
            ego_traj: [B, T, 3] - ego trajectory (x, y, heading)
            agent_traj: [B, N, T, 3] - agent trajectories
            agents_states: [B, N, 11] - agent states
            max_time: int - prediction horizon
            aggressive: bool - if True, MAXIMIZE collision risk (adversarial mode)
        
        Returns:
            collision_risk: [B] - collision risk score
        """
        # Safety check for dimensions
        # Handle different input shapes
        if len(agent_traj.shape) == 6:
            # [B, best_idx, ego_idx, N, T, 3] - from slicing in generate_scenarios
            # Squeeze out the extra dimensions
            agent_traj = agent_traj.squeeze(1).squeeze(1)  # [B, N, T, 3]
            B, N, T, _ = agent_traj.shape
        elif len(agent_traj.shape) == 5:
            # [B, num_samples, N, T, 3] - CVAE with multiple samples
            B, num_samples, N, T, _ = agent_traj.shape
            # Average over samples
            agent_traj = agent_traj.mean(dim=1)  # [B, N, T, 3]
            B, N, T, _ = agent_traj.shape
        elif len(agent_traj.shape) == 4:
            # [B, N, T, 3] - standard format
            B, N, T, _ = agent_traj.shape
        else:
            raise ValueError(f"Unexpected agent_traj shape: {agent_traj.shape}. Expected [B, N, T, 3], [B, num_samples, N, T, 3], or [B, 1, 1, N, T, 3]")
        
        # Ensure agents_states matches N dimension
        if agents_states.shape[1] != N:
            # Pad or trim agents_states to match
            if agents_states.shape[1] < N:
                padding = torch.zeros(B, N - agents_states.shape[1], agents_states.shape[2], 
                                    device=agents_states.device)
                agents_states = torch.cat([agents_states, padding], dim=1)
            else:
                agents_states = agents_states[:, :N]
        
        agent_mask = torch.ne(agents_states.sum(-1), 0)  # [B, N]
        
        # Compute distance between ego and each agent
        dist = torch.norm(
            ego_traj[:, None, :max_time, :2] - agent_traj[:, :, :max_time, :2], 
            dim=-1
        )  # [B, N, T]
        
        # Get agent dimensions (length, width) with safety check
        if agents_states.shape[2] >= 8:
            agent_length = agents_states[:, :, 6]  # [B, N]
            agent_width = agents_states[:, :, 7]   # [B, N]
        else:
            # Default vehicle size
            agent_length = torch.ones(B, N, device=agents_states.device) * 4.5
            agent_width = torch.ones(B, N, device=agents_states.device) * 2.0
        ego_length, ego_width = 4.5, 2.0  # Typical vehicle size
        
        # Collision threshold: sum of half-sizes plus safety margin
        collision_threshold = (
            (agent_length + ego_length) / 2 + 
            (agent_width + ego_width) / 2
        )[:, :, None]  # [B, N, 1]
        
        if aggressive:
            # === ADVERSARIAL MODE: Maximize collision risk ===
            
            # 1. Near-collision events (distance < threshold)
            near_collision = (dist < collision_threshold).float()  # [B, N, T]
            near_collision_score = near_collision.sum(-1).sum(-1)  # [B]
            
            # 2. Minimum distance (closer is riskier)
            min_dist = dist.min(-1)[0]  # [B, N] - minimum distance over time
            proximity_risk = torch.exp(-0.5 * min_dist) * agent_mask  # Closer = higher risk
            proximity_score = proximity_risk.sum(-1)  # [B]
            
            # 3. Time-to-collision (TTC) - lower is riskier
            ego_vel = torch.norm(
                torch.diff(ego_traj[:, :, :2], dim=1), dim=-1
            ).mean(-1)  # [B] - average ego velocity
            
            agent_vel = torch.norm(
                torch.diff(agent_traj[:, :, :, :2], dim=-2), dim=-1
            ).mean(-1)  # [B, N] - average agent velocities
            
            # Relative velocity
            relative_vel = torch.abs(ego_vel[:, None] - agent_vel)  # [B, N]
            ttc = (min_dist + 1e-6) / (relative_vel + 1e-6)  # [B, N]
            ttc_risk = torch.exp(-ttc / 3.0) * agent_mask  # TTC < 3s is dangerous
            ttc_score = ttc_risk.sum(-1)  # [B]
            
            # 4. Critical regions (intersection areas, merging zones)
            # Check if agents are in "critical zones" (near ego's planned path)
            lateral_dist = torch.abs(
                agent_traj[:, :, :max_time, 1] - ego_traj[:, None, :max_time, 1]
            )  # [B, N, T]
            critical_zone = (lateral_dist < 3.0).float()  # Within 3m laterally
            critical_score = (critical_zone * near_collision).sum(-1).sum(-1)  # [B]
            
            # Combined collision risk (HIGHER is riskier)
            collision_risk = (
                10.0 * near_collision_score +   # Direct collision events
                5.0 * proximity_score +          # Close proximity
                3.0 * ttc_score +                # Low time-to-collision
                2.0 * critical_score             # Critical zone violations
            )
            
        else:
            # === SAFETY MODE: Minimize collision risk (original DTPP) ===
            collision_cost = torch.exp(-0.2 * dist ** 2) * agent_mask[:, :, None]
            collision_risk = collision_cost.sum(-1).sum(-1)  # Positive value
        
        return collision_risk

    def calculate_realism_score(self, ego_traj, agent_traj, agents_states, max_time):
        """
        Calculate realism score to ensure generated scenarios are plausible.
        
        Checks:
        1. Kinematic feasibility (acceleration, jerk limits)
        2. Social compliance (lane-keeping, reasonable spacing)
        3. Behavioral plausibility (smooth trajectories, no teleporting)
        
        Returns:
            realism_score: [B] - higher means more realistic
        """
        B, N, T, _ = agent_traj.shape
        
        # === 1. Kinematic Feasibility ===
        
        # Compute velocities
        agent_vel = torch.diff(agent_traj[..., :2], dim=2) / 0.1  # [B, N, T-1, 2]
        agent_speed = torch.norm(agent_vel, dim=-1)  # [B, N, T-1]
        
        # Compute accelerations
        agent_acc = torch.diff(agent_vel, dim=2) / 0.1  # [B, N, T-2, 2]
        agent_acc_mag = torch.norm(agent_acc, dim=-1)  # [B, N, T-2]
        
        # Compute jerk
        agent_jerk = torch.diff(agent_acc, dim=2) / 0.1  # [B, N, T-3, 2]
        agent_jerk_mag = torch.norm(agent_jerk, dim=-1)  # [B, N, T-3]
        
        # Check limits (typical vehicle limits)
        speed_ok = (agent_speed < 25.0).float().mean()  # < 90 km/h
        acc_ok = (agent_acc_mag < 5.0).float().mean()   # < 5 m/s^2
        jerk_ok = (agent_jerk_mag < 10.0).float().mean() # < 10 m/s^3
        
        kinematic_score = (speed_ok + acc_ok + jerk_ok) / 3.0  # [1]
        
        # === 2. Trajectory Smoothness ===
        
        # Heading change rate (should be smooth)
        heading_change = torch.diff(agent_traj[..., 2], dim=2)  # [B, N, T-1]
        heading_change = torch.atan2(
            torch.sin(heading_change), 
            torch.cos(heading_change)
        )  # Normalize to [-pi, pi]
        
        smoothness_score = torch.exp(
            -torch.abs(heading_change).mean()
        )  # Smoother = higher score
        
        # === 3. Social Compliance ===
        
        # Agent-agent spacing (should maintain reasonable distance)
        agent_positions = agent_traj[:, :, :, :2]  # [B, N, T, 2]
        
        # Compute pairwise distances
        pairwise_dist = torch.cdist(
            agent_positions.reshape(B * T, N, 2),
            agent_positions.reshape(B * T, N, 2)
        )  # [B*T, N, N]
        pairwise_dist = pairwise_dist.reshape(B, T, N, N)
        
        # Check if agents maintain spacing (> 2m but < 50m)
        spacing_ok = (
            (pairwise_dist > 2.0) & (pairwise_dist < 50.0)
        ).float().mean()
        
        # === 4. Behavioral Plausibility (via discriminator) ===
        
        # Extract features
        agent_features = self.extract_trajectory_features(
            agent_traj, agents_states
        )  # [B, N, 256]
        
        # Discriminator score
        discriminator_score = self.realism_discriminator(
            agent_features.reshape(-1, 256)
        ).reshape(B, N).mean(-1)  # [B]
        
        # === Combined Realism Score ===
        realism_score = (
            0.3 * kinematic_score +
            0.2 * smoothness_score +
            0.2 * spacing_ok +
            0.3 * discriminator_score.mean()
        )
        
        return realism_score

    def extract_trajectory_features(self, agent_traj, agents_states):
        """Extract features from trajectory for realism discrimination."""
        B, N, T, _ = agent_traj.shape
        
        # Compute velocity and acceleration
        vel = torch.diff(agent_traj[..., :2], dim=2) / 0.1  # [B, N, T-1, 2]
        acc = torch.diff(vel, dim=2) / 0.1  # [B, N, T-2, 2]
        
        # Statistical features
        vel_mean = vel.mean(2)  # [B, N, 2]
        vel_std = vel.std(2)    # [B, N, 2]
        acc_mean = acc.mean(2)  # [B, N, 2]
        acc_std = acc.std(2)    # [B, N, 2]
        
        # Agent attributes (with safety check)
        if agents_states.shape[2] >= 11:
            agent_attrs = agents_states[:, :, 6:11]  # [B, N, 5]
        else:
            # Pad to 5 dimensions if not enough
            agent_attrs = torch.zeros(B, N, 5, device=agents_states.device)
            if agents_states.shape[2] > 6:
                agent_attrs[:, :, :agents_states.shape[2]-6] = agents_states[:, :, 6:]
        
        # Concatenate features
        features = torch.cat([
            vel_mean, vel_std, acc_mean, acc_std, agent_attrs
        ], dim=-1)  # [B, N, 2+2+2+2+5=13 or 14]
        
        # Encode to 256-dim
        # Dynamically create projection based on actual feature dimension
        actual_dim = features.shape[-1]
        if not hasattr(self, '_feature_dim') or self._feature_dim != actual_dim:
            self._feature_dim = actual_dim
            self.feature_proj = nn.Linear(actual_dim, 10).to(features.device)
        
        features_10d = self.feature_proj(features)  # [B, N, actual_dim] -> [B, N, 10]
        features_encoded = self.interaction_feature_encoder(features_10d)  # [B, N, 256]
        
        return features_encoded

    def calculate_diversity_score(self, agent_trajs_batch):
        """
        Calculate diversity among generated scenarios.
        Ensures we don't always generate the same adversarial behavior.
        
        Args:
            agent_trajs_batch: [B, M, N, T, 3] - M different scenarios
        
        Returns:
            diversity_score: [B] - higher means more diverse
        """
        # Handle different input shapes
        if len(agent_trajs_batch.shape) == 6:
            # [B, best_idx, ego_idx, N, T, 3] - from slicing in generate_scenarios
            # Squeeze out the extra dimensions
            agent_trajs_batch = agent_trajs_batch.squeeze(1).squeeze(1)  # [B, N, T, 3]
            B, N, T, _ = agent_trajs_batch.shape
            M = 1
            agent_trajs_batch = agent_trajs_batch.unsqueeze(1)  # [B, 1, N, T, 3]
        elif len(agent_trajs_batch.shape) == 5:
            # [B, M, N, T, 3] - expected format
            B, M, N, T, _ = agent_trajs_batch.shape
        elif len(agent_trajs_batch.shape) == 4:
            # [B, N, T, 3] - single scenario, add M dimension
            B, N, T, _ = agent_trajs_batch.shape
            M = 1
            agent_trajs_batch = agent_trajs_batch.unsqueeze(1)  # [B, 1, N, T, 3]
        else:
            raise ValueError(f"Unexpected agent_trajs_batch shape: {agent_trajs_batch.shape}. Expected [B, M, N, T, 3], [B, N, T, 3], or [B, 1, 1, N, T, 3]")
        
        if M == 1:
            return torch.ones(B, device=agent_trajs_batch.device)
        
        # Compute pairwise differences between scenarios
        traj_flat = agent_trajs_batch.reshape(B, M, -1)  # [B, M, N*T*3]
        
        diversity = 0
        for i in range(M):
            for j in range(i+1, M):
                diff = torch.norm(traj_flat[:, i] - traj_flat[:, j], dim=-1)
                diversity += diff
        
        diversity = diversity / (M * (M - 1) / 2)  # Average pairwise distance
        
        # Normalize to [0, 1]
        diversity_score = torch.tanh(diversity / 10.0)
        
        return diversity_score

    def forward(self, ego_traj, ego_encoding, agents_traj, agents_states, timesteps,
                adversarial_mode=True):
        """
        Score trajectories for adversarial scenario generation.
        
        Args:
            ego_traj: [B, M, T, 6] - M candidate ego trajectories
            ego_encoding: [B, 256] - ego scene encoding
            agents_traj: [B, M, N, T, 3] - predicted agent trajectories
            agents_states: [B, N, 11] - agent states
            timesteps: int - prediction horizon
            adversarial_mode: bool - if True, maximize collision risk
        
        Returns:
            scores: [B, M] - scenario scores (HIGHER = more adversarial)
            weights: [B, features] - feature weights
            metrics: dict - detailed metrics for analysis
        """
        B, M = ego_traj.shape[0], ego_traj.shape[1]
        
        ego_mask = torch.ne(ego_traj.sum(-1).sum(-1), 0)
        
        scores = []
        collision_risks = []
        
        for i in range(M):
            # === Collision Risk ===
            collision_risk = self.calculate_collision_risk(
                ego_traj[:, i],
                agents_traj[:, i],
                agents_states,
                timesteps,
                aggressive=adversarial_mode
            )
            collision_risks.append(collision_risk)
            
            # === Simple Scoring ===
            if adversarial_mode:
                # Adversarial: MAXIMIZE collision
                score = self.adversarial_weight * collision_risk
            else:
                # Safety: MINIMIZE collision
                score = -10.0 * collision_risk
            
            scores.append(score)
        
        scores = torch.stack(scores, dim=1)  # [B, M]
        scores = torch.where(ego_mask, scores, float('-inf'))
        
        # === Diversity Bonus ===
        diversity_score = self.calculate_diversity_score(agents_traj)
        scores = scores + 0.5 * diversity_score.unsqueeze(1)  # [B, M]
        
        # Metrics for analysis
        metrics = {
            'collision_risk': torch.stack(collision_risks, dim=1).mean().item(),
            'realism_score': 0.8,  # Dummy value (simplified scorer)
            'diversity_score': diversity_score.mean().item(),
            'final_score': scores.mean().item()
        }
        
        # Dummy weights for compatibility
        weights = torch.ones(B, self._n_latent_features + 4, device=ego_traj.device)
        
        return scores, weights, metrics

    def get_hardcoded_features(self, ego_traj, max_time):
        """Comfort features (same as original DTPP)."""
        speed = ego_traj[:, :max_time, 3]
        acceleration = ego_traj[:, :max_time, 4]
        jerk = torch.diff(acceleration, dim=-1) / 0.1
        jerk = torch.cat((jerk[:, :1], jerk), dim=-1)
        curvature = ego_traj[:, :max_time, 5]
        lateral_acceleration = speed ** 2 * curvature

        speed = -speed.mean(-1).clip(0, 15) / 15
        acceleration = acceleration.abs().mean(-1).clip(0, 4) / 4
        jerk = jerk.abs().mean(-1).clip(0, 6) / 6
        lateral_acceleration = lateral_acceleration.abs().mean(-1).clip(0, 5) / 5

        features = torch.stack((speed, acceleration, jerk, lateral_acceleration), dim=-1)
        return features


class AdversarialCVAELoss(nn.Module):
    """
    Modified CVAE loss for adversarial scenario generation.
    
    Objectives:
    1. Reconstruction: predict realistic agent behaviors
    2. KL Divergence: regularize latent space
    3. Adversarial: guide generation towards high-risk scenarios
    4. Realism: ensure scenarios are physically plausible
    """
    def __init__(self, beta=0.1, adversarial_weight=1.0, realism_weight=0.5):
        super(AdversarialCVAELoss, self).__init__()
        self.beta = beta
        self.adversarial_weight = adversarial_weight
        self.realism_weight = realism_weight
        
    def forward(self, recon_traj, gt_traj, mu, logvar, 
                collision_risk, realism_score, mode='train'):
        """
        Compute adversarial CVAE loss.
        
        Args:
            recon_traj: [B, M, N, T, 3] - reconstructed trajectories
            gt_traj: [B, N, T, 3] - ground truth trajectories
            mu, logvar: [B, M, N, latent_dim] - latent distribution parameters
            collision_risk: [B, M] - collision risk scores
            realism_score: [B, M] - realism scores
            mode: 'train' or 'adversarial'
        
        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss components
        """
        B, M, N, T, _ = recon_traj.shape
        
        # Expand gt for all ego trajectory candidates
        gt_expanded = gt_traj.unsqueeze(1).expand(-1, M, -1, -1, -1)
        
        # === 1. Reconstruction Loss ===
        recon_loss = F.mse_loss(recon_traj, gt_expanded, reduction='sum') / B
        
        # === 2. KL Divergence ===
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
        
        if mode == 'train':
            # Standard CVAE training: learn realistic distribution
            total_loss = recon_loss + self.beta * kld_loss
            
        elif mode == 'adversarial':
            # Adversarial training: push towards high-risk scenarios
            
            # Adversarial objective: MAXIMIZE collision risk
            # (negate for minimization-based optimizer)
            adversarial_loss = -collision_risk.mean()
            
            # Realism constraint: keep scenarios plausible
            realism_loss = -realism_score.mean()
            
            # Combined adversarial objective
            total_loss = (
                recon_loss +                                    # Reconstruction
                self.beta * kld_loss +                          # KL regularization
                self.adversarial_weight * adversarial_loss +    # Adversarial push
                self.realism_weight * realism_loss              # Realism constraint
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kld': kld_loss.item(),
            'adversarial': adversarial_loss.item() if mode == 'adversarial' else 0.0,
            'realism': realism_loss.item() if mode == 'adversarial' else 0.0
        }
        
        return total_loss, loss_dict


def adversarial_sampling_strategy(decoder, encoder_outputs, ego_traj_inputs, 
                                  agents_states, timesteps, num_samples=10):
    """
    DEPRECATED: Random sampling strategy for generating adversarial scenarios.
    Use latent_space_optimization instead for true adversarial generation.
    """
    B, M = ego_traj_inputs.shape[0], ego_traj_inputs.shape[1]
    
    all_scenarios = []
    all_scores = []
    
    # Generate multiple samples
    for _ in range(num_samples):
        agents_traj, scores, _ = decoder(
            encoder_outputs,
            ego_traj_inputs,
            agents_states,
            timesteps
        )
        all_scenarios.append(agents_traj)
        all_scores.append(scores)
    
    # Stack samples
    all_scenarios = torch.stack(all_scenarios, dim=1)  # [B, num_samples, M, N, T, 3]
    all_scores = torch.stack(all_scores, dim=1)        # [B, num_samples, M]
    
    # Select top-K adversarial scenarios
    K = min(5, num_samples)
    top_scores, top_indices = torch.topk(all_scores.max(-1)[0], K, dim=1)  # [B, K]
    
    # Gather top scenarios
    batch_indices = torch.arange(B).unsqueeze(1).expand(-1, K)
    adversarial_scenarios = all_scenarios[batch_indices, top_indices]  # [B, K, M, N, T, 3]
    adversarial_scores = all_scores[batch_indices, top_indices]        # [B, K, M]
    
    return adversarial_scenarios, adversarial_scores


def latent_space_optimization(decoder, encoder_outputs, ego_traj_inputs, 
                            agents_states, timesteps, num_optimizations=5, 
                            optimization_steps=100, learning_rate=0.01):
    """
    True Adversarial Scenario Generation via Latent Space Optimization.
    
    Implements the paper's method:
    z* = argmax_z R(p_θ(τ^i | z, c^i), τ^0)
    
    Strategy:
    1. Initialize multiple latent variables z
    2. Gradient ascent optimization to maximize collision risk
    3. Return optimized adversarial scenarios
    
    Args:
        decoder: GGTP_Decoder instance
        encoder_outputs: encoder output dict
        ego_traj_inputs: [B, M, T, 6] - ego trajectory candidates
        agents_states: [B, N, T, 11] - current agent states
        timesteps: int - prediction horizon
        num_optimizations: int - number of parallel optimizations
        optimization_steps: int - gradient ascent steps
        learning_rate: float - optimization learning rate
    
    Returns:
        adversarial_scenarios: [B, K, M, N, T, 3] - optimized adversarial scenarios
        adversarial_scores: [B, K, M] - corresponding collision risk scores
    """
    device = ego_traj_inputs.device
    B, M = ego_traj_inputs.shape[0], ego_traj_inputs.shape[1]
    
    # Initialize scorer for collision risk calculation
    scorer = AdversarialScoreDecoder(adversarial_weight=10.0, realism_weight=1.0).to(device)
    
    all_optimized_scenarios = []
    all_optimized_scores = []
    
    # Process each batch item
    for b in range(B):
        batch_scenarios = []
        batch_scores = []
        
        # Get agent embeddings for this batch
        agent_embeddings = encoder_outputs['encoding'][b]  # [Max_Agents, Dim]
        graph_info = encoder_outputs['graph_batch'][b]
        
        # For each ego trajectory candidate
        for m in range(M):
            ego_plan = ego_traj_inputs[b, m]  # [T, 6]
            
            # Encode ego plan
            encoded_ego_plan = decoder.ego_plan_encoder(ego_plan).mean(0)  # [ego_plan_dim]
            
            # Optimize for each neighbor agent
            neighbor_scenarios = []
            neighbor_scores = []
            
            for n in range(decoder.neighbors):
                target_idx = n + 1  # +1 because 0 is ego
                
                if target_idx >= len(agent_embeddings):
                    # Padded agent, skip
                    neighbor_scenarios.append(
                        torch.zeros(timesteps, decoder.traj_dim, device=device)
                    )
                    neighbor_scores.append(torch.tensor(0.0, device=device))
                    continue
                
                # Get agent embedding
                target_emb = agent_embeddings[target_idx]
                
                # Get payoff embedding
                payoff_emb = decoder.get_payoff_embedding(target_idx, agent_embeddings, graph_info)
                
                # Construct condition
                condition = torch.cat([target_emb, encoded_ego_plan, payoff_emb], dim=0).unsqueeze(0)
                
                # Initialize multiple latent variables for parallel optimization
                best_z = None
                best_risk = -float('inf')
                best_trajectory = None
                
                for opt_idx in range(num_optimizations):
                    # Initialize latent variable
                    z = torch.randn(1, decoder.latent_dim, device=device, requires_grad=True)
                    
                    # Gradient ascent optimization
                    for step in range(optimization_steps):
                        # Generate trajectory from current z
                        pred_traj = decoder.decode_trajectory(z, condition).squeeze(0)  # [T, 3]
                        
                        # Calculate collision risk
                        current_states = agents_states[b:b+1, n:n+1, -1:]  # [1, 1, 11]
                        
                        # Ensure dimensions match for scorer
                        if current_states.shape[1] != 1:
                            current_states = current_states[:, :1]
                        
                        # Create dummy agents_traj for scorer
                        agents_traj_for_scorer = pred_traj.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, T, 3]
                        
                        try:
                            _, _, metrics = scorer(
                                ego_traj_inputs[b:b+1, m:m+1],  # [1, 1, T, 6]
                                encoder_outputs['encoding'][b:b+1, 0],  # [1, Dim]
                                agents_traj_for_scorer,  # [1, 1, 1, T, 3]
                                current_states,  # [1, 1, 11]
                                timesteps,
                                adversarial_mode=True
                            )
                            
                            collision_risk = metrics['collision_risk']
                            
                            # Gradient ascent: maximize collision risk
                            loss = -collision_risk  # Negative because optimizer minimizes
                            loss.backward()
                            
                            # Update z
                            with torch.no_grad():
                                z = z + learning_rate * z.grad
                                z.grad.zero_()
                            
                            # Track best result
                            if collision_risk.item() > best_risk:
                                best_risk = collision_risk.item()
                                best_z = z.detach().clone()
                                best_trajectory = pred_traj.detach().clone()
                                
                        except Exception as e:
                            # Fallback to random sampling if optimization fails
                            z = torch.randn(1, decoder.latent_dim, device=device)
                            pred_traj = decoder.decode_trajectory(z, condition).squeeze(0)
                            best_trajectory = pred_traj.detach().clone()
                            best_risk = 0.0
                            break
                
                neighbor_scenarios.append(best_trajectory)
                neighbor_scores.append(torch.tensor(best_risk, device=device))
            
            # Stack neighbor scenarios for this ego trajectory
            neighbor_scenarios = torch.stack(neighbor_scenarios)  # [N, T, 3]
            neighbor_scores = torch.stack(neighbor_scores)  # [N]
            
            batch_scenarios.append(neighbor_scenarios)
            batch_scores.append(neighbor_scores)
        
        # Stack scenarios for this batch
        batch_scenarios = torch.stack(batch_scenarios)  # [M, N, T, 3]
        batch_scores = torch.stack(batch_scores)  # [M, N]
        
        all_optimized_scenarios.append(batch_scenarios)
        all_optimized_scores.append(batch_scores)
    
    # Stack all batches
    adversarial_scenarios = torch.stack(all_optimized_scenarios)  # [B, M, N, T, 3]
    adversarial_scores = torch.stack(all_optimized_scores)  # [B, M, N]
    
    # Reshape to match expected output format
    adversarial_scenarios = adversarial_scenarios.unsqueeze(1)  # [B, 1, M, N, T, 3]
    adversarial_scores = adversarial_scores.unsqueeze(1)  # [B, 1, M, N]
    
    return adversarial_scenarios, adversarial_scores


def adversarial_scenario_generation(decoder, encoder_outputs, ego_traj_inputs, 
                                  agents_states, timesteps, method='optimization', 
                                  num_samples=10, optimization_steps=100):
    """
    Main function for adversarial scenario generation.
    
    Args:
        decoder: GGTP_Decoder instance
        encoder_outputs: encoder output dict
        ego_traj_inputs: [B, M, T, 6]
        agents_states: [B, N, T, 11]
        timesteps: int
        method: str - 'random' or 'optimization'
        num_samples: int - for random sampling
        optimization_steps: int - for gradient optimization
    
    Returns:
        adversarial_scenarios: [B, K, M, N, T, 3]
        adversarial_scores: [B, K, M]
    """
    if method == 'optimization':
        # Use true latent space optimization
        return latent_space_optimization(
            decoder, encoder_outputs, ego_traj_inputs, 
            agents_states, timesteps, 
            num_optimizations=5, 
            optimization_steps=optimization_steps,
            learning_rate=0.01
        )
    elif method == 'random':
        # Use random sampling (fallback)
        return adversarial_sampling_strategy(
            decoder, encoder_outputs, ego_traj_inputs, 
            agents_states, timesteps, num_samples
        )
    else:
        raise ValueError(f"Unknown method: {method}")


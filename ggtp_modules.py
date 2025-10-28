import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


class GNNEncoder(nn.Module):
    """
    GNN-based encoder for multi-agent scene understanding.
    Replaces the original LSTM+Transformer encoder.
    """
    def __init__(self, node_dim=11, edge_dim=4, dim=256, heads=4, layers=2):
        super(GNNEncoder, self).__init__()
        self.dim = dim
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        
        # Ego agent encoder (different input dim)
        self.ego_encoder = nn.Sequential(
            nn.Linear(7, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        
        # GAT layers
        self.conv_layers = nn.ModuleList()
        for _ in range(layers):
            self.conv_layers.append(
                GATConv(dim, dim // heads, heads=heads, concat=True, dropout=0.1)
            )
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        
    def build_graph(self, agent_features, proximity_threshold=50.0):
        """
        Builds a graph from agent features for a single scene.
        
        Args:
            agent_features: [Num_Agents, Feature_Dim]
            proximity_threshold: Distance threshold for edge creation
        
        Returns:
            Data: PyG graph data object
        """
        num_agents = agent_features.shape[0]
        device = agent_features.device
        
        # Extract positions (assuming first 2 dims are x, y)
        positions = agent_features[:, :2]
        
        # Build edges based on proximity
        edge_indices = []
        edge_attrs = []
        
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                    
                # Calculate distance
                dist = torch.norm(positions[i] - positions[j])
                
                if dist < proximity_threshold:
                    edge_indices.append([i, j])
                    
                    # Edge attributes: [dx, dy, distance, relative_heading]
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    if agent_features.shape[1] >= 3:
                        relative_heading = agent_features[j, 2] - agent_features[i, 2]
                    else:
                        relative_heading = 0.0
                    edge_attrs.append([dx, dy, dist, relative_heading])
        
        # Handle case with no neighbors
        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, 4), dtype=torch.float, device=device)
        else:
            edge_index = torch.tensor(edge_indices, device=device).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, device=device, dtype=torch.float)
        
        return Data(x=agent_features, edge_index=edge_index, edge_attr=edge_attr)
    
    def forward(self, inputs):
        """
        Args:
            inputs: dict with keys:
                - 'ego_agent_past': [B, T, 7]
                - 'neighbor_agents_past': [B, N, T, 11]
                - 'map_lanes': [B, N_lanes, N_points, 7] (optional, can be ignored for now)
        
        Returns:
            encoder_outputs: dict with keys:
                - 'encoding': [B, N_agents+1, Dim] (ego + neighbors)
                - 'graph_batch': List of PyG Data objects (for payoff embedding)
        """
        B = inputs['ego_agent_past'].shape[0]
        
        # Use last timestep for graph construction
        ego_last = inputs['ego_agent_past'][:, -1, :5]  # [B, 5]
        neighbors_last = inputs['neighbor_agents_past'][:, :, -1, :5]  # [B, N, 5]
        
        # Combine ego and neighbors
        all_agents = torch.cat([ego_last.unsqueeze(1), neighbors_last], dim=1)  # [B, N+1, 5]
        
        # Build graphs for each batch item
        data_list = []
        for b in range(B):
            # Filter out padded agents (all zeros)
            valid_mask = (all_agents[b].abs().sum(-1) > 0)
            valid_agents = all_agents[b][valid_mask]
            
            if len(valid_agents) > 0:
                graph = self.build_graph(valid_agents)
                data_list.append(graph)
            else:
                # Empty graph
                data_list.append(Data(x=torch.zeros(1, 5, device=all_agents.device),
                                    edge_index=torch.empty((2, 0), dtype=torch.long, device=all_agents.device)))
        
        # Batch graphs
        graph_batch = Batch.from_data_list(data_list)
        
        # Encode node features
        # Separate ego and neighbors
        is_ego = torch.zeros(graph_batch.x.shape[0], dtype=torch.bool, device=graph_batch.x.device)
        batch_idx = graph_batch.batch
        for i, data in enumerate(data_list):
            is_ego[batch_idx == i] = torch.cat([
                torch.tensor([True], device=is_ego.device),
                torch.zeros(data.num_nodes - 1, dtype=torch.bool, device=is_ego.device)
            ])
        
        # Encode (assuming we expand features to match expected dims)
        # For now, pad to expected dims
        node_features = torch.zeros(graph_batch.x.shape[0], 11, device=graph_batch.x.device)
        node_features[:, :graph_batch.x.shape[1]] = graph_batch.x
        
        h = torch.where(is_ego.unsqueeze(-1), 
                       self.ego_encoder(node_features[:, :7]),
                       self.node_encoder(node_features))
        
        # GNN message passing
        for conv in self.conv_layers:
            h = F.relu(conv(h, graph_batch.edge_index))
        
        h = self.output_proj(h)
        
        # Reshape back to [B, N_agents, Dim]
        num_agents_per_graph = [data.num_nodes for data in data_list]
        h_split = torch.split(h, num_agents_per_graph)
        
        # Pad to max agents
        max_agents = max(num_agents_per_graph)
        h_padded = torch.zeros(B, max_agents, self.dim, device=h.device)
        for i, h_i in enumerate(h_split):
            h_padded[i, :len(h_i)] = h_i
        
        # Create mask
        mask = torch.zeros(B, max_agents, dtype=torch.bool, device=h.device)
        for i, n in enumerate(num_agents_per_graph):
            mask[i, n:] = True
        
        return {
            'encoding': h_padded,
            'mask': mask,
            'graph_batch': data_list
        }


class GGTP_Decoder(nn.Module):
    """
    CVAE-based decoder for game-theoretic trajectory prediction.
    """
    def __init__(self, neighbors=10, max_time=8, max_branch=30, 
                 agent_dim=256, ego_plan_dim=256, latent_dim=32, 
                 horizon=80, traj_dim=3):
        super(GGTP_Decoder, self).__init__()
        self.neighbors = neighbors
        self.max_time = max_time
        self.max_branch = max_branch
        self.horizon = horizon
        self.traj_dim = traj_dim
        self.latent_dim = latent_dim
        
        # Payoff embedding: concatenate K nearest neighbor embeddings
        self.num_neighbors_for_payoff = 4
        payoff_dim = self.num_neighbors_for_payoff * agent_dim
        
        # Ego plan encoder
        self.ego_plan_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, ego_plan_dim)
        )
        
        # Total condition dimension
        condition_dim = agent_dim + ego_plan_dim + payoff_dim
        
        # === CVAE Posterior Encoder ===
        self.posterior_encoder = nn.Sequential(
            nn.Linear(condition_dim + horizon * traj_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # === CVAE Decoder (Generator) ===
        self.generator = nn.Sequential(
            nn.Linear(condition_dim + latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * traj_dim)
        )
        
        # Score decoder (reuse from original)
        from prediction_modules import ScoreDecoder
        self.scorer = ScoreDecoder(variable_cost=False)
    
    def get_payoff_embedding(self, target_idx, agent_embeddings, graph_info):
        """
        Extract payoff embedding for target agent.
        
        Args:
            target_idx: Index of target agent in the graph
            agent_embeddings: [Num_Agents, Dim]
            graph_info: PyG Data object
        
        Returns:
            payoff_emb: [payoff_dim]
        """
        device = agent_embeddings.device
        agent_dim = agent_embeddings.shape[1]
        
        # Find neighbors in the graph
        edge_index = graph_info.edge_index
        neighbors = edge_index[1, edge_index[0] == target_idx].tolist()
        
        # Get embeddings of K nearest neighbors
        neighbor_embeddings = []
        for i in range(self.num_neighbors_for_payoff):
            if i < len(neighbors):
                neighbor_embeddings.append(agent_embeddings[neighbors[i]])
            else:
                # Pad with zeros
                neighbor_embeddings.append(torch.zeros(agent_dim, device=device))
        
        return torch.cat(neighbor_embeddings, dim=0)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_posterior(self, ground_truth_future, condition):
        """Encode posterior q(z|x,c)"""
        # ground_truth_future: [1, T, 3]
        # Flatten to [1, T*3]
        flat_future = ground_truth_future.reshape(ground_truth_future.shape[0], -1)
        
        # Expected input size: condition_dim + horizon * traj_dim
        expected_future_dim = self.horizon * self.traj_dim
        current_future_dim = flat_future.shape[1]
        
        # Pad or truncate to match expected dimension
        if current_future_dim < expected_future_dim:
            # Pad with zeros
            padding = torch.zeros(flat_future.shape[0], expected_future_dim - current_future_dim, 
                                device=flat_future.device)
            flat_future = torch.cat([flat_future, padding], dim=1)
        elif current_future_dim > expected_future_dim:
            # Truncate
            flat_future = flat_future[:, :expected_future_dim]
        
        x = torch.cat([condition, flat_future], dim=1)
        h = self.posterior_encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def decode_trajectory(self, z, condition, timesteps=None):
        """Decode p(x|z,c)"""
        x = torch.cat([condition, z], dim=1)
        flat_traj = self.generator(x)
        # Always generates full horizon, then truncate if needed
        traj = flat_traj.view(-1, self.horizon, self.traj_dim)
        if timesteps is not None and timesteps < self.horizon:
            traj = traj[:, :timesteps]
        return traj
    
    def forward(self, encoder_outputs, ego_traj_inputs, agents_states, timesteps, 
                ground_truth_futures=None):
        """
        Args:
            encoder_outputs: dict from GNNEncoder
            ego_traj_inputs: [B, M, T, 6] - M candidate ego trajectories
            agents_states: [B, N, T, 11] - neighbor states
            timesteps: int - prediction horizon
            ground_truth_futures: [B, N, T, 3] - for training only
        
        Returns:
            agents_trajectories: [B, M, N, T, 3]
            scores: [B, M]
            (mu, logvar): for training loss
        """
        B, M = ego_traj_inputs.shape[0], ego_traj_inputs.shape[1]
        encoding = encoder_outputs['encoding']  # [B, Max_Agents, Dim]
        graph_batch = encoder_outputs['graph_batch']  # List of graphs
        
        current_states = agents_states[:, :self.neighbors, -1]  # [B, N, 11]
        
        all_predictions = []
        all_mu = []
        all_logvar = []
        
        # Process each batch item
        for b in range(B):
            batch_predictions = []
            batch_mu = []
            batch_logvar = []
            
            agent_embeddings = encoding[b]  # [Max_Agents, Dim]
            graph_info = graph_batch[b]
            
            # For each ego trajectory candidate
            for m in range(M):
                ego_plan = ego_traj_inputs[b, m]  # [T, 6]
                
                # Encode ego plan (pool over time)
                encoded_ego_plan = self.ego_plan_encoder(ego_plan).mean(0)  # [ego_plan_dim]
                
                # Predict for each neighbor
                neighbor_predictions = []
                neighbor_mu = []
                neighbor_logvar = []
                
                for n in range(self.neighbors):
                    target_idx = n + 1  # +1 because 0 is ego
                    
                    if target_idx >= len(agent_embeddings):
                        # Padded agent, skip
                        neighbor_predictions.append(
                            torch.zeros(timesteps, self.traj_dim, device=encoding.device)
                        )
                        if self.training:
                            neighbor_mu.append(torch.zeros(self.latent_dim, device=encoding.device))
                            neighbor_logvar.append(torch.zeros(self.latent_dim, device=encoding.device))
                        continue
                    
                    # Get agent embedding
                    target_emb = agent_embeddings[target_idx]
                    
                    # Get payoff embedding (KEY TRICK!)
                    payoff_emb = self.get_payoff_embedding(target_idx, agent_embeddings, graph_info)
                    
                    # Construct condition
                    condition = torch.cat([target_emb, encoded_ego_plan, payoff_emb], dim=0).unsqueeze(0)
                    
                    if self.training and ground_truth_futures is not None:
                        # Training: use posterior
                        gt_future = ground_truth_futures[b, n, :timesteps]  # [T, 3]
                        mu, logvar = self.encode_posterior(gt_future.unsqueeze(0), condition)
                        z = self.reparameterize(mu, logvar)
                        pred_traj = self.decode_trajectory(z, condition, timesteps)
                        
                        neighbor_predictions.append(pred_traj.squeeze(0))
                        neighbor_mu.append(mu.squeeze(0))
                        neighbor_logvar.append(logvar.squeeze(0))
                    else:
                        # Inference: sample from prior
                        z = torch.randn(1, self.latent_dim, device=encoding.device)
                        pred_traj = self.decode_trajectory(z, condition, timesteps)
                        neighbor_predictions.append(pred_traj.squeeze(0))
                
                # Stack predictions for this ego trajectory
                neighbor_predictions = torch.stack(neighbor_predictions)  # [N, T, 3]
                batch_predictions.append(neighbor_predictions)
                
                if self.training:
                    batch_mu.append(torch.stack(neighbor_mu))
                    batch_logvar.append(torch.stack(neighbor_logvar))
            
            # Stack predictions for all ego trajectories
            batch_predictions = torch.stack(batch_predictions)  # [M, N, T, 3]
            all_predictions.append(batch_predictions)
            
            if self.training:
                all_mu.append(torch.stack(batch_mu))
                all_logvar.append(torch.stack(batch_logvar))
        
        # Final output
        agents_trajectories = torch.stack(all_predictions)  # [B, M, N, T, 3]
        
        # Score trajectories
        scores, weights = self.scorer(
            ego_traj_inputs,
            encoding[:, 0],  # Ego encoding
            agents_trajectories,
            current_states,
            timesteps
        )
        
        if self.training:
            mu = torch.stack(all_mu)  # [B, M, N, latent_dim]
            logvar = torch.stack(all_logvar)
            return agents_trajectories, scores, weights, mu, logvar
        else:
            return agents_trajectories, scores, weights


def cvae_loss_function(recon_traj, gt_traj, mu, logvar, beta=0.1):
    """
    CVAE loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        recon_traj: [B, M, N, T, 3]
        gt_traj: [B, N, T, 3]
        mu, logvar: [B, M, N, latent_dim]
        beta: KL divergence weight
    """
    B, M, N, T_pred, _ = recon_traj.shape
    T_gt = gt_traj.shape[2]
    
    # Handle different trajectory lengths
    if T_pred != T_gt:
        # Truncate or pad to match
        min_T = min(T_pred, T_gt)
        recon_traj = recon_traj[:, :, :, :min_T]
        gt_traj = gt_traj[:, :, :min_T]
    
    # Expand gt for all ego trajectory candidates
    gt_expanded = gt_traj.unsqueeze(1).expand(-1, M, -1, -1, -1)  # [B, M, N, T, 3]
    
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_traj, gt_expanded, reduction='sum') / B
    
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    total_loss = recon_loss + beta * kld_loss
    
    return total_loss, recon_loss, kld_loss


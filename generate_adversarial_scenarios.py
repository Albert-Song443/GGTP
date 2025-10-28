"""
Generate and Evaluate Adversarial Scenarios for AV Safety Testing

This script generates high-risk, realistic traffic scenarios using the
trained adversarial GGTP model, and evaluates their criticality.
"""
import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ggtp_modules import GNNEncoder, GGTP_Decoder
from adversarial_modules import AdversarialScoreDecoder, adversarial_scenario_generation
from torch.utils.data import DataLoader
from train_utils import *


class AdversarialScenarioGenerator:
    """Generate adversarial scenarios for safety testing."""
    
    def __init__(self, model_path, device='cuda', num_samples=10):
        self.device = device
        self.num_samples = num_samples
        
        # Load models
        checkpoint = torch.load(model_path, map_location=device)
        
        self.encoder = GNNEncoder(node_dim=11, dim=256, heads=4, layers=2).to(device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.encoder.eval()
        
        self.decoder = GGTP_Decoder(
            neighbors=10,
            max_time=8,
            max_branch=30,
            latent_dim=32
        ).to(device)
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.decoder.eval()
        
        self.scorer = AdversarialScoreDecoder(
            adversarial_weight=10.0,
            realism_weight=1.0
        ).to(device)
        self.scorer.load_state_dict(checkpoint['scorer'])
        self.scorer.eval()
    
    def generate_scenarios(self, data_loader, num_scenarios=100):
        """
        Generate adversarial scenarios from dataset.
        
        Returns:
            scenarios: list of dict containing:
                - ego_trajectory: [T, 6]
                - agent_trajectories: [N, T, 3]
                - collision_risk: float
                - realism_score: float
                - criticality_metrics: dict
        """
        scenarios = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating Scenarios"):
                if len(scenarios) >= num_scenarios:
                    break
                
                inputs = {
                    'ego_agent_past': batch[0].to(self.device),
                    'neighbor_agents_past': batch[1].to(self.device),
                    'map_lanes': batch[2].to(self.device),
                    'map_crosswalks': batch[3].to(self.device),
                    'route_lanes': batch[4].to(self.device)
                }
                
                ego_traj_candidates = batch[8].to(self.device)  # [B, M, T, 6]
                
                # Encode scene
                encoder_outputs = self.encoder(inputs)
                
                # Generate multiple adversarial samples using Latent Space Optimization
                adv_scenarios, adv_scores = adversarial_scenario_generation(
                    self.decoder,
                    encoder_outputs,
                    ego_traj_candidates,
                    inputs['neighbor_agents_past'],
                    timesteps=80,
                    method='optimization',  # Use true gradient optimization
                    optimization_steps=100  # Number of gradient ascent steps
                )
                
                # For each batch item
                B = ego_traj_candidates.shape[0]
                for b in range(B):
                    if len(scenarios) >= num_scenarios:
                        break
                    
                    # Select most adversarial scenario
                    best_idx = adv_scores[b, :, :].max(-1)[0].argmax()
                    best_ego_idx = adv_scores[b, best_idx].argmax()
                    
                    ego_traj = ego_traj_candidates[b, best_ego_idx].cpu().numpy()
                    agent_trajs = adv_scenarios[b, best_idx, best_ego_idx].cpu().numpy()
                    
                    # Evaluate criticality
                    current_states = inputs['neighbor_agents_past'][b:b+1, :10, -1]
                    _, _, metrics = self.scorer(
                        ego_traj_candidates[b:b+1, best_ego_idx:best_ego_idx+1],
                        encoder_outputs['encoding'][b:b+1, 0],
                        adv_scenarios[b:b+1, best_idx:best_idx+1, best_ego_idx:best_ego_idx+1],
                        current_states,
                        80,
                        adversarial_mode=True
                    )
                    
                    scenario = {
                        'ego_trajectory': ego_traj,
                        'agent_trajectories': agent_trajs,
                        'collision_risk': metrics['collision_risk'],
                        'realism_score': metrics['realism_score'],
                        'diversity_score': metrics['diversity_score'],
                        'criticality_metrics': self.compute_criticality_metrics(
                            ego_traj, agent_trajs
                        )
                    }
                    
                    scenarios.append(scenario)
        
        return scenarios
    
    def compute_criticality_metrics(self, ego_traj, agent_trajs):
        """
        Compute detailed criticality metrics for a scenario.
        
        Metrics:
        - TTC (Time-to-Collision): minimum TTC across all agents
        - PET (Post-Encroachment Time): time gap at conflict points
        - Minimum Distance: closest approach distance
        - Collision Probability: estimated probability of collision
        """
        metrics = {}
        
        # Minimum distance
        min_dists = []
        for agent_traj in agent_trajs:
            if agent_traj.sum() == 0:  # Skip padded agents
                continue
            dists = np.linalg.norm(
                ego_traj[:, :2] - agent_traj[:, :2], axis=1
            )
            min_dists.append(dists.min())
        
        metrics['min_distance'] = min(min_dists) if min_dists else np.inf
        
        # Time-to-Collision (TTC)
        ttcs = []
        for agent_traj in agent_trajs:
            if agent_traj.sum() == 0:
                continue
            
            # Compute relative velocity
            ego_vel = np.diff(ego_traj[:, :2], axis=0)
            agent_vel = np.diff(agent_traj[:, :2], axis=0)
            rel_vel = np.linalg.norm(ego_vel - agent_vel, axis=1)
            
            # Compute distance
            dists = np.linalg.norm(
                ego_traj[:-1, :2] - agent_traj[:-1, :2], axis=1
            )
            
            # TTC = distance / relative_velocity
            with np.errstate(divide='ignore', invalid='ignore'):
                ttc = dists / (rel_vel + 1e-6)
                ttc = ttc[ttc > 0]  # Only positive TTC
                if len(ttc) > 0:
                    ttcs.append(ttc.min())
        
        metrics['min_ttc'] = min(ttcs) if ttcs else np.inf
        
        # Collision count
        collision_threshold = 3.0  # meters
        collision_count = sum([d < collision_threshold for d in min_dists])
        metrics['collision_count'] = collision_count
        
        # Criticality level
        if metrics['min_distance'] < 2.0:
            metrics['criticality_level'] = 'CRITICAL'
        elif metrics['min_distance'] < 5.0:
            metrics['criticality_level'] = 'HIGH'
        elif metrics['min_distance'] < 10.0:
            metrics['criticality_level'] = 'MEDIUM'
        else:
            metrics['criticality_level'] = 'LOW'
        
        return metrics
    
    def save_scenarios(self, scenarios, output_dir):
        """Save generated scenarios to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each scenario
        for i, scenario in enumerate(scenarios):
            scenario_file = os.path.join(output_dir, f'scenario_{i:04d}.npz')
            np.savez(
                scenario_file,
                ego_trajectory=scenario['ego_trajectory'],
                agent_trajectories=scenario['agent_trajectories'],
                collision_risk=scenario['collision_risk'],
                realism_score=scenario['realism_score']
            )
        
        # Save metadata
        metadata = {
            'num_scenarios': len(scenarios),
            'avg_collision_risk': np.mean([s['collision_risk'] for s in scenarios]),
            'avg_realism': np.mean([s['realism_score'] for s in scenarios]),
            'criticality_distribution': self.get_criticality_distribution(scenarios)
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved {len(scenarios)} scenarios to {output_dir}")
        print(f"Average Collision Risk: {metadata['avg_collision_risk']:.4f}")
        print(f"Average Realism: {metadata['avg_realism']:.4f}")
        print(f"Criticality Distribution: {metadata['criticality_distribution']}")
    
    def get_criticality_distribution(self, scenarios):
        """Get distribution of criticality levels."""
        levels = [s['criticality_metrics']['criticality_level'] for s in scenarios]
        distribution = {
            'CRITICAL': levels.count('CRITICAL'),
            'HIGH': levels.count('HIGH'),
            'MEDIUM': levels.count('MEDIUM'),
            'LOW': levels.count('LOW')
        }
        return distribution
    
    def visualize_scenario(self, scenario, save_path=None):
        """Visualize a single scenario."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot trajectory view
        ax = axes[0]
        ego_traj = scenario['ego_trajectory']
        agent_trajs = scenario['agent_trajectories']
        
        # Plot ego trajectory
        ax.plot(ego_traj[:, 0], ego_traj[:, 1], 'b-', linewidth=3, label='Ego')
        ax.scatter(ego_traj[0, 0], ego_traj[0, 1], c='blue', s=100, marker='o')
        ax.scatter(ego_traj[-1, 0], ego_traj[-1, 1], c='blue', s=100, marker='x')
        
        # Plot agent trajectories
        for i, agent_traj in enumerate(agent_trajs):
            if agent_traj.sum() == 0:
                continue
            ax.plot(agent_traj[:, 0], agent_traj[:, 1], '--', alpha=0.7, 
                   label=f'Agent {i}')
            ax.scatter(agent_traj[0, 0], agent_traj[0, 1], s=50, marker='o')
            ax.scatter(agent_traj[-1, 0], agent_traj[-1, 1], s=50, marker='x')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Scenario Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Plot metrics
        ax = axes[1]
        metrics = scenario['criticality_metrics']
        metric_names = ['Collision Risk', 'Realism', 'Min Distance', 'Min TTC']
        metric_values = [
            scenario['collision_risk'],
            scenario['realism_score'],
            min(metrics['min_distance'], 20.0) / 20.0,  # Normalize
            min(metrics['min_ttc'], 10.0) / 10.0  # Normalize
        ]
        
        colors = ['red', 'green', 'orange', 'purple']
        ax.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Normalized Score')
        ax.set_title(f"Criticality: {metrics['criticality_level']}")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def evaluate_scenarios(self, scenarios):
        """Evaluate and rank scenarios by criticality."""
        print("\n" + "=" * 60)
        print("SCENARIO EVALUATION REPORT")
        print("=" * 60)
        
        # Sort by collision risk
        scenarios_sorted = sorted(
            scenarios,
            key=lambda x: x['collision_risk'],
            reverse=True
        )
        
        print(f"\nTotal Scenarios: {len(scenarios)}")
        print(f"\nTop 10 Most Critical Scenarios:")
        print("-" * 60)
        
        for i, scenario in enumerate(scenarios_sorted[:10]):
            metrics = scenario['criticality_metrics']
            print(f"\n{i+1}. Scenario - {metrics['criticality_level']}")
            print(f"   Collision Risk:  {scenario['collision_risk']:.4f}")
            print(f"   Realism Score:   {scenario['realism_score']:.4f}")
            print(f"   Min Distance:    {metrics['min_distance']:.2f} m")
            print(f"   Min TTC:         {metrics['min_ttc']:.2f} s")
            print(f"   Collisions:      {metrics['collision_count']}")
        
        # Statistics
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        
        if len(scenarios) == 0:
            print("\n⚠️  Warning: No scenarios were generated!")
            print("This may indicate:")
            print("  1. Model not loaded correctly")
            print("  2. Data path is empty or incorrect")
            print("  3. Batch size too large for available data")
            return []
        
        collision_risks = [s['collision_risk'] for s in scenarios]
        realism_scores = [s['realism_score'] for s in scenarios]
        min_distances = [s['criticality_metrics']['min_distance'] for s in scenarios]
        
        print(f"\nCollision Risk:  mean={np.mean(collision_risks):.4f}, "
              f"std={np.std(collision_risks):.4f}")
        print(f"Realism Score:   mean={np.mean(realism_scores):.4f}, "
              f"std={np.std(realism_scores):.4f}")
        print(f"Min Distance:    mean={np.mean(min_distances):.2f} m, "
              f"min={np.min(min_distances):.2f} m")
        
        # Criticality distribution
        distribution = self.get_criticality_distribution(scenarios)
        print(f"\nCriticality Distribution:")
        for level, count in distribution.items():
            percentage = count / len(scenarios) * 100
            print(f"  {level:10s}: {count:3d} ({percentage:5.1f}%)")
        
        return scenarios_sorted


def main(args):
    print("\n" + "=" * 60)
    print("ADVERSARIAL SCENARIO GENERATOR")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Num Scenarios: {args.num_scenarios}")
    print("=" * 60 + "\n")
    
    # Initialize generator
    generator = AdversarialScenarioGenerator(
        model_path=args.model_path,
        device=args.device,
        num_samples=args.num_samples
    )
    
    # Load test data
    from glob import glob
    # Search for .npz files recursively in subdirectories
    test_files = glob(os.path.join(args.data_path, '**', '*.npz'), recursive=True)
    
    if len(test_files) == 0:
        print(f"\n⚠️  Error: No .npz files found in {args.data_path}")
        print("Please run data_process.py first to generate processed data.")
        return
    
    print(f"Found {len(test_files)} processed scenarios")
    test_set = DrivingData(test_files[:100], n_neighbors=10, n_candidates=30)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    
    # Generate scenarios
    print("Generating adversarial scenarios...")
    scenarios = generator.generate_scenarios(test_loader, num_scenarios=args.num_scenarios)
    
    # Evaluate scenarios
    scenarios_ranked = generator.evaluate_scenarios(scenarios)
    
    # Save scenarios
    generator.save_scenarios(scenarios, args.output_dir)
    
    # Visualize top scenarios
    if args.visualize:
        print(f"\nVisualizing top {args.visualize_top} scenarios...")
        vis_dir = os.path.join(args.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        for i, scenario in enumerate(scenarios_ranked[:args.visualize_top]):
            save_path = os.path.join(vis_dir, f'scenario_{i:02d}.png')
            generator.visualize_scenario(scenario, save_path)
        
        print(f"Visualizations saved to {vis_dir}")
    
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated {len(scenarios)} adversarial scenarios")
    print(f"Saved to: {args.output_dir}")
    print("\nUse these scenarios to test your AV system's safety!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate Adversarial Scenarios for AV Safety Testing'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained adversarial GGTP model')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='adversarial_scenarios',
                       help='Output directory for generated scenarios')
    
    parser.add_argument('--num_scenarios', type=int, default=100,
                       help='Number of scenarios to generate')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of CVAE samples per scenario')
    parser.add_argument('--batch_size', type=int, default=4)
    
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize generated scenarios')
    parser.add_argument('--visualize_top', type=int, default=10,
                       help='Number of top scenarios to visualize')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    main(args)


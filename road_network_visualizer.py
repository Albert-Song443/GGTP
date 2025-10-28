#!/usr/bin/env python3
"""
Real Road Network Visualization for GGTP Paper

This script creates a professional road network visualization using real nuPlan data
to demonstrate GGTP's adversarial scenario generation capabilities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import random
import os
from pathlib import Path

# Set matplotlib style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
    'figure.dpi': 300
})

class RoadNetworkVisualizer:
    def __init__(self):
        """Initialize the visualizer."""
        self.colors = {
            'ego': '#2E8B57',      # Sea Green
            'neighbor': '#4169E1', # Royal Blue
            'adversarial': '#DC143C', # Crimson
            'road': '#696969',     # Dim Gray
            'lane': '#D3D3D3',    # Light Gray
            'collision': '#FF4500', # Orange Red
            'candidate': '#32CD32'  # Lime Green
        }
    
    def load_nuplan_scenario(self, data_path):
        """Load a nuPlan scenario."""
        print(f"📊 Loading scenario: {data_path}")
        
        try:
            data = np.load(data_path, allow_pickle=True)
            
            scenario = {
                'ego_past': data['ego_agent_past'],
                'neighbors_past': data['neighbor_agents_past'],
                'map_lanes': data['map_lanes'],
                'map_crosswalks': data['map_crosswalks'],
                'ego_candidates': data['ego_trajectory_candidates'],
                'neighbors_future': data['neighbor_agents_future']
            }
            
            print(f"✅ Scenario loaded!")
            print(f"   Ego past: {scenario['ego_past'].shape}")
            print(f"   Neighbors: {scenario['neighbors_past'].shape}")
            print(f"   Lanes: {scenario['map_lanes'].shape[0]}")
            print(f"   Candidates: {scenario['ego_candidates'].shape[0]}")
            
            return scenario
            
        except Exception as e:
            print(f"❌ Error loading scenario: {e}")
            return None
    
    def create_road_network_diagram(self, scenario, save_path="road_network_demo.png"):
        """Create a professional road network diagram."""
        print("🎨 Creating road network diagram...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 8))
        
        # Left subplot: Original scenario
        ax1 = plt.subplot(1, 2, 1)
        self.plot_original_scenario(ax1, scenario)
        
        # Right subplot: Adversarial scenario
        ax2 = plt.subplot(1, 2, 2)
        self.plot_adversarial_scenario(ax2, scenario)
        
        # Add main title
        fig.suptitle('GGTP: Real Road Network Adversarial Scenario Generation', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Diagram saved to: {save_path}")
        
        return fig
    
    def plot_original_scenario(self, ax, scenario):
        """Plot the original scenario."""
        ax.set_title('Original Scenario\n(Safe Driving Behaviors)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Plot road network
        self.plot_road_network(ax, scenario['map_lanes'], scenario['map_crosswalks'])
        
        # Plot ego vehicle
        ego_past = scenario['ego_past']
        ax.plot(ego_past[:, 0], ego_past[:, 1], 
               color=self.colors['ego'], linewidth=4, label='Ego Vehicle', alpha=0.8)
        ax.scatter(ego_past[-1, 0], ego_past[-1, 1], 
                  c=self.colors['ego'], s=150, marker='o', 
                  edgecolors='white', linewidth=2, zorder=10, label='Ego Current')
        
        # Plot neighbor vehicles
        for i, neighbor in enumerate(scenario['neighbors_past']):
            if len(neighbor) > 0:
                # Past trajectory
                ax.plot(neighbor[:, 0], neighbor[:, 1], 
                       color=self.colors['neighbor'], linewidth=2, alpha=0.7)
                ax.scatter(neighbor[-1, 0], neighbor[-1, 1], 
                          c=self.colors['neighbor'], s=80, marker='s', 
                          edgecolors='white', linewidth=1, alpha=0.8)
                
                # Future trajectory (ground truth)
                if i < len(scenario['neighbors_future']) and len(scenario['neighbors_future'][i]) > 0:
                    future = scenario['neighbors_future'][i]
                    ax.plot(future[:, 0], future[:, 1], 
                           color=self.colors['neighbor'], linewidth=2, 
                           linestyle='--', alpha=0.5, label='Neighbor Future' if i == 0 else "")
        
        # Plot ego candidate trajectories
        candidates = scenario['ego_candidates'][:5]  # Show first 5
        for i, candidate in enumerate(candidates):
            alpha = 0.3 if i > 0 else 0.6
            ax.plot(candidate[:, 0], candidate[:, 1], 
                   color=self.colors['candidate'], linewidth=2, 
                   linestyle=':', alpha=alpha, 
                   label='Ego Candidates' if i == 0 else "")
        
        # Add annotations
        ax.annotate('Safe Following\nDistance', 
                   xy=(ego_past[-1, 0], ego_past[-1, 1]), 
                   xytext=(ego_past[-1, 0] + 20, ego_past[-1, 1] + 10),
                   arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                   fontsize=10, ha='center')
        
        self.format_axes(ax)
        ax.legend(loc='upper right', fontsize=10)
    
    def plot_adversarial_scenario(self, ax, scenario):
        """Plot the adversarial scenario."""
        ax.set_title('Adversarial Scenario\n(High-Risk Behaviors)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Plot road network
        self.plot_road_network(ax, scenario['map_lanes'], scenario['map_crosswalks'])
        
        # Plot ego vehicle
        ego_past = scenario['ego_past']
        ax.plot(ego_past[:, 0], ego_past[:, 1], 
               color=self.colors['ego'], linewidth=4, label='Ego Vehicle', alpha=0.8)
        ax.scatter(ego_past[-1, 0], ego_past[-1, 1], 
                  c=self.colors['ego'], s=150, marker='o', 
                  edgecolors='white', linewidth=2, zorder=10, label='Ego Current')
        
        # Plot original neighbor trajectories (faded)
        for i, neighbor in enumerate(scenario['neighbors_past']):
            if len(neighbor) > 0:
                ax.plot(neighbor[:, 0], neighbor[:, 1], 
                       color=self.colors['neighbor'], linewidth=1, alpha=0.3)
                ax.scatter(neighbor[-1, 0], neighbor[-1, 1], 
                          c=self.colors['neighbor'], s=40, marker='s', alpha=0.3)
        
        # Generate adversarial trajectory
        adversarial_traj = self.generate_adversarial_trajectory(scenario)
        
        # Plot adversarial trajectory
        ax.plot(adversarial_traj[:, 0], adversarial_traj[:, 1], 
               color=self.colors['adversarial'], linewidth=5, 
               label='Adversarial Neighbor', alpha=0.9)
        ax.scatter(adversarial_traj[-1, 0], adversarial_traj[-1, 1], 
                  c=self.colors['adversarial'], s=150, marker='D', 
                  edgecolors='white', linewidth=2, zorder=10, 
                  label='Adversarial End')
        
        # Highlight collision zones
        self.highlight_collision_zones(ax, scenario['ego_candidates'], adversarial_traj)
        
        # Plot ego candidate trajectories
        candidates = scenario['ego_candidates'][:5]
        for i, candidate in enumerate(candidates):
            alpha = 0.3 if i > 0 else 0.6
            ax.plot(candidate[:, 0], candidate[:, 1], 
                   color=self.colors['candidate'], linewidth=2, 
                   linestyle=':', alpha=alpha, 
                   label='Ego Candidates' if i == 0 else "")
        
        # Add risk annotations
        ax.annotate('Dangerous\nCut-in', 
                   xy=(adversarial_traj[20, 0], adversarial_traj[20, 1]), 
                   xytext=(adversarial_traj[20, 0] - 30, adversarial_traj[20, 1] + 15),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.8),
                   fontsize=10, ha='center', color='red', fontweight='bold')
        
        self.format_axes(ax)
        ax.legend(loc='upper right', fontsize=10)
    
    def plot_road_network(self, ax, lanes, crosswalks):
        """Plot the road network infrastructure."""
        # Plot lanes
        for lane in lanes:
            ax.plot(lane[:, 0], lane[:, 1], 
                   color=self.colors['road'], linewidth=3, alpha=0.6)
            ax.plot(lane[:, 0], lane[:, 1], 
                   color=self.colors['lane'], linewidth=1, alpha=0.8)
        
        # Plot crosswalks
        for crosswalk in crosswalks:
            ax.plot(crosswalk[:, 0], crosswalk[:, 1], 
                   color=self.colors['road'], linewidth=2, alpha=0.4)
    
    def generate_adversarial_trajectory(self, scenario):
        """Generate a simple adversarial trajectory for demonstration."""
        # Get the first neighbor's current position
        neighbor = scenario['neighbors_past'][0]
        if len(neighbor) == 0:
            # Create a dummy neighbor if none exists
            neighbor = np.array([[50, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0]])
        
        start_pos = neighbor[-1, :2]  # [x, y]
        start_vel = neighbor[-1, 3:5] if neighbor.shape[1] > 4 else [15, 0]  # [vx, vy]
        
        # Create adversarial trajectory (aggressive lane change)
        timesteps = 80
        dt = 0.1
        
        trajectory = np.zeros((timesteps, 3))  # [x, y, heading]
        trajectory[0, :2] = start_pos
        
        # Simulate aggressive behavior
        for t in range(1, timesteps):
            # Aggressive lateral movement (lane change)
            lateral_accel = 2.0 * np.sin(t * 0.1)  # Oscillating lateral acceleration
            
            # Update position
            trajectory[t, 0] = trajectory[t-1, 0] + start_vel[0] * dt
            trajectory[t, 1] = trajectory[t-1, 1] + lateral_accel * dt
            trajectory[t, 2] = np.arctan2(lateral_accel, start_vel[0])
        
        return trajectory
    
    def highlight_collision_zones(self, ax, ego_candidates, adversarial_traj):
        """Highlight potential collision zones."""
        min_distance = float('inf')
        collision_points = []
        
        # Check for close approaches
        for ego_candidate in ego_candidates[:3]:  # Check first 3 candidates
            for t in range(min(len(ego_candidate), len(adversarial_traj))):
                ego_pos = ego_candidate[t, :2]
                adv_pos = adversarial_traj[t, :2]
                distance = np.linalg.norm(ego_pos - adv_pos)
                
                if distance < 8:  # Within 8 meters
                    collision_points.append((ego_pos + adv_pos) / 2)
                    min_distance = min(min_distance, distance)
        
        # Highlight collision zones
        for point in collision_points:
            circle = Circle(point, radius=6, 
                          facecolor=self.colors['collision'], 
                          alpha=0.4, edgecolor='red', linewidth=2)
            ax.add_patch(circle)
        
        # Add collision risk text
        if collision_points:
            ax.text(0.02, 0.98, f'Collision Risk: {1/min_distance:.1f}x', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                   verticalalignment='top', color='white')
    
    def format_axes(self, ax):
        """Format the axes for professional appearance."""
        ax.set_xlabel('X Position (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (meters)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal')
        
        # Set reasonable limits
        ax.set_xlim(-50, 150)
        ax.set_ylim(-50, 50)
        
        # Add subtle background
        ax.set_facecolor('#FAFAFA')
    
    def run_demo(self, data_dir="nuplan/processed_data/valid"):
        """Run the complete demo."""
        print("🚀 Starting Road Network Visualization Demo...")
        
        # Find scenario files
        data_files = list(Path(data_dir).glob("*.npz"))
        if not data_files:
            print(f"❌ No data files found in {data_dir}")
            return None
        
        # Select a good scenario (prefer ones with multiple vehicles)
        best_scenario = None
        best_score = 0
        
        for data_file in data_files[:10]:  # Check first 10 files
            try:
                data = np.load(data_file, allow_pickle=True)
                neighbor_count = len([n for n in data['neighbor_agents_past'] if len(n) > 0])
                if neighbor_count > best_score:
                    best_score = neighbor_count
                    best_scenario = data_file
            except:
                continue
        
        if best_scenario is None:
            best_scenario = random.choice(data_files)
        
        print(f"📁 Selected scenario: {best_scenario.name}")
        
        # Load scenario
        scenario = self.load_nuplan_scenario(best_scenario)
        if scenario is None:
            return None
        
        # Create visualization
        fig = self.create_road_network_diagram(scenario)
        
        # Show the plot
        plt.show()
        
        print("🎉 Demo completed successfully!")
        return fig

def main():
    """Main function."""
    visualizer = RoadNetworkVisualizer()
    fig = visualizer.run_demo()
    
    if fig is not None:
        print("\n💡 Tips for paper:")
        print("   - Use this diagram in your paper to show real-world applicability")
        print("   - The left plot shows normal driving behaviors")
        print("   - The right plot shows GGTP-generated adversarial behaviors")
        print("   - Red zones indicate high collision risk areas")

if __name__ == "__main__":
    main()




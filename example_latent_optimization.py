#!/usr/bin/env python3
"""
Example usage of Latent Space Optimization for Adversarial Scenario Generation.

This script demonstrates how to use the new gradient-based optimization
method instead of random sampling for generating adversarial scenarios.
"""

import torch
import numpy as np
from adversarial_modules import adversarial_scenario_generation
from ggtp_modules import GNNEncoder, GGTP_Decoder

def example_usage():
    """Example of using latent space optimization."""
    print("🎯 Example: Latent Space Optimization for Adversarial Scenarios")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load your trained models
    print("📥 Loading trained models...")
    # checkpoint = torch.load('path/to/your/model.pth', map_location=device)
    # encoder = GNNEncoder(...).to(device)
    # decoder = GGTP_Decoder(...).to(device)
    # encoder.load_state_dict(checkpoint['encoder'])
    # decoder.load_state_dict(checkpoint['decoder'])
    
    # For this example, create dummy models
    encoder = GNNEncoder(node_dim=11, dim=256, heads=4, layers=2).to(device)
    decoder = GGTP_Decoder(neighbors=10, max_time=8, max_branch=30, latent_dim=32).to(device)
    
    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()
    
    print("✅ Models loaded successfully")
    
    # Prepare input data
    print("\n📊 Preparing input data...")
    
    # Scene data
    batch_size = 1
    num_ego_candidates = 30
    num_neighbors = 10
    timesteps = 80
    
    # Create dummy scene inputs
    inputs = {
        'ego_agent_past': torch.randn(batch_size, timesteps, 7).to(device),
        'neighbor_agents_past': torch.randn(batch_size, num_neighbors, timesteps, 11).to(device),
        'map_lanes': torch.randn(batch_size, 50, 4).to(device),
        'map_crosswalks': torch.randn(batch_size, 10, 4).to(device),
        'route_lanes': torch.randn(batch_size, 20, 4).to(device)
    }
    
    # Ego trajectory candidates (30 different plans)
    ego_traj_candidates = torch.randn(batch_size, num_ego_candidates, timesteps, 6).to(device)
    
    print("✅ Input data prepared")
    
    # Encode scene
    print("\n🧠 Encoding scene...")
    with torch.no_grad():
        encoder_outputs = encoder(inputs)
    print("✅ Scene encoded")
    
    # Generate adversarial scenarios using Latent Space Optimization
    print("\n🎯 Generating adversarial scenarios with Latent Space Optimization...")
    
    try:
        adversarial_scenarios, adversarial_scores = adversarial_scenario_generation(
            decoder=decoder,
            encoder_outputs=encoder_outputs,
            ego_traj_inputs=ego_traj_candidates,
            agents_states=inputs['neighbor_agents_past'],
            timesteps=timesteps,
            method='optimization',  # Use gradient optimization
            optimization_steps=100,  # Number of gradient ascent steps
            num_samples=10  # Not used for optimization method
        )
        
        print("✅ Adversarial scenarios generated successfully!")
        print(f"   Scenarios shape: {adversarial_scenarios.shape}")
        print(f"   Scores shape: {adversarial_scores.shape}")
        print(f"   Max collision risk: {adversarial_scores.max().item():.4f}")
        print(f"   Mean collision risk: {adversarial_scores.mean().item():.4f}")
        
        # Analyze results
        print("\n📈 Analyzing results...")
        
        # Find most dangerous scenario
        max_score_idx = adversarial_scores.argmax()
        max_score = adversarial_scores.max().item()
        
        print(f"   Most dangerous scenario score: {max_score:.4f}")
        print(f"   This represents a {max_score:.1f}x increase in collision risk")
        
        # Compare with random sampling
        print("\n🔄 Comparing with random sampling...")
        
        random_scenarios, random_scores = adversarial_scenario_generation(
            decoder=decoder,
            encoder_outputs=encoder_outputs,
            ego_traj_inputs=ego_traj_candidates,
            agents_states=inputs['neighbor_agents_past'],
            timesteps=timesteps,
            method='random',  # Use random sampling
            num_samples=100
        )
        
        random_max = random_scores.max().item()
        improvement = ((max_score - random_max) / random_max * 100) if random_max > 0 else 0
        
        print(f"   Random sampling max score: {random_max:.4f}")
        print(f"   Optimization max score: {max_score:.4f}")
        print(f"   Improvement: {improvement:.2f}%")
        
        if improvement > 0:
            print("🎉 Latent Space Optimization successfully generates more dangerous scenarios!")
        else:
            print("⚠️  Optimization may need parameter tuning")
            
    except Exception as e:
        print(f"❌ Error during adversarial generation: {e}")
        print("   This might be due to model compatibility issues")
        print("   Try using the fallback random sampling method")
        
        # Fallback to random sampling
        print("\n🔄 Falling back to random sampling...")
        try:
            adversarial_scenarios, adversarial_scores = adversarial_scenario_generation(
                decoder=decoder,
                encoder_outputs=encoder_outputs,
                ego_traj_inputs=ego_traj_candidates,
                agents_states=inputs['neighbor_agents_past'],
                timesteps=timesteps,
                method='random',
                num_samples=100
            )
            print("✅ Random sampling fallback successful")
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")

def compare_methods():
    """Compare different optimization parameters."""
    print("\n🔧 Comparing Different Optimization Parameters...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create minimal test setup
    encoder = GNNEncoder(node_dim=11, dim=256, heads=4, layers=2).to(device)
    decoder = GGTP_Decoder(neighbors=5, max_time=4, max_branch=10, latent_dim=16).to(device)
    
    inputs = {
        'ego_agent_past': torch.randn(1, 20, 7).to(device),
        'neighbor_agents_past': torch.randn(1, 5, 20, 11).to(device),
        'map_lanes': torch.randn(1, 20, 4).to(device),
        'map_crosswalks': torch.randn(1, 5, 4).to(device),
        'route_lanes': torch.randn(1, 10, 4).to(device)
    }
    
    ego_traj_candidates = torch.randn(1, 10, 20, 6).to(device)
    
    with torch.no_grad():
        encoder_outputs = encoder(inputs)
    
    # Test different optimization steps
    steps_list = [25, 50, 100]
    
    for steps in steps_list:
        print(f"\n   Testing with {steps} optimization steps...")
        try:
            scenarios, scores = adversarial_scenario_generation(
                decoder=decoder,
                encoder_outputs=encoder_outputs,
                ego_traj_inputs=ego_traj_candidates,
                agents_states=inputs['neighbor_agents_past'],
                timesteps=20,
                method='optimization',
                optimization_steps=steps
            )
            
            max_score = scores.max().item()
            mean_score = scores.mean().item()
            
            print(f"   ✅ Success!")
            print(f"      Max score: {max_score:.4f}")
            print(f"      Mean score: {mean_score:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    print("🚀 Latent Space Optimization Example")
    print("=" * 50)
    
    # Run main example
    example_usage()
    
    # Run comparison
    compare_methods()
    
    print("\n✨ Example completed!")
    print("\n💡 Key Takeaways:")
    print("   1. Latent Space Optimization uses gradient ascent to find dangerous scenarios")
    print("   2. It should generate higher collision risk than random sampling")
    print("   3. More optimization steps generally lead to better results")
    print("   4. The method implements the paper's z* = argmax_z R(p_θ(τ^i | z, c^i), τ^0)")

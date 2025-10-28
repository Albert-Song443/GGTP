# GGTP: Generative Game-Theoretic Prediction for Adversarial Autonomous Driving Validation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📖 Overview

**GGTP** addresses the critical challenge of autonomous vehicle safety validation by generating **adversarial yet realistic** traffic scenarios. Unlike traditional methods that rely on random perturbations or manually designed edge cases, GGTP uses a game-theoretic approach combining Graph Neural Networks (GNN) and Conditional Variational Autoencoders (CVAE) to intelligently create challenging test scenarios.

### Key Results
- **8x increase** in collision risk while maintaining behavioral realism
- **21% reduction** in prediction error (ADE) compared to GraphSAGE baseline
- **67.8% success rate** in generating valid adversarial scenarios
- **39% improvement** over LSTM-based predictors

### Core Innovation

GGTP introduces three key innovations:

1. **GNN Scene Encoder** - Graph Attention Network (GAT) captures spatial agent interactions as a graph
2. **Opponent Payoff Embedding** - Models game-theoretic decision context by aggregating neighbor embeddings
3. **Latent Space Optimization** - Gradient-based search in CVAE latent space to generate adversarial scenarios

---

## 🏗️ Architecture

### System Components

```
Traffic Scene → GNN Encoder → Opponent Payoff → CVAE Decoder → Trajectory Predictions
                    ↓              Embedding           ↓
                Graph Build         ↓              Multi-modal
                (Nodes/Edges)    Game Context      Distribution
```

### 1. GNN Scene Encoder

**Purpose**: Model traffic scenes as interaction graphs

**Implementation**:
- **Nodes**: Traffic agents (ego vehicle + neighbors)
- **Edges**: Spatial proximity (< 50m distance threshold)
- **Architecture**: 2-layer Graph Attention Network (GAT) with 8 attention heads
- **Output**: Context-aware embeddings `h_i` for each agent

**Key Equation**:
```python
h'_i = σ(Σ_{j∈N(i)} α_ij W h_j)  # Attention-weighted aggregation
```

### 2. Opponent Payoff Embedding

**Purpose**: Model game-theoretic interactions between agents

**Innovation**: Instead of only considering physical states, we model each agent's perception of others' intentions:

```python
# Traditional approach
condition = [agent_state, ego_plan]

# GGTP approach
payoff_embedding = Aggregate([h_j for j in Neighbors(i)])  # From GNN
condition = [agent_state, ego_plan, payoff_embedding]
```

**Intuition**: Neighbor embeddings encode their "payoffs" in the local traffic game, allowing prediction of interactive behaviors like yielding, competing, or lane-changing.

### 3. CVAE Generator

**Purpose**: Generate multi-modal, realistic trajectory distributions

**Training (Posterior)**:
```python
# Encode ground truth to posterior distribution
μ, log_σ = Encoder(ground_truth_traj, condition)

# Reparameterization trick
z = μ + σ ⊙ ε, where ε ~ N(0, I)

# Decode to trajectory
pred_traj = Decoder(z, condition)

# ELBO Loss
Loss = MSE(pred_traj, ground_truth) + β × KL(q(z|x,c) || N(0,I))
```

**Inference (Prior)**:
```python
# Sample from prior for diversity
z ~ N(0, I)
pred_traj = Decoder(z, condition)
```

### 4. Adversarial Generation via Latent Optimization

**Purpose**: Find high-risk scenarios through gradient-based optimization

**Method**:
```python
# Initialize latent variable with gradients
z = torch.randn(latent_dim, requires_grad=True)
optimizer = Adam([z], lr=0.01)

# Gradient ascent to maximize collision risk
for step in range(100):
    # Generate trajectory from current z
    neighbor_traj = Decoder(z, condition)
    
    # Calculate collision risk
    risk = compute_collision_risk(neighbor_traj, ego_traj)
    
    # Maximize risk (minimize negative risk)
    loss = -risk
    loss.backward()
    optimizer.step()

# Result: z* that generates most dangerous scenario
```

**Risk Function**:
```python
Risk = 10.0 × Near_Collision(d < 2m)     # Critical proximity
     + 5.0  × Proximity_Score             # exp(-0.5 × min_distance)
     + 3.0  × TTC_Score                   # Time-to-collision
     + 2.0  × Critical_Zone               # Intersection/merge areas
```

---

## 🚀 Quick Start

### ⚠️ System Requirements

**IMPORTANT**: GGTP requires **Ubuntu/Linux** environment due to nuPlan devkit dependencies.

- **Recommended**: Ubuntu 20.04 or 22.04
- **Windows**: Not supported (nuPlan devkit has Linux-specific dependencies like `fcntl`)
- **macOS**: May work but not officially tested

### Installation (Ubuntu/Linux)

```bash
# Clone repository
git clone https://github.com/Albert-Song443/GGTP.git
cd GGTP

# Create conda environment (Python 3.9 recommended for nuPlan compatibility)
conda create -n ggtp python=3.9
conda activate ggtp

# Install PyTorch using pip (recommended for WSL2/Ubuntu compatibility)
# For CUDA 11.7:
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# For CPU only:
# pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install PyTorch Geometric
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Install all other dependencies
pip install -r requirements.txt
```

**For Windows Users**:
- Use WSL2 (Windows Subsystem for Linux) with Ubuntu
- Or use a virtual machine with Ubuntu
- Or use Docker with Ubuntu base image

### Install nuPlan Devkit

nuPlan devkit must be installed from source:

```bash
# Clone nuPlan devkit (in parent directory)
cd ..
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit

# Install nuPlan devkit
pip install -e .

# Return to GGTP directory
cd ../GGTP
```

**Verify installation**:
```bash
python check_environment.py
```

### Download nuPlan Dataset

1. **Register and download** from [nuPlan website](https://www.nuscenes.org/nuplan#download)
2. **Place dataset** in `nuplan/dataset/` directory
3. **Configure dataset** as described in [dataset setup guide](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html)

Your directory structure should look like:
```
GGTP/
├── nuplan/
│   └── dataset/
│       ├── maps/
│       │   ├── nuplan-maps-v1.0.json
│       │   ├── us-nv-las-vegas-strip/
│       │   └── ...
│       └── nuplan-v1.1/
│           └── splits/
│               ├── mini/
│               ├── train/
│               └── val/
```

### Data Processing

Before training the GGTP model, preprocess the raw data from nuPlan:

```bash
python data_process.py \
    --data_path nuplan/dataset/nuplan-v1.1/splits/mini \
    --map_path nuplan/dataset/maps \
    --save_path nuplan/processed_data
```

**Required Arguments**:
- `--data_path`: Path to the stored nuPlan dataset 
- `--map_path`: Path to the nuPlan map data
- `--save_path`: Path to save the processed data

**Optional Arguments**:
- `--total_scenarios`: Limit number of scenarios to process (useful for quick testing)

**Output**: Processed `.npz` files in `nuplan/processed_data/` containing:
- Agent trajectories (past 2s + future 8s at 10Hz)
- Map information (lanes, crosswalks, route)
- Ego trajectory candidates for conditional prediction

### Training

**GGTP uses a unified two-stage training script** that automatically handles both stages:

```bash
python train_adversarial_ggtp.py \
    --train_set nuplan/processed_data/train \
    --valid_set nuplan/processed_data/valid \
    --name Adversarial_GGTP_Experiment \
    --stage1_epochs 20 \
    --stage2_epochs 30 \
    --adversarial_weight 10.0 \
    --realism_weight 1.0 \
    --batch_size 8 \
    --device cuda
```

**Training Process**:
- **Stage 1 (Epochs 1-20)**: Learns realistic traffic behaviors using standard CVAE loss
- **Stage 2 (Epochs 21-50)**: Fine-tunes for adversarial generation while maintaining realism

**Expected Output**:
```
[Stage 1] Epoch 1/20 - Realism Training
loss=142.34, recon=135.21, kld=7.13
plannerADE: 3.45, predictorADE: 2.87

[Stage 1] Epoch 20/20
loss=48.23, recon=45.10, kld=3.13
plannerADE: 1.95, predictorADE: 1.65

[Stage 2] Epoch 1/30 - Adversarial Fine-tuning
loss=52.45, recon=48.10, kld=2.95, collision_risk=125.34

[Stage 2] Epoch 30/30
loss=35.67, recon=42.15, kld=2.82, collision_risk=897.21
Collision risk increased by 7.2x while maintaining realism!
```

**Alternative**: If you only want realistic prediction (no adversarial):
```bash
python train_ggtp.py \
    --train_set nuplan/processed_data/train \
    --valid_set nuplan/processed_data/valid \
    --train_epochs 30 \
    --device cuda
```

### Generate Adversarial Scenarios

**Important**: Use processed data (`.npz` files), not raw data (`.db` files)

```bash
python generate_adversarial_scenarios.py \
    --model_path training_log/Adversarial_GGTP_Experiment/best_adversarial_model.pth \
    --data_path nuplan/processed_data \
    --num_scenarios 100 \
    --num_samples 10 \
    --visualize \
    --device cuda
```

**Available Arguments**:
- `--model_path`: Path to trained adversarial model (required)
- `--data_path`: Path to **processed** data directory containing `.npz` files (required)
- `--output_dir`: Output directory (default: `adversarial_scenarios`)
- `--num_scenarios`: Number of scenarios to generate (default: 100)
- `--num_samples`: Number of CVAE samples per scenario (default: 10)
- `--batch_size`: Batch size for generation (default: 4)
- `--visualize`: Enable visualization of generated scenarios
- `--visualize_top`: Number of top scenarios to visualize (default: 10)
- `--device`: Device to use (default: `cuda`)

**Note**: You must first complete data processing and model training before generating scenarios.

---

## 📊 Performance

### Prediction Accuracy

| Method | PredADE ↓ | PredFDE ↓ | Miss Rate ↓ |
|--------|-----------|-----------|-------------|
| LSTM | 2.89 m | 5.67 m | 42.0% |
| GraphSAGE | 2.31 m | 4.52 m | 35.0% |
| **GGTP (Ours)** | **1.82 m** | **3.58 m** | **28.0%** |

### Adversarial Generation Quality

| Method | Collision Risk ↑ | Realism Score ↑ | Success Rate ↑ | TTC ↓ |
|--------|------------------|-----------------|----------------|-------|
| Random Sampling | 1.2x | 65% | 38.4% | 4.2s |
| LSTM-Adversarial | 2.3x | 72% | 45.2% | 3.1s |
| GraphSAGE-Adv | 3.8x | 78% | 58.3% | 2.4s |
| **GGTP (Ours)** | **8.0x** | **85%** | **67.8%** | **1.3s** |

**Interpretation**:
- **Collision Risk**: 8x higher than baseline while maintaining realism
- **Realism Score**: 85% scenarios pass kinematic/social feasibility checks
- **Success Rate**: 67.8% of generated scenarios successfully stress-test planners
- **TTC (Time-to-Collision)**: Significantly reduced, indicating more critical scenarios

---

## 🔬 Key Features

### Two-Stage Training Strategy

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: Realism Training (20 epochs)              │
│                                                      │
│  Objective: Learn realistic traffic behavior        │
│  Loss: CVAE ELBO (Reconstruction + KL)              │
│  Dataset: nuPlan training set (822 scenarios)       │
│  Output: Base model with realistic predictions      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ↓ Fine-tune
┌─────────────────────────────────────────────────────┐
│  Stage 2: Adversarial Fine-tuning (30 epochs)       │
│                                                      │
│  Objective: Generate high-risk scenarios            │
│  Loss: ELBO - α×Collision + β×Realism               │
│  Strategy: Maximize risk, maintain plausibility     │
│  Output: Adversarial generator for testing          │
└─────────────────────────────────────────────────────┘
```

### Loss Functions

**Stage 1 - Realism**:
```python
L_realism = MSE(pred, truth) + β × KL(q(z|x,c) || p(z))
```

**Stage 2 - Adversarial**:
```python
L_adversarial = MSE(pred, truth) 
                + β × KL(q(z|x,c) || p(z))
                - α × Collision_Risk(pred, ego)  # Maximize collision
                + γ × Realism_Penalty            # Maintain feasibility
```

**Parameters**:
- `β = 0.1`: KL divergence weight (prevents posterior collapse)
- `α = 0.3`: Collision weight (controls adversarial strength)
- `γ = 0.2`: Realism weight (ensures physical plausibility)

### Realism Constraints

To prevent unrealistic adversarial behaviors:

**Kinematic Constraints**:
```python
- Max speed: 25 m/s (90 km/h)
- Max acceleration: 5 m/s²
- Max jerk: 10 m/s³
- Min turning radius: Based on vehicle dynamics
```

**Social Constraints**:
```python
- Agent spacing: 2-50m (no collision, reasonable proximity)
- Lane keeping: Maintain lane structure
- Trajectory smoothness: Continuous heading/velocity
- Interactive behavior: Respect right-of-way
```

**Physical Constraints**:
```python
- No teleportation: Continuous position
- Curvature limits: Realistic steering
- Heading continuity: Smooth orientation changes
```

---

## 📁 Project Structure

```
GGTP/
├── Core Models
│   ├── ggtp_modules.py                # GNN encoder + CVAE decoder
│   ├── adversarial_modules.py         # Adversarial training + latent optimization
│   ├── prediction_modules.py          # Base prediction networks (MLP, attention)
│   └── scenario_tree_prediction.py    # DTPP baseline (for comparison)
│
├── Training & Testing
│   ├── train_ggtp.py                  # Stage 1: Realism training
│   ├── train_adversarial_ggtp.py      # Stage 2: Adversarial fine-tuning
│   ├── test_ggtp_integration.py       # Integration tests
│   ├── test_latent_optimization.py    # Latent optimization tests
│   └── generate_adversarial_scenarios.py  # Scenario generation pipeline
│
├── Data & Utilities
│   ├── data_process.py                # nuPlan data preprocessing
│   ├── data_utils.py                  # Data loading & batching
│   ├── train_utils.py                 # Training helpers (loss, metrics)
│   └── common_utils.py                # Common utilities
│
├── Planning Integration
│   ├── planner_ggtp.py                # GGTP planner (nuPlan integration)
│   ├── planner_utils.py               # Planning utilities
│   ├── path_planner.py                # Spline-based path planning
│   ├── spline_planner.py              # Cubic spline planner
│   └── bezier_path.py                 # Bezier curve generation
│
├── Visualization & Examples
│   ├── road_network_visualizer.py     # Scenario visualization
│   ├── example_latent_optimization.py # Latent optimization demo
│   └── check_environment.py           # Environment setup checker
│
├── Data (gitignored)
│   ├── nuplan/                        # nuPlan dataset
│   │   ├── dataset/                   # Raw data (.db files)
│   │   └── processed_data/            # Processed data (.npz files)
│   └── training_log/                  # Training logs & checkpoints
│
├── Documentation
│   ├── README.md                      # This file
│   ├── LICENSE                        # MIT License
│   └── .gitignore                     # Git ignore rules
```

---

## 🎯 Usage Examples

### Example 1: Basic Training

```bash
# Quick start with default parameters
python train_ggtp.py \
    --train_set nuplan/processed_data/train \
    --valid_set nuplan/processed_data/valid \
    --device cuda
```

### Example 2: Custom Hyperparameters

```bash
# Advanced configuration for better performance
python train_ggtp.py \
--train_set nuplan/processed_data/train \
    --valid_set nuplan/processed_data/valid \
    --name GGTP_HighCapacity \
    --latent_dim 64 \              # Larger latent space (default: 32)
    --beta 0.05 \                  # Lower KL weight (default: 0.1)
    --batch_size 16 \              # Larger batch (default: 32)
    --learning_rate 5e-5 \         # Lower LR for stability
    --train_epochs 50 \            # More epochs
    --device cuda
```

### Example 3: Programmatic Adversarial Generation

```python
import torch
from ggtp_modules import GGTP_Decoder
from adversarial_modules import adversarial_scenario_generation

# Load trained model
decoder = GGTP_Decoder(latent_dim=32, condition_dim=512)
decoder.load_state_dict(torch.load('best_adversarial_model.pth'))
decoder.eval()

# Prepare inputs
encoder_outputs = {...}  # From GNN encoder
ego_trajectories = {...}  # Candidate ego plans
neighbor_history = {...}  # Past neighbor trajectories

# Generate adversarial scenarios
adv_scenarios, adv_scores = adversarial_scenario_generation(
    decoder=decoder,
    encoder_outputs=encoder_outputs,
    ego_traj_candidates=ego_trajectories,
    neighbor_agents_past=neighbor_history,
    timesteps=80,                   # 8 seconds at 10 Hz
    method='optimization',          # Use latent optimization
    optimization_steps=100,         # Gradient ascent steps
    num_optimizations=5,            # Parallel optimizations
    learning_rate=0.01
)

# Select most dangerous scenario
best_idx = torch.argmax(adv_scores)
most_dangerous = adv_scenarios[best_idx]

print(f"Collision risk: {adv_scores[best_idx].item():.2f}")
print(f"Adversarial trajectory shape: {most_dangerous.shape}")
```

### Example 4: Batch Scenario Generation

```python
from generate_adversarial_scenarios import AdversarialScenarioGenerator

# Initialize generator
generator = AdversarialScenarioGenerator(
    model_path='training_log/best_adversarial_model.pth',
    data_path='nuplan/dataset/nuplan-v1.1/splits/val',
    map_path='nuplan/dataset/maps'
)

# Generate 100 adversarial scenarios
scenarios = generator.generate_batch(
    num_scenarios=100,
    optimization_steps=100,
    save_dir='adversarial_scenarios/'
)

print(f"Generated {len(scenarios)} adversarial scenarios")
print(f"Average collision risk: {scenarios['collision_risk'].mean():.2f}")
print(f"Average realism score: {scenarios['realism_score'].mean():.2f}")
```

---

## 🔧 Hyperparameter Guide

### Training Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `latent_dim` | 32 | 16-64 | CVAE latent dimension |
| `beta` | 0.1 | 0.01-1.0 | KL divergence weight |
| `batch_size` | 32 | 8-64 | Training batch size |
| `learning_rate` | 1e-4 | 5e-5 to 5e-4 | Adam learning rate |
| `train_epochs` | 20-30 | 10-50 | Number of epochs |

**Tuning Tips**:
- **Larger `latent_dim`**: Better capacity, higher memory usage
- **Lower `beta`**: More diversity, risk of posterior collapse
- **Higher `beta`**: More regularization, less diversity
- **Smaller `batch_size`**: Better for limited GPU memory

### Adversarial Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `adversarial_alpha` | 0.3 | 0.1-0.5 | Collision weight |
| `realism_beta` | 0.2 | 0.1-0.3 | Realism constraint weight |
| `optimization_steps` | 100 | 50-200 | Gradient ascent iterations |
| `num_optimizations` | 5 | 3-10 | Parallel optimizations |
| `learning_rate` | 0.01 | 0.005-0.05 | Latent optimization LR |

**Tuning Tips**:
- **Higher `adversarial_alpha`**: More aggressive scenarios, may reduce realism
- **Higher `realism_beta`**: More realistic, may reduce adversarial strength
- **More `optimization_steps`**: Better optimization, slower generation
- **More `num_optimizations`**: Higher diversity, more computation

### Common Issues & Solutions

#### Issue 1: KL Divergence Collapses to 0

**Symptom**: `kld` in training log → 0, model becomes deterministic

**Solution**:
```bash
# Reduce beta weight
python train_ggtp.py --beta 0.01

# Or increase latent dimension
python train_ggtp.py --latent_dim 64

# Or use free bits technique (modify code)
kl_loss = max(kl_divergence, free_bits_threshold)
```

#### Issue 2: High Reconstruction Loss

**Symptom**: `recon_loss` >> 100 and not decreasing

**Solution**:
```bash
# Increase model capacity
python train_ggtp.py --latent_dim 64

# Reduce KL weight
python train_ggtp.py --beta 0.05

# Lower learning rate for stability
python train_ggtp.py --learning_rate 5e-5
```

#### Issue 3: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce batch size
python train_ggtp.py --batch_size 8

# Or reduce model size
python train_ggtp.py --latent_dim 16

# Or use gradient checkpointing (modify code)
```

#### Issue 4: Unrealistic Adversarial Scenarios

**Symptom**: Generated trajectories violate physical constraints

**Solution**:
```bash
# Increase realism weight
python train_adversarial_ggtp.py --realism_beta 0.3

# Reduce adversarial weight
python train_adversarial_ggtp.py --adversarial_alpha 0.2

# Add stronger constraints (modify code)
```

---

## 📈 Monitoring Training

### Training Logs

Monitor real-time training progress:

```bash
# View latest training log
tail -f training_log/GGTP_experiment/train.log

# Expected output:
# Epoch 15/30
# loss=45.23, recon=42.10, kld=3.13
# plannerADE: 2.15, plannerFDE: 4.32
# predictorADE: 1.89, predictorFDE: 3.67
# val-loss: 47.56, val-ADE: 2.23
```

### Key Metrics

**Stage 1 (Realism)**:
- **Total Loss**: Should steadily decrease (150 → 50)
- **KL Divergence**: Should stabilize in range 1-10 (not collapse to 0)
- **Reconstruction Loss**: Should decrease (140 → 45)
- **ADE/FDE**: Should improve over epochs

**Stage 2 (Adversarial)**:
- **Collision Risk**: Should increase significantly (target: 5-10x)
- **Realism Score**: Should remain high (> 0.7)
- **Total Loss**: May increase slightly (adversarial vs realism trade-off)
- **Success Rate**: Should be > 50%

### Visualization

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training log
log = pd.read_csv('training_log/GGTP_experiment/train_log.csv')

# Plot training curves
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Loss curves
axes[0,0].plot(log['epoch'], log['loss'], label='Train')
axes[0,0].plot(log['epoch'], log['val-loss'], label='Val')
axes[0,0].set_title('Total Loss')
axes[0,0].legend()

# KL divergence
axes[0,1].plot(log['epoch'], log['kld'])
axes[0,1].set_title('KL Divergence')
axes[0,1].axhline(y=0, color='r', linestyle='--', alpha=0.3)

# Reconstruction loss
axes[0,2].plot(log['epoch'], log['recon'])
axes[0,2].set_title('Reconstruction Loss')

# Planning metrics
axes[1,0].plot(log['epoch'], log['train-planningADE'], label='Train')
axes[1,0].plot(log['epoch'], log['val-planningADE'], label='Val')
axes[1,0].set_title('Planning ADE')
axes[1,0].legend()

# Prediction metrics
axes[1,1].plot(log['epoch'], log['train-predictionADE'], label='Train')
axes[1,1].plot(log['epoch'], log['val-predictionADE'], label='Val')
axes[1,1].set_title('Prediction ADE')
axes[1,1].legend()

# Collision risk (if adversarial training)
if 'collision_risk' in log.columns:
    axes[1,2].plot(log['epoch'], log['collision_risk'])
    axes[1,2].set_title('Collision Risk')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
print("Training curves saved to training_curves.png")
```

---

## 🎓 Citation

If you find this work useful in your research, please cite:

```bibtex
@article{song2024ggtp,
  title={GGTP: Generative Game-Theoretic Prediction for Adversarial Autonomous Driving Validation},
  author={Song, Peiyan and Zhang, Hao},
  journal={arXiv preprint},
  year={2024}
}
```

### Related Work

This project builds upon and extends:

```bibtex
@inproceedings{huang2024dtpp,
  title={DTPP: Differentiable Joint Conditional Prediction and Cost Evaluation for Tree Policy Planning in Autonomous Driving},
  author={Huang, Zhiyu and Karkus, Peter and Ivanovic, Boris and Chen, Yuxiao and Pavone, Marco and Lv, Chen},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={6806--6812},
  year={2024}
}
```

---

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:

- **[DTPP](https://github.com/MCZhi/DTPP)**: Differentiable Tree Policy Planning framework
- **[nuPlan](https://www.nuscenes.org/nuplan)**: Large-scale autonomous driving benchmark
- **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)**: Graph neural network library
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

Special thanks to the autonomous driving research community for their foundational work in behavior prediction, safety validation, and adversarial testing.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔬 Research Disclaimer

**⚠️ Important**: This is a research project for academic purposes. The generated adversarial scenarios are intended for **safety validation and testing only**. Do not use this system for:

- Real-world autonomous vehicle deployment without extensive validation
- Safety-critical applications without proper verification
- Any purpose that could endanger public safety

Always conduct thorough testing and validation before any real-world deployment.

---

<p align="center">
  <strong>Made with ❤️ for safer autonomous driving</strong>
</p>

<p align="center">
  <sub>GGTP: Pushing the boundaries of AV safety through intelligent adversarial testing</sub>
</p>

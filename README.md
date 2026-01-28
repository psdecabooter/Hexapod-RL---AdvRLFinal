# Hexapod Locomotion Learning with PPO

This project implements a custom PyBullet-based hexapod locomotion environment and trains reinforcement learning agents using Proximal Policy Optimization (PPO) to learn goal-directed walking without hand-coded gaits.

## Prerequisites

### System Requirements
- Linux (tested on Ubuntu)
- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster training)

### Required Software
1. **Anaconda/Miniconda** - For environment management
2. **Git** - For cloning the repository

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/psdecabooter/AdvRLFinal.git
cd AdvRLFinal/cleanrl
```

### 2. Create and Activate Conda Environment
```bash
conda create -n cleanrl python=3.10
conda activate cleanrl
```

### 3. Install Dependencies
```bash
pip install -r requirements/requirements.txt
```

Key dependencies include:
- `gymnasium` - RL environment interface
- `pybullet` - Physics simulation
- `torch` - Deep learning framework
- `tensorboard` - Training visualization
- `moviepy` - Video recording

### 4. Verify Installation
```bash
python -c "import gymnasium; import pybullet; import torch; print('Installation successful!')"
```

## Running the Code

### Quick Start - Train with Default Settings

The simplest way to train the hexapod:

```bash
cd /path/to/AdvRLFinal/cleanrl
conda activate cleanrl
./train_hexapod.sh
```

This will:
- Train for 2M timesteps
- Use the **goal-directed** reward (our proposed method)
- Use 4 parallel environments
- Save model checkpoints
- Capture evaluation videos
- Log to TensorBoard

### Training Configuration Options

#### 1. Goal-Directed Reward (Proposed - Default)
```bash
./train_hexapod.sh
# or explicitly:
REWARD_TYPE=goal_directed ./train_hexapod.sh
```

#### 2. Forward Motion Reward (Baseline)
```bash
REWARD_TYPE=forward_motion ./train_hexapod.sh
```

#### 3. Change Random Seed
Edit `train_hexapod.sh` and modify the `--seed` parameter:
```bash
python cleanrl/ppo_continuous_action.py \
    --env-id Hexapod-v0 \
    --total-timesteps 2000000 \
    --seed 42 \                    # Change this
    --num-envs 4 \
    --save-model \
    --capture-video
```

#### 4. Custom Training Parameters
You can directly run the PPO script with custom parameters:
```bash
python cleanrl/ppo_continuous_action.py \
    --env-id Hexapod-v0 \
    --total-timesteps 2000000 \
    --num-envs 4 \
    --learning-rate 3e-4 \
    --seed 1 \
    --save-model \
    --capture-video \
    --track                         # Enable wandb tracking
```

## Monitoring Training

### TensorBoard
Training metrics are automatically logged. To view them:

```bash
tensorboard --logdir=runs
```

Then open your browser to `http://localhost:6006`

Key metrics to monitor:
- `charts/episodic_return` - Total reward per episode
- `charts/distance_to_goal` - How close the robot gets to the goal
- `reward/movement` - Movement reward component
- `reward/velocity` - Velocity reward component
- `reward/goal_bonus` - Bonus for reaching goal
- `reward/stability_penalty` - Penalty for tilting
- `charts/learning_rate` - Current learning rate
- `losses/policy_loss` - Policy network loss
- `losses/value_loss` - Value network loss

### Video Output
Evaluation videos are saved to:
```
videos/Hexapod-v0__ppo_continuous_action__<seed>__<timestamp>-eval/
```

Videos show the robot's behavior during evaluation episodes.

## Project Structure

```
AdvRLFinal/cleanrl/
├── hexapod_env.py              # Custom hexapod environment
├── train_hexapod.sh            # Training script
├── cleanrl/
│   └── ppo_continuous_action.py  # PPO implementation
├── test_description/
│   └── urdf/
│       └── test.urdf           # Hexapod robot URDF model
├── runs/                       # TensorBoard logs (created during training)
├── videos/                     # Evaluation videos (created during training)
└── HEXAPOD_README.md           # This file
```

## Environment Details

### Hexapod Robot
- **18 DOF**: 6 legs × 3 joints (hip, knee, ankle)
- **Observation Space**: 52-dimensional
  - Base position (3D), orientation (4D)
  - Base velocities (6D)
  - Joint positions and velocities (36D)
  - Goal vector (2D), distance to goal (1D)
- **Action Space**: 18-dimensional continuous [-1, 1] per joint
- **Episode Length**: 1000 steps
- **Goal**: Reach a point 1.5m forward from start

### Reward Function

#### Goal-Directed (Proposed)
- Movement: 1000 × (distance_reduction)
- Velocity: 5 × speed × cos(velocity_angle_to_goal)
- Goal bonus: +100 when within 0.5m
- Penalties: stability, height, energy, contact quality

#### Forward Motion (Baseline)
- Movement: 1000 × distance_moved × cos(angle_to_forward)
- Velocity: 10 × speed × cos(velocity_angle_to_forward)
- Same penalties as goal-directed

## Reproducing Paper Results

To reproduce the main experiment comparing goal-directed vs. forward motion:

### Goal-Directed (3 seeds)
```bash
# Seed 1
./train_hexapod.sh

# Seed 564
./train_hexapod.sh

# Seed 923
./train_hexapod.sh
```

### Forward Motion Baseline (3 seeds)
```bash
# Seed 1
REWARD_TYPE=forward_motion ./train_hexapod.sh

# Seed 564
REWARD_TYPE=forward_motion ./train_hexapod.sh

# Seed 923
REWARD_TYPE=forward_motion ./train_hexapod.sh
```

Each run takes approximately .5 hours on a modern CPU with 8 parallel environments.

## Citation

If you use this code in your research, please cite:

```
@misc{huang2025hexapod,
  author = {Huang, Yucheng and Decabooter, Patrick},
  title = {Learning Hexapod Locomotion via Proximal Policy Optimization with Contact-Aware Reward Shaping},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/psdecabooter/AdvRLFinal}
}
```

## License

This project is based on [CleanRL](https://github.com/vwxyzjn/cleanrl) and inherits its MIT license.

## Authors

- Yucheng Huang
- Patrick Decabooter

CS 839: Advanced Reinforcement Learning  
University of Wisconsin-Madison  
Fall 2025

## Acknowledgments

- CleanRL framework by Costa Huang
- PyBullet physics engine
- Gymnasium environment interface

## AI Usage Declaration

This project was developed with assistance from GitHub Copilot (Claude Sonnet 4.5) as a coding assistant. AI was used to help with environment implementation boilerplate, reward function formulation, debugging suggestions, and documentation generation. All AI-generated code was tested, validated, and understood by the authors. The core design decisions, experimental validation, reward engineering iterations, and bug fixes based on real training runs were performed by the human authors. We maintain full understanding of all submitted code and take responsibility for the implementation.

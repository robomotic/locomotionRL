# Mastering Locomotion in MuJoCo: Replicated Tutorial

This project replicates the "Mastering Locomotion in MuJoCo" tutorial by Arjun R.

## Overview
The tutorial covers:
1. Setting up MuJoCo and Gymnasium.
2. Understanding MJCF (MuJoCo XML) models.
3. Simulating a robot with random actions.
4. Training a locomotion policy using Proximal Policy Optimization (PPO).
5. Extending to advanced algorithms: Soft Actor-Critic (SAC) and Recurrent PPO (LSTM).
6. Implementing Domain Randomization for robustness.

## The Robot and Environment: Ant-v5
The primary environment used in this project is **Ant-v5**, a classic benchmark in reinforcement learning provided by Gymnasium (MuJoCo).

### The Robot
The robot is a **3D quadrupedal (four-legged)** agent. It consists of:
- **A torso**: The central body.
- **Four limbs**: Each limb has two links and two joints.
- **8 actuators**: The agent controls the torque applied to each of its 8 joints.

### The Objective
The goal of the agent is to learn to coordinate its limbs to move forward as fast as possible while maintaining stability and minimizing energy consumption (control effort). 

### What's new in v5?
`Ant-v5` is the latest iteration of the environment, offering:
- **Improved physics accuracy**: More stable contact dynamics.
- **Cleaner API**: Modern Gymnasium standards for observation and reward handling.
- **Easier customization**: Better support for overriding simulation parameters (which we use for Domain Randomization).

## Quick Start

1. **Setup the environment**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the environment**:
   ```bash
   source venv/bin/activate
   ```

3. **Run a basic simulation**:
   ```bash
   python scripts/basic_sim.py
   ```

## Pause and Recover Training
All training scripts (`train_ppo.py`, `train_sac.py`, `train_recurrent_ppo.py`) now support pausing and recovering. 
- To **pause**: Simply stop the script (e.g., Ctrl+C).
- To **recover**: Run the same training script again. It will automatically detect the latest checkpoint or final model and resume training from that point.

## Directional Control (WASD Style)
You can now train the robot to follow a specific direction command (Forward, Backward, Left, Right).
### Training
To train a directional model, use the `--directional` flag:
```bash
python scripts/train_ppo.py --directional
```
This adds a 2D goal vector to the observation space and modifies the reward to encourage following that goal. During training, the goal changes randomly every 200 steps to force the agent to learn all directions.

### Evaluation
To evaluate a directional model:
```bash
python scripts/enjoy_models.py --algo ppo --directional
```
*Note: Interactive keyboard control (W/A/S/D) during evaluation depends on the local rendering environment setup. The script currently defaults to "Forward" but can be easily extended for full interactivity.*

## Navigation
- `setup.sh`: Bash script to automate virtual environment and dependency setup.
- `scripts/basic_sim.py`: Basic simulation script with random actions.
- `scripts/train_ppo.py`: PPO training script for the Ant-v5 environment.
- `scripts/train_sac.py`: SAC training script with Domain Randomization.
- `scripts/train_recurrent_ppo.py`: Recurrent PPO (LSTM) training script with Domain Randomization.
- `scripts/enjoy_models.py`: Universal evaluation script for PPO, SAC, and RecurrentPPO.
- `scripts/record_video.py`: Script to record agent performance to a video file.
- `scripts/utils/domain_randomization.py`: Gymnasium wrapper for MuJoCo domain randomization.
- `models/simple_biped.xml`: Educational MJCF model of a bipedal robot.
- `CLOUD.md`: Comprehensive report on scaling training using AWS and GPUs.

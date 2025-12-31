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

## Directional Control: Separate Policies

The approach for directional locomotion has been redesigned. Instead of training a single goal-conditioned policy with direction vectors in the observation space, we now train **separate policies** for each direction. This approach is simpler, more effective, and doesn't require augmenting the observation space.

### How It Works
- **Non-directional mode**: Uses the standard Gymnasium reward function (forward walking reward)
- **Directional mode**: Trains dedicated policies for specific directions:
  - **2D robots (Walker2d, Hopper)**: `forward`, `backward`
  - **3D robots (Ant)**: `forward`, `backward`, `left`, `right`

### Training a Single Direction
```bash
# Train Walker2d to walk forward (standard behavior)
python scripts/train_ppo.py --env Walker2d-v5

# Train Walker2d to walk backward
python scripts/train_ppo.py --env Walker2d-v5 --direction backward

# Train Ant to strafe left
python scripts/train_ppo.py --env Ant-v5 --direction left
```

### Training All Directions
To train separate policies for all valid directions automatically:
```bash
# For Walker2d: trains forward and backward policies
python scripts/train_ppo.py --env Walker2d-v5 --all-directions

# For Ant: trains forward, backward, left, and right policies
python scripts/train_ppo.py --env Ant-v5 --all-directions
```

### Advanced Training with Configurations
To train with high parallelism, terrain curriculum, and custom hyperparameters from JSON configurations:
```bash
python3 scripts/train_ppo.py \
    --env Walker2d-v5 \
    --all-directions \
    --n-envs 16 \
    --terrain \
    --ppo-config configs/ppo_config.json \
    --timesteps 10000000
```
- **JSON Configs**: The script automatically loads `configs/walker2d_config.json` for environment parameters and uses the specified `--ppo-config` for algorithm hyperparameters.
- **Explaination**: See `configs/ppo_config_explanation.md` for a detailed description of each PPO parameter.

### Recording Directional Models
Record high-quality videos and optimized GIFs for your trained directional models:
```bash
# Record forward policy on flat ground
python scripts/record_video.py --env Walker2d-v5 --algo ppo --direction forward --slope 0

# Record backward policy on a 5-degree slope
python scripts/record_video.py --env Walker2d-v5 --algo ppo --direction backward --slope 5
```
- **Output**: Files are saved in `videos/` as both `.mp4` and `.gif`.
- **Slopes**: Use the `--slope` argument to test how directional models handle different inclinations.

### Evaluation: Single Direction
```bash
# Test the backward-walking Walker2d
python scripts/enjoy_models.py --env Walker2d-v5 --direction backward

# Test with floor slope
python scripts/enjoy_models.py --env Walker2d-v5 --direction forward --slope 10
```

### Evaluation: Interactive Mode
Switch between all trained directional policies using the keyboard:
```bash
python scripts/enjoy_models.py --env Ant-v5 --interactive
```
- **W/A/S/D or Arrow Keys**: Switch to the corresponding direction's policy
- **+/-**: Adjust floor inclination in real-time

### Analytical Metrics
To evaluate direction-specific policies:
```bash
python scripts/evaluate_metrics.py --algo ppo --env Walker2d-v5 --direction backward --episodes 50
```

## Example: Training the Hopper
The **Hopper-v5** is a 2D one-legged robot. It is a great environment to start with because it trains much faster than the Ant.

### 1. Start Training
Train both forward and backward policies for the Hopper:
```bash
python scripts/train_ppo.py --env Hopper-v5 --all-directions
```
Or train just the forward direction (standard behavior):
```bash
python scripts/train_ppo.py --env Hopper-v5
```

### 2. Monitor in Real-Time
While the script is training, open a **new terminal** and run the dashboard to see the progress:
```bash
python scripts/training_dashboard.py --env Hopper-v5
```
The dashboard will show live updates of:
- **Total Flips**: How many times the robot fell over.
- **Straight Meters**: Cumulative distance traveled toward the target.
- **Action Jitter**: Smoothness of the movements.
- **Stability**: Average angular velocity (lower is better).

### 3. Test Your Model
Once training is finished (or reached a good level), test it:
```bash
# Test forward policy
python scripts/enjoy_models.py --env Hopper-v5

# Test backward policy
python scripts/enjoy_models.py --env Hopper-v5 --direction backward
```

### 4. Record a Video
To save a video of your robot's performance:
```bash
python scripts/record_video.py --env Hopper-v5
```
You can even record it on a slope:
```bash
python scripts/record_video.py --env Hopper-v5 --direction backward --slope 15
```
The videos will be saved in the `videos/` directory as both **MP4** and **GIF** formats. The GIF is automatically generated with optimized settings (480px width, 10 fps) for easy sharing.


## Using the Training Dashboard
The dashboard is designed to help you diagnose training issues without waiting for the full process to finish.

- **Check Jitter**: If "Action Jitter" is very high, it means the robot's movements are jerky. You might need to increase the `ent_coef` (entropy coefficient) or add an action smoothness penalty.
- **Watch the Slopes**: If you enable terrain curriculum (`--terrain`), keep an eye on "Straight Meters". If it levels off when a slope is introduced, the robot might be stuck and unable to climb.
- **Stability**: A decreasing "Avg. Ang. Vel" usually indicates the robot is becoming more stable and balanced.

## Hyperparameter Tuning
Optimizing training parameters is essential for achieving high-performance locomotion. This project leverages best practices for PPO and SAC.

### Key Strategies:
- **Learning Rate**: Typical values range from `1e-4` to `1e-3`. For PPO, a **linear decay schedule** (starting high and decreasing over time) often leads to better convergence.
- **Batch Size**: Larger batches (256-512) stabilize gradients, especially in environments with complex contact physics like the Ant.
- **Automated Optimization**: We recommend using **Optuna** for Bayesian hyperparameter optimization to automatically find the best values for `n_steps`, `learning_rate`, and `ent_coef`.

For a deep dive into tuning these parameters, see the [Hyperparameter Tuning Guide](hyperparameter_tuning.md).

## Future Research & Scaling
Inspired by the state-of-the-art paper *"Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning"* ([Arxiv](https://arxiv.org/abs/2109.11978)), here are suggestions for taking this project to the next level:

1.  **Massive Parallelism (GPU Simulation)**
    *   **Current**: We use 8 parallel CPU environments.
    *   **Goal**: Scale to **4,000+ environments**.
    *   **How**: Migrate to **MuJoCo MJX** (JAX-based) or **Gym-Isaac** (NVIDIA). This allows the entire simulation to run on the GPU, eliminating CPU bottlenecks and enabling training in minutes rather than hours.

2.  **Curriculum Learning**
    *   **Idea**: Instead of training on the final task immediately, progressively increase difficulty.
    *   **Implementation**: Start with flat ground and easy directional goals. Slowly introduce rougher terrain (via height-fields) and faster goal changing frequencies as the agent improves.

3.  **Asymmetric Actor-Critic**
    *   **Idea**: Give the "Critic" (Teacher) more information than the "Actor" (Student).
    *   **Implementation**: The Critic sees exact friction coefficients, terrain height-maps, and robot mass. The Actor only sees noisy sensor data (joint angles, IMU). This creates a robust policy that can adapt to the real world without "cheating".

## Navigation
- `setup.sh`: Bash script to automate virtual environment and dependency setup.
- `scripts/basic_sim.py`: Basic simulation script with random actions.
- `scripts/train_ppo.py`: PPO training script for the Ant-v5 environment.
- `scripts/train_sac.py`: SAC training script with Domain Randomization.
- `scripts/train_recurrent_ppo.py`: Recurrent PPO (LSTM) training script with Domain Randomization.
- `scripts/enjoy_models.py`: Universal evaluation script with **interactive 3D arrow guidance and WASD control**.
- `scripts/evaluate_metrics.py`: Headless evaluation script for calculating **straight-line efficiency and stability metrics**.
- `scripts/record_video.py`: Script to record agent performance to a video file.
- `scripts/training_dashboard.py`: Real-time monitoring dashboard for tracking training progress.
- `scripts/utils/directional_control.py`: Wrapper for directional training with **Cross-Track Error (CTE)** and **Heading Alignment** rewards.
- `scripts/utils/domain_randomization.py`: Gymnasium wrapper for MuJoCo domain randomization.
- `scripts/utils/terrain.py`: Curriculum wrapper for progressive floor inclination.
- `scripts/utils/callbacks.py`: Custom SB3 callbacks for logging locomotion-specific metrics.
- `models/simple_biped.xml`: Educational MJCF model of a bipedal robot.
- `CLOUD.md`: Comprehensive report on scaling training using AWS, Lambda Labs, RunPod, and Vast.ai.
- `hyperparameter_tuning.md`: Strategies for optimizing RL agent performance.

# Hyperparameter Tuning Guide for Locomotion RL

Optimizing hyperparameters is crucial for squeezing the best performance out of Reinforcement Learning (RL) algorithms. This guide outlines strategies for manual tuning, learning rate scheduling, and automated optimization using Optuna for **PPO** and **SAC** in Stable Baselines3 (SB3).

## 1. Key Hyperparameters to Tune

### Proximal Policy Optimization (PPO)
PPO is generally stable but sensitive to the learning rate and batch size.

| Parameter | Default | Suggested Range | Description |
| :--- | :--- | :--- | :--- |
| `learning_rate` | 3e-4 | 1e-5 – 1e-3 | Step size for specific updates. Too high = unstable; too low = slow. |
| `n_steps` | 2048 | 1024 – 8192 | Steps per environment per update. Longer horizons help with sparse rewards. |
| `batch_size` | 64 | 64 – 512 | Minibatch size. Larger batches stabilize gradients but need more memory. |
| `n_epochs` | 10 | 3 – 20 | Passes over the buffer per update. More epochs = more sample efficiency but higher risk of overfitting. |
| `ent_coef` | 0.0 | 0.0 – 0.01 | Entropy coefficient. Increases exploration. Increase if policy converges prematurely. |

### Soft Actor-Critic (SAC)
SAC is sample-efficient and robust, but tuning the temperature and network size can help.

| Parameter | Default | Suggested Range | Description |
| :--- | :--- | :--- | :--- |
| `learning_rate` | 3e-4 | 1e-4 – 1e-3 | Q-Network and Policy learning rate. |
| `buffer_size` | 1e6 | 1e5 – 1e6 | Replay buffer size. Reduce if running out of RAM. |
| `batch_size` | 256 | 256 – 1024 | Batch size for updates. |
| `tau` | 0.005 | 0.005 – 0.02 | Target smoothing coefficient. |
| `train_freq` | 1 | 1 – 64 | Frequency of training steps. |
| `gradient_steps`| 1 | 1 – 64 | Gradient steps per training step. |

## 2. Learning Rate Scheduling

A dynamic learning rate can improve convergence. Start high to explore, then decrease to fine-tune.

### Built-in Linear Schedule
SB3 allows passing a function for the learning rate.
```python
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

model = PPO("MlpPolicy", env, learning_rate=linear_schedule(1e-3))
```

## 3. Automated Tuning with Optuna

**Optuna** is the state-of-the-art framework for automated hyperparameter optimization. It uses Bayesian optimization to efficiently search the parameter space.

### Workflow
1.  **Define Objective**: A function that takes a `trial` object, samples hyperparameters, trains a model, and returns a score (e.g., mean reward).
2.  **Pruning**: Stop unpromising trials early to save compute.
3.  **Optimize**: Run the study for a set number of trials or time.

### Quick Start Example (Conceptual)

```python
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def objective(trial):
    # 1. Sample Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    
    # 2. Create Model and Train
    model = PPO("MlpPolicy", "Ant-v5", learning_rate=lr, n_steps=n_steps)
    model.learn(total_timesteps=100000)
    
    # 3. Evaluate
    mean_reward, _ = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

## 4. Practical Strategy for Locomotion RL

1.  **Baseline**: Run with default Parameters (`lr=3e-4`).
2.  **Coarse Search**: Use Optuna to tune `learning_rate` and `n_steps` on a shorter run (e.g., 1M steps).
3.  **Scheduling**: Once a good initial LR is found, implement a linear or cosine decay schedule starting from that value.
4.  **Fine-tuning**: If the policy is jittery, increase `batch_size` or decrease `learning_rate`. If it's stuck, increase `ent_coef`.

# PPO Parameter Explanations

This document explains the parameters used in the PPO (Proximal Policy Optimization) configuration.

| Parameter | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `learning_rate` | float | `0.0003` (3e-4) | The step size for the optimizer. A higher value leads to faster learning but can cause instability. |
| `n_steps` | int | `2048` | The number of steps to run for each environment per update (i.e., rollout length). |
| `batch_size` | int | `256` | The number of samples per gradient update. Should be a divisor of `n_steps * n_envs`. |
| `n_epochs` | int | `10` | The number of times to pass through the rollout buffer for each update. |
| `gamma` | float | `0.99` | The discount factor for future rewards. Values close to 1.0 favor long-term rewards. |
| `gae_lambda` | float | `0.95` | Factor for trade-off of bias vs variance for Generalized Advantage Estimator. |
| `clip_range` | float | `0.2` | The clipping parameter for the surrogate objective, restricting how much the policy can change in one update. |
| `ent_coef` | float | `0.01` | Entropy coefficient for the loss calculation. Encourages exploration by penalizing a policy that is too certain. |

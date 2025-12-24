# Cloud Scaling Report: Locomotion RL

To significantly speed up training from your local machine, you should focus on three areas: **Hardware Selection**, **Parallelization**, and **Cloud Provider**.

## 1. Recommended AWS GPU Instances

| Instance Family | GPU Type | Use Case | Recommendation |
| :--- | :--- | :--- | :--- |
| **G5** | NVIDIA A10G | Best cost/performance for standard PPO/SAC. | **Top Choice** for prototyping. |
| **G6 / G6e** | NVIDIA L4 / L40S | Latest Gen. 2x performance for robotics sim. | **Best Current Gen** for speed. |
| **P4d / P5** | NVIDIA A100 / H100 | Massive multi-GPU training. | Overkill for simple Ant-v5, but good for complex humanoids. |

> [!TIP]
> Use **EC2 Spot Instances** (e.g., `g5.xlarge` or `g6.xlarge`) to save up to 70-90% on costs. Since we implemented **Pause & Recover**, your training will resume automatically if an instance is reclaimed.

---

## 2. Infrastructure Options

### Option A: Amazon EC2 (Control & Cost)
- **What to use**: Deep Learning AMI (Ubuntu 22.04).
- **Pros**: Full control, lowest cost with Spot Instances.
- **Scaling**: Increase the number of environments using SB3's `SubprocVecEnv` to utilize all CPU cores while the GPU handles the neural network.

### Option B: Amazon SageMaker RL (Managed Optimization)
- **What to use**: SageMaker RL Estimators.
- **Pros**: Manages the cluster for you; integrates with **AWS RoboMaker** for high-fidelity parallel simulations.
- **Scaling**: Scales across multiple instances automatically.

---

## 3. Simplified AI Clouds (Easier Alternatives)

If AWS feels too complex or expensive, many RL researchers use "AI-First" clouds. They are often significantly cheaper and come with pre-installed drivers.

### Lambda Labs (Highly Recommended)
- **Top Choice**: Best for "1-click" GPU Ubuntu instances.
- **Pricing**: Often 50-70% cheaper than AWS on-demand (e.g., A10s around $0.60/hr, A100s around $1.29/hr).
- **Lambda Stack**: Comes with PyTorch, CUDA, and Drivers pre-installed and verified.
- **Simplicity**: No complex VPCs or IAM roles. Just launch and SSH.

### RunPod
- **What to use**: Community or Secure Cloud.
- **Pros**: Even cheaper than Lambda; supports "Pods" (Docker containers) which are great for reproducible RL environments.
- **Hibernation**: You can stop a pod but keep the disk, which is cheaper than keeping the GPU.

---

## 4. Technical Speed-up Strategies

### Vectorized Environments
Instead of training on a single `Ant-v5` instance, you can run 16, 32, or 64 environments in parallel.
```python
from stable_baselines3.common.vec_env import SubprocVecEnv
# Use this to scale across many CPU cores on a large G5 instance
env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
```

### Transition to MuJoCo MJX (Hardware Acceleration)
If you need massive scale (thousands of parallel ants):
- **MJX** (MuJoCo XLA) allows the entire physics simulation to run on the GPU using JAX.
- This removes the "CPU-GPU bottleneck" where the CPU simulates and the GPU learns.
- **Speedup**: Can reach millions of steps per second on an A100/H100.

---

## 5. Recommendation for Next Steps
1. Start with a **`g5.xlarge`** instance on EC2 using the **Deep Learning AMI**.
2. Switch `DummyVecEnv` to **`SubprocVecEnv`** in the code to utilize all 4-8 vCPUs.
3. If training is still slow, evaluate **MJX** for purely GPU-based simulation.

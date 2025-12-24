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
- **Top Choice**: Best for "1-click" GPU Ubuntu instances with zero configuration.
- **Why Lambda?**: Often 50-70% cheaper than AWS on-demand. Their "Lambda Stack" ensures PyTorch and CUDA are perfectly matched and pre-installed.
- **Workflow Strategy**:
  1. **Launch**: Select an **A10 (24GB VRAM)** for balance or **A100 (40GB/80GB)** for heavy recurrent training.
  2. **Persistence**: 
     - **CRITICAL**: Use **Cloud Storage** (attached at `/home/ubuntu/storage`). Standard instance disks are ephemeral and data is lost on termination.
     - Save your `.pth` checkpoints and `tensorboard` logs to the persistent mount.
  3. **Connectivity**: 
     - Add your SSH key to the console.
     - Connect via terminal: `ssh ubuntu@<INSTANCE_IP>`.
     - Supports **JupyterLab** directly from the dashboard if you prefer notebooks.
  4. **Management**: Termination is the only way to stop billing; ensure all checkpoints are synced to persistent storage before hitting "Terminate".
- **Simplicity**: No complex VPCs, Security Groups, or IAM roles. Just launch and go.

### RunPod (Community Cloud)
- **Why RunPod?**: Often the cheapest option for high-end consumer GPUs (RTX 4090, 3090) which are excellent for single-agent RL training.
- **Workflow Strategy**:
  1. **Select Pod**: Go to "Community Cloud" for best rates. Look for **RTX 4090** (great for rapid prototyping) or **A6000/A40** (larger VRAM).
  2. **Template**: Use the official **RunPod PyTorch 2.1** template. It comes with CUDA, drivers, and Docker pre-configured.
  3. **Setup**:
     - **Volume**: Allocate at least **20GB** of network storage to persist your dataset and model checkpoints.
     - **Ports**: Ensure SSH (22) and TCP (for TensorBoard) are open.
  4. **Development**:
     - Connect via **VS Code Remote - SSH** for a local-like experience.
     - Run `git clone` to fetch this repository.
     - Install dependencies: `pip install -r requirements.txt`.
  5. **Cost Saving**: Use the "Stop" feature to release the GPU when analyzing results, paying only for storage (~$0.10/GB/month).

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

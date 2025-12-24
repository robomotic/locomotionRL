# Animation Transfer Guide: RL to Game Engines

This guide explains how to transfer the motions developed in this MuJoCo/Stable-Baselines3 project into game engines like **Unity** or **Godot**.

---

## 🚀 Method 1: The "Baked" Workflow (Simulation to FBX)
Best for: Cutscenes, background NPCs, or high-fidelity pre-rendered trailers.

### 1. Unity Workflow (Recommended)
1.  **Install MuJoCo Unity Plugin**: Download the official plugin from the [MuJoCo GitHub](https://github.com/google-deepmind/mujoco_unity).
2.  **Import MJCF**: Import your `.xml` robot model into Unity. The plugin handles the conversion of MuJoCo elements to Unity GameObjects.
3.  **Drive with Data**: Run the simulation in Unity using the plugin or feed joint positions (from your Python logs) into the GameObjects' transforms.
4.  **Record**: Use the **Unity Recorder** package or **Timeline Recorder** to "bake" the simulation into an `.anim` clip.
5.  **Export**: Export the GameObject with its new Animation Clip as an **FBX** file using the "FBX Exporter" package.

### 2. Manual Export (Python/Blender)
1.  Log joint angles ($qpos$) during a successful `enjoy.py` run to a `.csv` or `.json`.
2.  In Blender, use a script to map these joint angles to the `rotation` values of your robot's armature.
3.  Keyframe the armature and export as `.fbx`.

---

## 🧠 Method 2: The "Live Inference" Workflow (Agent in-Engine)
Best for: Interactive characters, dynamic AI enemies, or physics-based gameplay.

### 1. Export Model to ONNX
Game engines cannot run `.zip` or `.pth` files directly. You must convert your Stable-Baselines3 model to **ONNX** format.

```python
import torch as th
from stable_baselines3 import PPO

model = PPO.load("ppo_ant")
# Create dummy input for the observation space
dummy_input = th.randn(1, *model.observation_space.shape)
# Export the policy (actor) network
th.onnx.export(model.policy, dummy_input, "agent.onnx")
```

### 2. Running in Unity (Sentis)
1.  Import `agent.onnx` into Unity.
2.  Use **Unity Sentis** (formerly Barracuda) to run the model.
3.  **Loop**:
    - Script gathers observations (joint angles, velocities, orientation).
    - Model outputs actions (torques or desired positions).
    - Script applies actions to the Unity physics bodies or joint components.

### 3. Running in Godot (NDot/GDScript)
1.  Use the [Godot ONNX](https://github.com/godot-extended-libraries/godot-onnx) module or a custom C# wrapper.
2.  Implement the same observation-action loop in GDScript or C#.

---

## 🦴 Joint Mapping Logic
MuJoCo and Game Engines often use different coordinate systems (Z-Up vs Y-Up).

| Element | MuJoCo (Local) | Game Engine (Bones) |
| :--- | :--- | :--- |
| **Pivot** | Center of Mass | Joint Origin |
| **Orientation** | Quaternions (W,X,Y,Z) | Quaternions (X,Y,Z,W) or Euler |
| **Hierarchy** | Flat (Body > Joint) | Nested (Transform > Child) |

> [!TIP]
> Always ensure your visual mesh in the game engine has the same **resting pose** (TPose) as your MuJoCo model's `initial_qpos` to avoid limb rotation offsets.

---

## 🛠️ Recommended Tools
- **MuJoCo Unity Plugin**: For direct physics parity.
- **Unity Sentis**: For cross-platform neural network inference.
- **Blender**: For rigging/retargeting before engine import.

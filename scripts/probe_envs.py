import gymnasium as gym
import numpy as np

for env_id in ["Ant-v5", "Hopper-v5", "Walker2d-v5"]:
    print(f"\n--- Probimg {env_id} ---")
    try:
        env = gym.make(env_id)
        obs, info = env.reset()
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        
        print(f"Info keys: {list(info.keys())}")
        print(f"Observation shape: {obs.shape}")
        
        # Check qpos/qvel structure
        model = env.unwrapped.model
        data = env.unwrapped.data
        print(f"qpos shape: {data.qpos.shape}")
        print(f"qvel shape: {data.qvel.shape}")
        
        # Check where the 'root' body is (usually the first one after worldbody)
        root_name = model.body(1).name
        print(f"Root body name: {root_name}")
        
        env.close()
    except Exception as e:
        print(f"Error probing {env_id}: {e}")

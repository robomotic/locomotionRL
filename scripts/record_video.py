import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import os

def record():
    model_path = "./models/ppo_ant/ppo_ant_final.zip"
    video_dir = "./videos/"
    
    if not os.path.exists(model_path):
        print("Final model not found. Training for 10,000 steps to demonstrate recording...")
        # Train a quick model if one doesn't exist
        env = gym.make("Ant-v5")
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)
        model.save(model_path)
        env.close()

    # Create a new environment for recording
    env = gym.make("Ant-v5", render_mode="rgb_array")
    
    # Wrap the environment to record video
    env = RecordVideo(env, video_folder=video_dir, name_prefix="ant_performance")
    
    model = PPO.load(model_path)

    obs, info = env.reset()
    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    print(f"Video recorded in {video_dir}")
    env.close()

if __name__ == "__main__":
    record()

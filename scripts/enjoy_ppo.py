import gymnasium as gym
from stable_baselines3 import PPO
import time
import os

def enjoy(model_path=None):
    if model_path is None:
        model_path = "./models/ppo_ant/ppo_ant_final.zip"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run train_ppo.py first or specify a valid path.")
        return

    # Create environment for evaluation
    env = gym.make("Ant-v5", render_mode="human")
    
    # Load the trained model
    model = PPO.load(model_path)

    obs, info = env.reset()
    print("Starting evaluation. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, info = env.reset()
                print("Episode end, resetting...")
            
            # Slow down for visualization
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    enjoy()

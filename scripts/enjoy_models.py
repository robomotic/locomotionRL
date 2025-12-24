import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import time
import os
import argparse

def enjoy(algo, model_path, directional=False):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Helper function for environment creation
    def make_env():
        if directional:
            from utils.directional_control import wrap_directional
            return wrap_directional("Ant-v5", render_mode="human")
        return gym.make("Ant-v5", render_mode="human")

    # Create environment for evaluation
    env = DummyVecEnv([make_env])
    
    # Load normalization stats if they exist
    stats_path = model_path.replace("_final.zip", "_stats.pkl").replace(".zip", "_stats.pkl")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        # Disable reward normalization during evaluation
        env.training = False
        env.norm_reward = False
    
    # Target direction for directional models
    target_dir = np.array([1.0, 0.0]) # Default: Forward (W)
    
    # Load the trained model
    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "sac":
        model = SAC.load(model_path)
    elif algo == "rec_ppo":
        model = RecurrentPPO.load(model_path)
    else:
        print(f"Unknown algorithm: {algo}")
        return

    obs = env.reset()
    print(f"Starting evaluation of {algo} model. Press Ctrl+C to stop.")
    
    # For recurrent models, we need to manage states
    lstm_states = None
    episode_start = True

    try:
        while True:
            # If directional, we need to append the target direction to obs
            if directional:
                # Add target_dir to observation
                extended_obs = np.concatenate([obs, target_dir])
            else:
                extended_obs = obs

            # Get action from the model
            if algo == "rec_ppo":
                action, lstm_states = model.predict(extended_obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = False
            else:
                action, _states = model.predict(extended_obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            # If render_mode is human, we can try to get keyboard input via pygame if available
            # But simpler for a demo is to just print instructions or cycle
            if directional:
                print(f"\rCurrent Goal: {target_dir} (W: [1,0], S: [-1,0], A: [0,1], D: [0,-1])", end="")

            if dones[0]:
                # VecEnv resets automatically, but we might want to reset LSTM states
                lstm_states = None
                episode_start = True
                print("Episode end, resetting...")
            
            # Slow down for visualization
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--directional", action="store_true", help="Whether the model was trained with directional controls")
    args = parser.parse_args()
    
    # Set default paths if not provided
    paths = {
        "ppo": "./models/ppo_ant_dir/ppo_ant_dir_final.zip" if args.directional else "./models/ppo_ant/ppo_ant_final.zip",
        "sac": "./models/sac_ant/sac_ant_final.zip",
        "rec_ppo": "./models/rec_ppo_ant/rec_ppo_ant_final.zip"
    }
    
    model_path = args.path if args.path else paths.get(args.algo)
    
    enjoy(args.algo, model_path, directional=args.directional)

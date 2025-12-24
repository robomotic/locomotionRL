from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import os
import glob
import argparse
from utils.directional_control import wrap_directional
import gymnasium as gym

def make_env(env_id, directional=False):
    def _init():
        if directional:
            return wrap_directional(env_id)
        return gym.make(env_id)
    return _init

def train(directional=False, n_envs=8):
    # Create environment
    env_id = "Ant-v5"
    
    # Use vectorized environments
    env = SubprocVecEnv([make_env(env_id, directional) for _ in range(n_envs)])
    
    # Add normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    log_suffix = "_dir" if directional else ""

    # Directory to save logs and models
    log_dir = f"./logs/ppo_ant{log_suffix}/"
    model_dir = f"./models/ppo_ant{log_suffix}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Path for the latest model
    model_name = "ppo_ant_dir" if directional else "ppo_ant"
    latest_model_path = os.path.join(model_dir, f"{model_name}_final.zip")
    
    # Check for checkpoints if final model doesn't exist
    if not os.path.exists(latest_model_path):
        checkpoints = glob.glob(os.path.join(model_dir, f"{model_name}_model_*.zip"))
        if checkpoints:
            latest_model_path = max(checkpoints, key=os.path.getctime)

    stats_path = os.path.join(model_dir, f"{model_name}_stats.pkl")

    if os.path.exists(latest_model_path):
        print(f"Loading existing model from {latest_model_path}...")
        model = PPO.load(latest_model_path, env=env, tensorboard_log=log_dir)
        if os.path.exists(stats_path):
            print(f"Loading normalization stats from {stats_path}...")
            env = VecNormalize.load(stats_path, env)
        reset_num_timesteps = False
    else:
        print("Starting training from scratch...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256, # Increased from 64 for more stable gradients
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01, # Increased from 0.0 to prevent premature local optima
        )
        reset_num_timesteps = True

    # Setup callbacks
    from stable_baselines3.common.callbacks import CallbackList
    from utils.callbacks import LocomotionMetricsCallback
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_ant_model"
    )
    
    metrics_callback = LocomotionMetricsCallback(log_dir=log_dir)
    callback = CallbackList([checkpoint_callback, metrics_callback])

    # Start training
    print(f"Starting training on {env_id}...")
    total_timesteps = 1000000 
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps
    )

    # Save the final model
    model.save(f"{model_dir}/{model_name}_final")
    env.save(f"{model_dir}/{model_name}_stats.pkl")
    print(f"Training finished. Model saved to {model_dir}/{model_name}_final")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directional", action="store_true", help="Train for WASD-style directional control")
    args = parser.parse_args()
    train(directional=args.directional)

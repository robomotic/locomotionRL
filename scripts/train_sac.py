import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob
import argparse
from utils.domain_randomization import wrap_env
from utils.directional_control import wrap_directional

def make_env(env_id, directional=False):
    def _init():
        if directional:
            from utils.directional_control import wrap_directional
            return wrap_directional(env_id)
        from utils.domain_randomization import wrap_env
        return wrap_env(env_id)
    return _init

def train(directional=False, n_envs=4): # SAC is slower, fewer envs often better
    # Create environment
    env_id = "Ant-v5"
    
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    env = SubprocVecEnv([make_env(env_id, directional) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    log_suffix = "_dir" if directional else ""
    # Directory to save logs and models
    log_dir = f"./logs/sac_ant{log_suffix}/"
    model_dir = f"./models/sac_ant{log_suffix}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Path for the latest model
    model_name = "sac_ant_dir" if directional else "sac_ant"
    latest_model_path = os.path.join(model_dir, f"{model_name}_final.zip")
    stats_path = os.path.join(model_dir, f"{model_name}_stats.pkl")

    if os.path.exists(latest_model_path):
        print(f"Loading existing model from {latest_model_path}...")
        model = SAC.load(latest_model_path, env=env, tensorboard_log=log_dir)
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
        reset_num_timesteps = False
    else:
        print("Starting training from scratch...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            buffer_size=1000000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
        )
        reset_num_timesteps = True

    # Setup callbacks
    from stable_baselines3.common.callbacks import CallbackList
    from utils.callbacks import LocomotionMetricsCallback
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="sac_ant_model"
    )
    
    metrics_callback = LocomotionMetricsCallback(log_dir=log_dir)
    callback = CallbackList([checkpoint_callback, metrics_callback])

    # Start training
    print(f"Starting training on {env_id} with SAC...")
    total_timesteps = 500000 
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

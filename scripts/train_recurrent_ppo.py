import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob
import argparse
from utils.domain_randomization import wrap_env
from utils.directional_control import wrap_directional

def train(directional=False):
    # Create environment with domain randomization
    env_id = "Ant-v5"
    if directional:
        print(f"Training directional-conditioned {env_id}...")
        env = wrap_directional(env_id)
        log_suffix = "_dir"
    else:
        print(f"Training standard {env_id}...")
        env = wrap_env(env_id)
        log_suffix = ""

    # Directory to save logs and models
    log_dir = f"./logs/rec_ppo_ant{log_suffix}/"
    model_dir = f"./models/rec_ppo_ant{log_suffix}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Path for the latest model
    model_name = "rec_ppo_ant_dir" if directional else "rec_ppo_ant"
    latest_model_path = os.path.join(model_dir, f"{model_name}_final.zip")
    
    # Check for checkpoints if final model doesn't exist
    if not os.path.exists(latest_model_path):
        checkpoints = glob.glob(os.path.join(model_dir, f"{model_name}_model_*.zip"))
        if checkpoints:
            latest_model_path = max(checkpoints, key=os.path.getctime)

    if os.path.exists(latest_model_path):
        print(f"Loading existing model from {latest_model_path}...")
        model = RecurrentPPO.load(latest_model_path, env=env, tensorboard_log=log_dir)
        reset_num_timesteps = False
    else:
        print("Starting training from scratch...")
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=128, # Smaller n_steps common for recurrent
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
        reset_num_timesteps = True

    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="rec_ppo_ant_model"
    )

    # Start training
    print(f"Starting training on {env_id} with RecurrentPPO and Domain Randomization...")
    total_timesteps = 100000 
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=reset_num_timesteps
    )

    # Save the final model
    model.save(f"{model_dir}/{model_name}_final")
    print(f"Training finished. Model saved to {model_dir}/{model_name}_final")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directional", action="store_true", help="Train for WASD-style directional control")
    args = parser.parse_args()
    train(directional=args.directional)

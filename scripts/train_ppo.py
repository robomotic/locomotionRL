from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import os
import glob
import argparse
import json
from utils.directional_control import wrap_directional_policy, get_valid_directions
from utils.terrain import TerrainCurriculumWrapper
import gymnasium as gym

def make_env(env_id, direction=None, terrain_curriculum=False, total_timesteps=1000000):
    """
    Factory function to create environment instances.
    
    Args:
        env_id: Gymnasium environment ID
        direction: If specified, use directional policy wrapper for this direction.
                   If None, use standard environment with default reward.
        terrain_curriculum: Enable terrain curriculum learning
        total_timesteps: Total training timesteps (for curriculum scheduling)
    """
    def _init():
        if direction:
            # Use direction-specific policy wrapper
            env = wrap_directional_policy(env_id, direction=direction)
        else:
            # Standard environment with default reward (forward walking)
            env = gym.make(env_id)
            
        if terrain_curriculum:
            env = TerrainCurriculumWrapper(env, total_timesteps=total_timesteps)
        return env
    return _init

def train(env_id="Ant-v5", direction=None, terrain_curriculum=False, n_envs=8, total_timesteps=10000000, ppo_config=None):
    """
    Train a PPO agent.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "Walker2d-v5", "Ant-v5")
        direction: Direction for policy training. Options:
                   - None: Standard forward-walking reward
                   - "forward", "backward": Valid for all environments
                   - "left", "right": Only valid for 3D environments (Ant)
        terrain_curriculum: Enable terrain inclination curriculum
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
    """
    # Validate direction if specified
    if direction:
        valid_dirs = get_valid_directions(env_id)
        if direction not in valid_dirs:
            raise ValueError(f"Invalid direction '{direction}' for {env_id}. Valid: {valid_dirs}")
    
    # Use vectorized environments
    env = SubprocVecEnv([make_env(env_id, direction, terrain_curriculum, total_timesteps) for _ in range(n_envs)])
    
    # Add normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    env_name_clean = env_id.replace("-v5", "").lower()
    log_suffix = ""
    if direction: 
        log_suffix += f"_{direction}"
    if terrain_curriculum: 
        log_suffix += "_terrain"

    # Directory to save logs and models
    log_dir = f"./logs/ppo_{env_name_clean}{log_suffix}/"
    model_dir = f"./models/ppo_{env_name_clean}{log_suffix}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Path for the latest model
    model_name = f"ppo_{env_name_clean}{log_suffix}"
    latest_model_path = os.path.join(model_dir, f"{model_name}_final.zip")
    
    # Check for checkpoints if final model doesn't exist
    if not os.path.exists(latest_model_path):
        checkpoints = glob.glob(os.path.join(model_dir, f"ppo_*_model_*.zip"))
        if checkpoints:
            latest_model_path = max(checkpoints, key=os.path.getctime)

    stats_path = os.path.join(model_dir, f"{model_name}_stats.pkl")

    if os.path.exists(latest_model_path):
        print(f"Loading existing model {latest_model_path}...")
        model = PPO.load(latest_model_path, env=env, tensorboard_log=log_dir)
        if os.path.exists(stats_path):
            print(f"Loading normalization stats from {stats_path}...")
            env = VecNormalize.load(stats_path, env)
        reset_num_timesteps = False
    else:
        print(f"Starting training {env_id} from scratch...")
        if direction:
            print(f"  Direction: {direction}")
        else:
            print(f"  Mode: Standard forward reward")
        # Default PPO params
        ppo_params = {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
        }
        
        # Override with config file if provided
        if ppo_config and os.path.exists(ppo_config):
            print(f"Loading PPO config from {ppo_config}...")
            with open(ppo_config, 'r') as f:
                config_params = json.load(f)
                ppo_params.update(config_params)
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            **ppo_params
        )
        reset_num_timesteps = True

    # Setup callbacks
    from stable_baselines3.common.callbacks import CallbackList
    from utils.callbacks import LocomotionMetricsCallback, TerrainCurriculumCallback
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=f"ppo_{env_name_clean}_model"
    )
    
    metrics_callback = LocomotionMetricsCallback(log_dir=log_dir)
    callbacks = [checkpoint_callback, metrics_callback]
    
    if terrain_curriculum:
        callbacks.append(TerrainCurriculumCallback())
        
    callback = CallbackList(callbacks)

    # Start training
    print(f"Starting training on {env_id}...")
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


def train_all_directions(env_id="Walker2d-v5", terrain_curriculum=False, n_envs=8, total_timesteps=10000000, ppo_config=None):
    """
    Train separate policies for all valid directions.
    
    For 2D robots (Walker2d, Hopper): trains forward and backward
    For 3D robots (Ant): trains forward, backward, left, and right
    """
    directions = get_valid_directions(env_id)
    print(f"Training {len(directions)} directional policies for {env_id}: {directions}")
    
    for direction in directions:
        print(f"\n{'='*60}")
        print(f"Training {direction.upper()} policy")
        print(f"{'='*60}\n")
        train(
            env_id=env_id,
            direction=direction,
            terrain_curriculum=terrain_curriculum,
            n_envs=n_envs,
            total_timesteps=total_timesteps,
            ppo_config=ppo_config
        )
    
    print(f"\nAll {len(directions)} directional policies trained successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for locomotion")
    parser.add_argument("--env", type=str, default="Ant-v5", 
                        help="Gymnasium environment ID (e.g., Hopper-v5, Walker2d-v5, Ant-v5)")
    parser.add_argument("--direction", type=str, default=None,
                        choices=["forward", "backward", "left", "right"],
                        help="Train policy for specific direction. If not specified, uses standard reward.")
    parser.add_argument("--all-directions", action="store_true",
                        help="Train separate policies for all valid directions")
    parser.add_argument("--terrain", action="store_true", 
                        help="Enable progressive terrain inclination curriculum")
    parser.add_argument("--timesteps", type=int, default=10000000, 
                        help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--ppo-config", type=str, default=None,
                        help="Path to PPO configuration JSON file")
    
    args = parser.parse_args()
    
    if args.all_directions:
        train_all_directions(
            env_id=args.env, 
            terrain_curriculum=args.terrain, 
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            ppo_config=args.ppo_config
        )
    else:
        train(
            env_id=args.env, 
            direction=args.direction,
            terrain_curriculum=args.terrain,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            ppo_config=args.ppo_config
        )

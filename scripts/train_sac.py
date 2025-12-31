import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import os
import glob
import argparse
from utils.directional_control import wrap_directional_policy, get_valid_directions
from utils.domain_randomization import wrap_env
from utils.terrain import TerrainCurriculumWrapper

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
            # Standard environment with domain randomization
            env = wrap_env(env_id)
            
        if terrain_curriculum:
            env = TerrainCurriculumWrapper(env, total_timesteps=total_timesteps)
        return env
    return _init

def train(env_id="Ant-v5", direction=None, terrain_curriculum=False, n_envs=4, total_timesteps=10000000):
    """
    Train a SAC agent.
    
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
    
    # Create environment
    env = SubprocVecEnv([make_env(env_id, direction, terrain_curriculum, total_timesteps) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    env_name_clean = env_id.replace("-v5", "").lower()
    log_suffix = ""
    if direction: 
        log_suffix += f"_{direction}"
    if terrain_curriculum: 
        log_suffix += "_terrain"
    
    # Directory to save logs and models
    log_dir = f"./logs/sac_{env_name_clean}{log_suffix}/"
    model_dir = f"./models/sac_{env_name_clean}{log_suffix}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Path for the latest model
    model_name = f"sac_{env_name_clean}{log_suffix}"
    latest_model_path = os.path.join(model_dir, f"{model_name}_final.zip")
    stats_path = os.path.join(model_dir, f"{model_name}_stats.pkl")

    if os.path.exists(latest_model_path):
        print(f"Loading existing model {latest_model_path}...")
        model = SAC.load(latest_model_path, env=env, tensorboard_log=log_dir)
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
        reset_num_timesteps = False
    else:
        print(f"Starting training {env_id} from scratch...")
        if direction:
            print(f"  Direction: {direction}")
        else:
            print(f"  Mode: Standard forward reward")
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
    from utils.callbacks import LocomotionMetricsCallback, TerrainCurriculumCallback
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix=f"sac_{env_name_clean}_model"
    )
    
    metrics_callback = LocomotionMetricsCallback(log_dir=log_dir)
    callbacks = [checkpoint_callback, metrics_callback]
    
    if terrain_curriculum:
        callbacks.append(TerrainCurriculumCallback())
        
    callback = CallbackList(callbacks)

    # Start training
    print(f"Starting training on {env_id} with SAC...")
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


def train_all_directions(env_id="Walker2d-v5", terrain_curriculum=False, n_envs=4, total_timesteps=10000000):
    """
    Train separate policies for all valid directions.
    
    For 2D robots (Walker2d, Hopper): trains forward and backward
    For 3D robots (Ant): trains forward, backward, left, and right
    """
    directions = get_valid_directions(env_id)
    print(f"Training {len(directions)} directional policies for {env_id}: {directions}")
    
    for direction in directions:
        print(f"\n{'='*60}")
        print(f"Training {direction.upper()} policy with SAC")
        print(f"{'='*60}\n")
        train(
            env_id=env_id,
            direction=direction,
            terrain_curriculum=terrain_curriculum,
            n_envs=n_envs,
            total_timesteps=total_timesteps
        )
    
    print(f"\nAll {len(directions)} directional policies trained successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent for locomotion")
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
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    
    args = parser.parse_args()
    
    if args.all_directions:
        train_all_directions(
            env_id=args.env, 
            terrain_curriculum=args.terrain, 
            n_envs=args.n_envs,
            total_timesteps=args.timesteps
        )
    else:
        train(
            env_id=args.env, 
            direction=args.direction,
            terrain_curriculum=args.terrain,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps
        )

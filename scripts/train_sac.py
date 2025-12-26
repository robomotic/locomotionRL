import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import glob
import argparse
from utils.domain_randomization import wrap_env
from utils.directional_control import wrap_directional
from utils.terrain import TerrainCurriculumWrapper

def make_env(env_id, directional=False, terrain_curriculum=False, total_timesteps=1000000):
    def _init():
        if directional:
            from utils.directional_control import wrap_directional
            env = wrap_directional(env_id)
        else:
            from utils.domain_randomization import wrap_env
            env = wrap_env(env_id)
            
        if terrain_curriculum:
            env = TerrainCurriculumWrapper(env, total_timesteps=total_timesteps)
        return env
    return _init

def train(env_id="Ant-v5", directional=False, terrain_curriculum=False, n_envs=4, total_timesteps=10000000):
    # Create environment
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    env = SubprocVecEnv([make_env(env_id, directional, terrain_curriculum, total_timesteps) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    env_name_clean = env_id.replace("-v5", "").lower()
    log_suffix = ""
    if directional: log_suffix += "_dir"
    if terrain_curriculum: log_suffix += "_terrain"
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
    # total_timesteps defined above
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
    parser.add_argument("--env", type=str, default="Ant-v5", help="Gymnasium environment ID")
    parser.add_argument("--directional", action="store_true", help="Train for WASD-style directional control")
    parser.add_argument("--terrain", action="store_true", help="Enable progressive terrain inclination curriculum")
    parser.add_argument("--timesteps", type=int, default=10000000, help="Total training timesteps")
    args = parser.parse_args()
    train(env_id=args.env, directional=args.directional, terrain_curriculum=args.terrain, total_timesteps=args.timesteps)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from gymnasium.wrappers import RecordVideo
import os
import argparse

def record(algo, env_id, model_path, direction=None, slope=0.0, video_length=1000):
    """
    Record a video of a trained model.
    
    Args:
        algo: Algorithm used ("ppo", "sac", "rec_ppo")
        env_id: Gymnasium environment ID
        model_path: Path to the trained model
        direction: If specified, uses directional policy wrapper for this direction.
                   If None, uses standard environment.
        slope: Floor inclination in degrees
        video_length: Number of steps to record
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    video_dir = "./videos/"
    os.makedirs(video_dir, exist_ok=True)
    
    # Helper function for environment creation
    def make_env():
        if direction:
            from utils.directional_control import wrap_directional_policy
            env = wrap_directional_policy(env_id, direction=direction, render_mode="rgb_array")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        
        # Always add terrain wrapper to allow slope adjustments
        from utils.terrain import TerrainCurriculumWrapper
        env = TerrainCurriculumWrapper(env)
        if slope != 0:
            env.current_angle_deg = slope
            env._apply_inclination(slope)
        return env

    # Create the base env
    base_env = make_env()
    
    # Wrap to record video
    video_prefix = f"{env_id}_{algo}"
    if direction: 
        video_prefix += f"_{direction}"
    if slope != 0: 
        video_prefix += f"_slope{int(slope)}"
    
    env = RecordVideo(base_env, video_folder=video_dir, name_prefix=video_prefix,
                      episode_trigger=lambda x: x == 0)  # Record the first episode
    
    # SB3 needs a VecEnv
    venv = DummyVecEnv([lambda: env])
    
    # Load normalization stats if they exist
    stats_path = model_path.replace("_final.zip", "_stats.pkl").replace(".zip", "_stats.pkl")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        venv = VecNormalize.load(stats_path, venv)
        venv.training = False
        venv.norm_reward = False
    
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

    obs = venv.reset()
    print(f"Recording evaluation of {algo} model on {env_id}...")
    if direction:
        print(f"Direction: {direction}")
    if slope != 0:
        print(f"Floor slope: {slope}°")
    
    lstm_states = None
    episode_start = True

    for _ in range(video_length):
        if algo == "rec_ppo":
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = False
        else:
            action, _states = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, infos = venv.step(action)
        
        if dones[0]:
            lstm_states = None
            episode_start = True
            # We only record one continuous video or the first episode
            break

    venv.close()
    print(f"Video recording finished. Check the '{video_dir}' directory.")
    
    # Convert the video to GIF
    try:
        import glob
        try:
            # Try moviepy 2.x import structure first
            from moviepy.video.io.VideoFileClip import VideoFileClip
        except ImportError:
            # Fall back to moviepy 1.x import structure
            from moviepy.editor import VideoFileClip
        
        # Find the most recently created video file
        video_files = glob.glob(os.path.join(video_dir, f"{video_prefix}*.mp4"))
        if video_files:
            latest_video = max(video_files, key=os.path.getctime)
            gif_path = latest_video.replace('.mp4', '.gif')
            
            print(f"\nConverting video to GIF...")
            clip = VideoFileClip(latest_video)
            
            # Optimize GIF: reduce fps and resize if needed
            # For locomotion videos, 10 fps is usually sufficient
            clip_resized = clip.resized(width=480)  # Resize to 480px width for smaller file size
            clip_resized.write_gif(gif_path, fps=10)
            clip.close()
            
            print(f"✓ GIF saved to: {gif_path}")
        else:
            print("Warning: Could not find recorded video file for GIF conversion")
    except ImportError:
        print("\nNote: Install moviepy to enable GIF conversion: pip install moviepy")
    except Exception as e:
        print(f"\nWarning: GIF conversion failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record video of trained locomotion models")
    parser.add_argument("--env", type=str, default="Ant-v5", 
                        help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--direction", type=str, default=None,
                        choices=["forward", "backward", "left", "right"],
                        help="Direction the model was trained for. If not set, uses standard model.")
    parser.add_argument("--slope", type=float, default=0.0, 
                        help="Floor inclination in degrees")
    parser.add_argument("--length", type=int, default=1000, 
                        help="Number of steps to record")
    args = parser.parse_args()
    
    # Set default paths if not provided
    env_name_clean = args.env.replace("-v5", "").lower()
    log_suffix = f"_{args.direction}" if args.direction else ""
    model_name = f"{args.algo}_{env_name_clean}{log_suffix}"
    model_path = args.path if args.path else f"./models/{model_name}/{model_name}_final.zip"
    
    record(args.algo, args.env, model_path, direction=args.direction, slope=args.slope, video_length=args.length)

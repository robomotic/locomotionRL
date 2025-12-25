import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import time
import os
import argparse

def enjoy(algo, env_id, model_path, directional=False, sleep_time=0.01):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Helper function for environment creation
    def make_env():
        # Use None for render_mode so we can use our own passive viewer for markers
        render_mode = "human" if not directional else None
        if directional:
            from utils.directional_control import wrap_directional
            return wrap_directional(env_id, render_mode=render_mode)
        return gym.make(env_id, render_mode=render_mode)

    # Create environment for evaluation
    env = DummyVecEnv([make_env])
    
    # Load normalization stats if they exist
    stats_path = model_path.replace("_final.zip", "_stats.pkl").replace(".zip", "_stats.pkl")
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    
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
    print(f"Starting evaluation of {algo} model on {env_id}. Press Ctrl+C to stop.")
    if directional:
        print("CONTROLS: Use ARROW KEYS to steer the robot. Press 'M' to toggle Manual/Random mode.")
    
    # For recurrent models
    lstm_states = None
    episode_start = True

    # Variables for keyboard control
    manual_mode = directional # Start in manual if directional
    cmd_goal = np.array([1.0, 0.0])

    # Setup for custom visualization if directional
    viewer = None
    if directional:
        import mujoco.viewer
        # Access the base MuJoCo environment through the wrappers
        base_env = env.unwrapped.envs[0].unwrapped
        
        # Key callback for steering (Arrows to avoid MuJoCo conflicts)
        is_2d_robot = "Ant" not in env_id
        def key_callback(keycode):
            nonlocal cmd_goal, manual_mode
            # GLFW Keycodes for Arrows
            # UP: 265, DOWN: 264, LEFT: 263, RIGHT: 262
            if is_2d_robot:
                if keycode in [265, 262, ord('W'), ord('D')]: # Up/Right
                    cmd_goal = np.array([1.0, 0.0])
                elif keycode in [264, 263, ord('S'), ord('A')]: # Down/Left
                    cmd_goal = np.array([-1.0, 0.0])
            else:
                if keycode == 265 or keycode == ord('W'): # Forward
                    cmd_goal = np.array([1.0, 0.0])
                elif keycode == 264 or keycode == ord('S'): # Backward
                    cmd_goal = np.array([-1.0, 0.0])
                elif keycode == 263 or keycode == ord('A'): # Left
                    cmd_goal = np.array([0.0, 1.0])
                elif keycode == 262 or keycode == ord('D'): # Right
                    cmd_goal = np.array([0.0, -1.0])
            
            if 32 <= keycode <= 126 and chr(keycode) == 'M': 
                manual_mode = not manual_mode
                print(f"\nMode switched to: {'MANUAL' if manual_mode else 'RANDOM'}")

        viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data, key_callback=key_callback)
        # Zoom out for better visibility
        viewer.cam.distance = 5.0

    try:
        while True:
            if directional and manual_mode:
                # Override the wrapper's random goal with our keyboard input
                env.unwrapped.envs[0].current_goal = cmd_goal

            # Get action from the model
            extended_obs = obs
            if algo == "rec_ppo":
                action, lstm_states = model.predict(extended_obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = False
            else:
                action, _states = model.predict(extended_obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            if directional and viewer and viewer.is_running():
                # Update the 3D arrow visualization
                directional_wrapper = env.unwrapped.envs[0]
                base_env = directional_wrapper.unwrapped
                goal = directional_wrapper.current_goal
                
                # 1. Camera Follow: Update lookat to torso position
                # data.xpos[1] is the global 3D position of the root body
                viewer.cam.lookat[:] = base_env.data.xpos[1]

                # 2. Center the arrow on the robot
                robot_pos = base_env.data.xpos[1]
                
                # Add a marker (arrow) to the scene
                viewer.user_scn.ngeom = 0
                import mujoco
                # Arrow points from robot_pos in direction of goal
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([0.05, 0.05, 0.4], dtype=np.float64),
                    rgba=np.array([0, 1, 0, 1], dtype=np.float32) if manual_mode else np.array([1, 0, 0, 1], dtype=np.float32), 
                    pos=(robot_pos + [0, 0, 0.5]).astype(np.float64), 
                    mat=np.eye(3).flatten().astype(np.float64)
                )
                
                # Calculate rotation to point in goal direction [dx, dy]
                angle = np.arctan2(goal[1], goal[0])
                c, s = np.cos(angle), np.sin(angle)
                rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
                viewer.user_scn.geoms[0].mat = rot_mat
                
                viewer.sync()
            
            if dones[0]:
                lstm_states = None
                episode_start = True
                print("\nEpisode end, resetting...")
            
            # User defined slow down
            time.sleep(sleep_time)
            
            if viewer and not viewer.is_running():
                break
                
    except KeyboardInterrupt:
        print("\nEvaluation stopped.")
    finally:
        if viewer:
            viewer.close()
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Ant-v5", help="Gymnasium environment ID (e.g., Hopper-v5, Walker2d-v5)")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--directional", action="store_true", help="Whether the model was trained with directional controls")
    parser.add_argument("--sleep", type=float, default=0.01, help="Time to sleep between steps (increase to slow down)")
    args = parser.parse_args()
    
    # Set default paths if not provided
    env_name_clean = args.env.replace("-v5", "").lower()
    log_suffix = "_dir" if args.directional else ""
    model_name = f"{args.algo}_{env_name_clean}{log_suffix}"
    
    model_path = args.path if args.path else f"./models/{model_name}/{model_name}_final.zip"
    
    enjoy(args.algo, args.env, model_path, directional=args.directional, sleep_time=args.sleep)

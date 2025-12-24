import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import time
import os
import argparse

def enjoy(algo, model_path, directional=False, sleep_time=0.01):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Helper function for environment creation
    def make_env():
        # Use None for render_mode so we can use our own passive viewer for markers
        render_mode = "human" if not directional else None
        if directional:
            from utils.directional_control import wrap_directional
            return wrap_directional("Ant-v5", render_mode=render_mode)
        return gym.make("Ant-v5", render_mode=render_mode)

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
    print(f"Starting evaluation of {algo} model. Press Ctrl+C to stop.")
    if directional:
        print("MAPPING: W: Forward, S: Back, A: Left, D: Right")
    
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
        
        # Key callback for steering
        def key_callback(keycode):
            nonlocal cmd_goal, manual_mode
            # MuJoCo keycodes are ASCII or special constants
            if chr(keycode) == 'W': cmd_goal = np.array([1.0, 0.0])
            elif chr(keycode) == 'S': cmd_goal = np.array([-1.0, 0.0])
            elif chr(keycode) == 'A': cmd_goal = np.array([0.0, 1.0])
            elif chr(keycode) == 'D': cmd_goal = np.array([0.0, -1.0])
            elif chr(keycode) == 'M': 
                manual_mode = not manual_mode
                print(f"\nMode switched to: {'MANUAL' if manual_mode else 'RANDOM'}")

        viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data, key_callback=key_callback)

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
                
                # Center the arrow on the ant (root body position)
                ant_pos = base_env.data.qpos[:3]
                
                # Add a marker (arrow) to the scene
                viewer.user_scn.ngeom = 0
                import mujoco
                # Arrow points from ant_pos in direction of goal
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([0.05, 0.05, 0.4], dtype=np.float64),
                    rgba=np.array([0, 1, 0, 1], dtype=np.float32) if manual_mode else np.array([1, 0, 0, 1], dtype=np.float32), 
                    pos=(ant_pos + [0, 0, 0.5]).astype(np.float64), 
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
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--directional", action="store_true", help="Whether the model was trained with directional controls")
    parser.add_argument("--sleep", type=float, default=0.01, help="Time to sleep between steps (increase to slow down)")
    args = parser.parse_args()
    
    # Set default paths if not provided
    paths = {
        "ppo": "./models/ppo_ant_dir/ppo_ant_dir_final.zip" if args.directional else "./models/ppo_ant/ppo_ant_final.zip",
        "sac": "./models/sac_ant/sac_ant_final.zip",
        "rec_ppo": "./models/rec_ppo_ant/rec_ppo_ant_final.zip"
    }
    
    model_path = args.path if args.path else paths.get(args.algo)
    
    enjoy(args.algo, model_path, directional=args.directional, sleep_time=args.sleep)

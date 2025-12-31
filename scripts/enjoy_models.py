import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import time
import os
import argparse

def enjoy(algo, env_id, model_path, direction=None, sleep_time=0.01, slope=0.0):
    """
    Run a trained model for visualization/enjoyment.
    
    Args:
        algo: Algorithm used ("ppo", "sac", "rec_ppo")
        env_id: Gymnasium environment ID
        model_path: Path to the trained model
        direction: If specified, uses directional policy wrapper for this direction.
                   If None, uses standard environment.
        sleep_time: Time to sleep between steps
        slope: Floor inclination in degrees
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Helper function for environment creation
    def make_env():
        render_mode = "human" if not direction else None
        if direction:
            from utils.directional_control import wrap_directional_policy
            env = wrap_directional_policy(env_id, direction=direction, render_mode=render_mode)
        else:
            env = gym.make(env_id, render_mode=render_mode)
        
        # Always add terrain wrapper to allow real-time slope adjustments via keyboard
        from utils.terrain import TerrainCurriculumWrapper
        env = TerrainCurriculumWrapper(env)
        if slope != 0:
            env.current_angle_deg = slope
            env._apply_inclination(slope)
        return env

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
    if direction:
        print(f"Direction: {direction.upper()}")
        print("Press '+'/'-' to adjust floor slope")
    
    # For recurrent models
    lstm_states = None
    episode_start = True

    # Setup for custom visualization if direction specified
    viewer = None
    applied_slope = slope
    if direction:
        import mujoco.viewer
        # Access the base MuJoCo environment through the wrappers
        base_env = env.unwrapped.envs[0].unwrapped
        
        # Get the direction vector for visualization
        from utils.directional_control import DirectionalPolicyWrapper
        direction_wrapper = env.unwrapped.envs[0]
        while not isinstance(direction_wrapper, DirectionalPolicyWrapper):
            if hasattr(direction_wrapper, 'env'):
                direction_wrapper = direction_wrapper.env
            else:
                direction_wrapper = None
                break
        
        goal_vector = direction_wrapper.goal_vector if direction_wrapper else np.array([1.0, 0.0])
        
        # Key callback for slope adjustment
        def key_callback(keycode):
            nonlocal applied_slope
            if keycode == ord('='): # Plus key
                applied_slope += 1.0
                print(f"\rSlope: {applied_slope:.1f}°", end="")
            elif keycode == ord('-'): # Minus key
                applied_slope -= 1.0
                print(f"\rSlope: {applied_slope:.1f}°", end="")

        viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data, key_callback=key_callback)
        viewer.cam.distance = 5.0

    try:
        while True:
            if direction:
                # Apply dynamic slope if changed via keyboard
                inner_env = env.unwrapped.envs[0]
                curr = inner_env
                while hasattr(curr, 'env'):
                    from utils.terrain import TerrainCurriculumWrapper
                    if isinstance(curr, TerrainCurriculumWrapper):
                        if curr.current_angle_deg != applied_slope:
                            curr.current_angle_deg = applied_slope
                            curr._apply_inclination(applied_slope)
                        break
                    curr = curr.env

            # Get action from the model
            if algo == "rec_ppo":
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = False
            else:
                action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, rewards, dones, infos = env.step(action)
            
            if direction and viewer and viewer.is_running():
                # Update visualization
                from utils.directional_control import DirectionalPolicyWrapper
                
                # Find the wrapper
                curr = env.unwrapped.envs[0]
                while hasattr(curr, 'env'):
                    if isinstance(curr, DirectionalPolicyWrapper):
                        break
                    curr = curr.env
                
                base_env = curr.unwrapped
                goal = curr.goal_vector
                
                # Camera Follow
                viewer.cam.lookat[:] = base_env.data.xpos[1]

                # Arrow visualization
                robot_pos = base_env.data.xpos[1]
                
                viewer.user_scn.ngeom = 0
                import mujoco
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_ARROW,
                    size=np.array([0.05, 0.05, 0.4], dtype=np.float64),
                    rgba=np.array([0, 1, 0, 1], dtype=np.float32),  # Green for fixed direction
                    pos=(robot_pos + [0, 0, 0.5]).astype(np.float64), 
                    mat=np.eye(3).flatten().astype(np.float64)
                )
                
                # Calculate rotation to point in goal direction
                angle = np.arctan2(goal[1], goal[0])
                c, s = np.cos(angle), np.sin(angle)
                rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
                viewer.user_scn.geoms[0].mat = rot_mat
                
                viewer.sync()
            
            if dones[0]:
                lstm_states = None
                episode_start = True
                print("\nEpisode end, resetting...")
            
            time.sleep(sleep_time)
            
            if viewer and not viewer.is_running():
                break
                
    except KeyboardInterrupt:
        print("\nEvaluation stopped.")
    finally:
        if viewer:
            viewer.close()
        env.close()


def enjoy_interactive(algo, env_id, model_paths_dict, sleep_time=0.01, slope=0.0):
    """
    Interactive mode: switch between directional policies using keyboard.
    
    Args:
        algo: Algorithm used
        env_id: Environment ID
        model_paths_dict: Dict mapping direction -> model_path
        sleep_time: Time between steps
        slope: Initial floor slope
    """
    from utils.directional_control import wrap_directional_policy, get_valid_directions
    
    valid_dirs = get_valid_directions(env_id)
    current_direction = "forward"
    
    # Load all models
    models = {}
    stats = {}
    for direction, path in model_paths_dict.items():
        if direction not in valid_dirs:
            continue
        if os.path.exists(path):
            if algo == "ppo":
                models[direction] = PPO.load(path)
            elif algo == "sac":
                models[direction] = SAC.load(path)
            elif algo == "rec_ppo":
                models[direction] = RecurrentPPO.load(path)
            
            stats_path = path.replace("_final.zip", "_stats.pkl").replace(".zip", "_stats.pkl")
            if os.path.exists(stats_path):
                stats[direction] = stats_path
            print(f"Loaded {direction} policy from {path}")
        else:
            print(f"Warning: {direction} policy not found at {path}")
    
    if not models:
        print("No models loaded!")
        return
    
    # Create environment
    def make_env():
        env = wrap_directional_policy(env_id, direction=current_direction)
        from utils.terrain import TerrainCurriculumWrapper
        env = TerrainCurriculumWrapper(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load stats for initial direction if available
    if current_direction in stats:
        env = VecNormalize.load(stats[current_direction], env)
        env.training = False
        env.norm_reward = False
    
    import mujoco.viewer
    base_env = env.unwrapped.envs[0].unwrapped
    
    applied_slope = slope
    
    def key_callback(keycode):
        nonlocal current_direction, applied_slope
        
        is_2d = "Ant" not in env_id
        
        if is_2d:
            if keycode in [265, 262, ord('W'), ord('D')]:  # Forward
                current_direction = "forward"
            elif keycode in [264, 263, ord('S'), ord('A')]:  # Backward
                current_direction = "backward"
        else:
            if keycode == 265 or keycode == ord('W'):
                current_direction = "forward"
            elif keycode == 264 or keycode == ord('S'):
                current_direction = "backward"
            elif keycode == 263 or keycode == ord('A'):
                current_direction = "left"
            elif keycode == 262 or keycode == ord('D'):
                current_direction = "right"
        
        if keycode == ord('='):
            applied_slope += 1.0
            print(f"\rSlope: {applied_slope:.1f}°", end="")
        elif keycode == ord('-'):
            applied_slope -= 1.0
            print(f"\rSlope: {applied_slope:.1f}°", end="")
    
    viewer = mujoco.viewer.launch_passive(base_env.model, base_env.data, key_callback=key_callback)
    viewer.cam.distance = 5.0
    
    obs = env.reset()
    print(f"Interactive mode. Use WASD/Arrow keys to switch directions. Press Ctrl+C to stop.")
    print(f"Available directions: {list(models.keys())}")
    
    lstm_states = None
    episode_start = True
    
    try:
        while viewer.is_running():
            # Get current model
            model = models.get(current_direction)
            if model is None:
                model = models[list(models.keys())[0]]
            
            if algo == "rec_ppo":
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = False
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, rewards, dones, infos = env.step(action)
            
            # Update visualization
            viewer.cam.lookat[:] = base_env.data.xpos[1]
            viewer.sync()
            
            if dones[0]:
                obs = env.reset()
                lstm_states = None
                episode_start = True
            
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        viewer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trained locomotion models")
    parser.add_argument("--env", type=str, default="Ant-v5", 
                        help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--direction", type=str, default=None,
                        choices=["forward", "backward", "left", "right"],
                        help="Direction the model was trained for. If not set, uses standard model.")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: switch between all directional policies with keyboard")
    parser.add_argument("--sleep", type=float, default=0.01, 
                        help="Time to sleep between steps")
    parser.add_argument("--slope", type=float, default=0.0, 
                        help="Floor inclination in degrees")
    args = parser.parse_args()
    
    env_name_clean = args.env.replace("-v5", "").lower()
    
    if args.interactive:
        # Load all directional models
        from utils.directional_control import get_valid_directions
        valid_dirs = get_valid_directions(args.env)
        model_paths = {}
        for direction in valid_dirs:
            model_name = f"{args.algo}_{env_name_clean}_{direction}"
            model_paths[direction] = f"./models/{model_name}/{model_name}_final.zip"
        
        enjoy_interactive(args.algo, args.env, model_paths, sleep_time=args.sleep, slope=args.slope)
    else:
        # Single model mode
        if args.direction:
            log_suffix = f"_{args.direction}"
        else:
            log_suffix = ""
        
        model_name = f"{args.algo}_{env_name_clean}{log_suffix}"
        model_path = args.path if args.path else f"./models/{model_name}/{model_name}_final.zip"
        
        enjoy(args.algo, args.env, model_path, direction=args.direction, sleep_time=args.sleep, slope=args.slope)

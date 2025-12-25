import gymnasium as gym
import numpy as np
import json
import argparse
import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

def record_motion(algo, env_id, model_path, output_file="motion_data.json", max_steps=1000, directional=False):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    # Helper function for environment creation
    def make_env():
        render_mode = None
        if directional:
            from utils.directional_control import wrap_directional
            return wrap_directional(env_id, render_mode=render_mode)
        return gym.make(env_id, render_mode=render_mode)

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

    print(f"Starting recording of {env_id}...")
    
    obs = env.reset()
    
    # Data storage
    motion_data = {
        "frames": [],
        "metadata": {
            "env_id": env_id,
            "algo": algo,
            "dt": 0.008, # standard MuJoCo/Gym control step is usually frameskip * timestep. Hopper is usually 0.008 (4 * 0.002)
        }
    }
    
    # For recurrent models
    lstm_states = None
    episode_start = True

    try:
        # Access the underlying MuJoCo environment to get qpos
        # Env hierarchy: DummyVecEnv -> VecNormalize (maybe) -> Monitor (maybe) -> TimeLimit -> OrderEnforcing -> PassiveEnvChecker -> GymEnv
        # We need to dig down to the base env
        base_env = env.envs[0].unwrapped
        
        for step in range(max_steps):
            # Get action
            if algo == "rec_ppo":
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = False
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, rewards, dones, infos = env.step(action)
            
            # Extract Data from MuJoCo data structures
            # qpos structure for Hopper-v5:
            # 0: root_x
            # 1: root_z (height)
            # 2: root_y (angle)
            # 3: thigh_joint
            # 4: leg_joint
            # 5: foot_joint
            # total 6 dof
            
            qpos = base_env.data.qpos.copy()
            
            frame_data = {
                "step": step,
                "root_pos": [float(qpos[0]), float(qpos[1])], # x, z
                "root_angle": float(qpos[2]),                 # y-axis rotation
                "joints": [float(qpos[3]), float(qpos[4]), float(qpos[5])] # thigh, leg, foot
            }
            motion_data["frames"].append(frame_data)
            
            if dones[0]:
                print(f"Episode ended at step {step}")
                break
                
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        env.close()
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(motion_data, f, indent=2)
        print(f"Saved {len(motion_data['frames'])} frames to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v5", help="Gymnasium environment ID")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--steps", type=int, default=500, help="Number of frames to record")
    parser.add_argument("--output", type=str, default="motion_data.json", help="Output JSON file path")
    
    parser.add_argument("--directional", action="store_true", help="Whether the model was trained with directional controls")
    
    args = parser.parse_args()
    
    # Default model path logic (same as enjoy_models.py)
    if not args.path:
        env_name_clean = args.env.replace("-v5", "").lower()
        log_suffix = "_dir" if args.directional else ""
        model_name = f"{args.algo}_{env_name_clean}{log_suffix}"
        args.path = f"./models/{model_name}/{model_name}_final.zip"
    
    record_motion(args.algo, args.env, args.path, args.output, args.steps, args.directional)

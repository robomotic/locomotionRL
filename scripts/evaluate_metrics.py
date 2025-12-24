import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import os
import argparse
from tqdm import tqdm

def evaluate_metrics(algo, model_path, directional=False, n_episodes=20):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    def make_env():
        # Evaluation should be headless for speed
        if directional:
            from utils.directional_control import wrap_directional
            return wrap_directional("Ant-v5")
        return gym.make("Ant-v5")

    env = DummyVecEnv([make_env])
    
    stats_path = model_path.replace("_final.zip", "_stats.pkl").replace(".zip", "_stats.pkl")
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    
    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "sac":
        model = SAC.load(model_path)
    elif algo == "rec_ppo":
        model = RecurrentPPO.load(model_path)
    else:
        print(f"Unknown algorithm: {algo}")
        return

    print(f"Evaluating {model_path} over {n_episodes} episodes...")
    
    all_distances = []
    all_efficiencies = [] # How straight it walks
    all_flips = 0
    episodes_completed = 0

    for ep in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        start_pos = None
        current_pos = None
        
        # Track coordinates
        path = []
        
        # For directional models, target is part of obs or wrapper state
        target_dir = np.array([1.0, 0.0])
        if directional:
            target_dir = env.unwrapped.envs[0].current_goal

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            
            # Ant-v5 info contains x_position, y_position
            info = infos[0]
            pos = np.array([info.get('x_position', 0), info.get('y_position', 0)])
            
            if start_pos is None:
                start_pos = pos
            
            current_pos = pos
            path.append(pos)
            
            if done:
                # Check if it flipped (terminated) or reached time limit (truncated)
                # In SB3 VecEnv, 'terminated' and 'truncated' are collapsed into 'done'
                # but info['TimeLimit.truncated'] tells us if it was a timeout.
                was_truncated = info.get('TimeLimit.truncated', False)
                was_healthy = info.get('reward_survive', 0) > 0 # Ant-v5 healthy check
                
                # If not truncated and done, it likely flipped or went out of bounds
                if not was_truncated:
                    all_flips += 1
                
                # Calculate metrics
                displacement = current_pos - start_pos
                distance = np.linalg.norm(displacement)
                all_distances.append(distance)
                
                if distance > 0.1:
                    # Efficiency: Dot product of displacement and target direction
                    # (How much of the movement was in the intended direction)
                    unit_displacement = displacement / distance
                    efficiency = np.dot(unit_displacement, target_dir)
                    all_efficiencies.append(efficiency)
                
                episodes_completed += 1

    # Aggregate results
    avg_dist = np.mean(all_distances)
    avg_eff = np.mean(all_efficiencies) if all_efficiencies else 0
    flip_rate = (all_flips / n_episodes) * 100

    print("\n" + "="*30)
    print(f" EVALUATION REPORT: {os.path.basename(model_path)}")
    print("="*30)
    print(f"Episodes Run:          {n_episodes}")
    print(f"Avg. Distance Traveled: {avg_dist:.2f} meters")
    print(f"Flip/Failure Rate:     {flip_rate:.1f}% ({all_flips}/{n_episodes} episodes)")
    
    if directional:
        print(f"Straight Line Score:   {avg_eff*100:.1f}% (Directional Efficiency)")
    else:
        # For non-directional, we assume 'Forward' (X+) is the goal
        print(f"Straight Line Score:   {avg_eff*100:.1f}% (X-axis alignment)")
        
    print(f"Survival Probability:  {(1 - all_flips/n_episodes)*100:.1f}%")
    print("="*30)

    if flip_rate > 50:
        print("ADVICE: The robot is unstable. Consider increasing 'healthy_reward' or using Domain Randomization.")
    if avg_eff < 0.7:
        print("ADVICE: The robot's movement is erratic. Consider adding a 'straight-line' penalty in the reward function.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--directional", action="store_true", help="Whether the model was trained with directional controls")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to evaluate")
    args = parser.parse_args()
    
    paths = {
        "ppo": "./models/ppo_ant_dir/ppo_ant_dir_final.zip" if args.directional else "./models/ppo_ant/ppo_ant_final.zip",
        "sac": "./models/sac_ant/sac_ant_final.zip",
        "rec_ppo": "./models/rec_ppo_ant/rec_ppo_ant_final.zip"
    }
    
    model_path = args.path if args.path else paths.get(args.algo)
    evaluate_metrics(args.algo, model_path, directional=args.directional, n_episodes=args.episodes)

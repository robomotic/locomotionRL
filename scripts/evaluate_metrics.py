import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
import os
import argparse
from tqdm import tqdm

def evaluate_metrics(algo, env_id, model_path, direction=None, n_episodes=20):
    """
    Evaluate a trained model and compute locomotion metrics.
    
    Args:
        algo: Algorithm used ("ppo", "sac", "rec_ppo")
        env_id: Gymnasium environment ID
        model_path: Path to the trained model
        direction: If specified, uses directional policy wrapper for this direction.
                   If None, uses standard environment.
        n_episodes: Number of episodes to evaluate
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    def make_env():
        # Evaluation should be headless for speed
        if direction:
            from utils.directional_control import wrap_directional_policy
            return wrap_directional_policy(env_id, direction=direction)
        return gym.make(env_id)

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
    if direction:
        print(f"Direction: {direction}")
    
    all_distances = []
    all_efficiencies = []  # How straight it walks
    all_flips = 0
    episodes_completed = 0

    # Get target direction based on mode
    if direction:
        from utils.directional_control import DirectionalPolicyWrapper
        direction_map = {
            "forward": np.array([1.0, 0.0]),
            "backward": np.array([-1.0, 0.0]),
            "left": np.array([0.0, 1.0]),
            "right": np.array([0.0, -1.0]),
        }
        target_dir = direction_map.get(direction, np.array([1.0, 0.0]))
    else:
        # Default: forward (+X direction)
        target_dir = np.array([1.0, 0.0])

    for ep in tqdm(range(n_episodes)):
        obs = env.reset()
        done = False
        start_pos = None
        current_pos = None
        
        # Track coordinates
        path = []

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
                was_truncated = info.get('TimeLimit.truncated', False)
                
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
    
    if direction:
        print(f"Direction:             {direction.upper()}")
        print(f"Straight Line Score:   {avg_eff*100:.1f}% (Directional Efficiency)")
    else:
        print(f"Straight Line Score:   {avg_eff*100:.1f}% (X-axis alignment)")
        
    print(f"Survival Probability:  {(1 - all_flips/n_episodes)*100:.1f}%")
    print("="*30)

    if flip_rate > 50:
        print("ADVICE: The robot is unstable. Consider increasing 'healthy_reward' or using Domain Randomization.")
    if avg_eff < 0.7:
        print("ADVICE: The robot's movement is erratic. Consider adding a 'straight-line' penalty in the reward function.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained locomotion models")
    parser.add_argument("--env", type=str, default="Ant-v5", 
                        help="Gymnasium environment ID (e.g., Hopper-v5, Walker2d-v5)")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac", "rec_ppo"], default="ppo")
    parser.add_argument("--path", type=str, help="Path to the model zip file")
    parser.add_argument("--direction", type=str, default=None,
                        choices=["forward", "backward", "left", "right"],
                        help="Direction the model was trained for. If not set, uses standard model.")
    parser.add_argument("--episodes", type=int, default=20, 
                        help="Number of episodes to evaluate")
    args = parser.parse_args()
    
    # Set default paths if not provided
    env_name_clean = args.env.replace("-v5", "").lower()
    log_suffix = f"_{args.direction}" if args.direction else ""
    model_name = f"{args.algo}_{env_name_clean}{log_suffix}"
    
    model_path = args.path if args.path else f"./models/{model_name}/{model_name}_final.zip"
    
    evaluate_metrics(args.algo, args.env, model_path, direction=args.direction, n_episodes=args.episodes)

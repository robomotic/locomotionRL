import gymnasium as gym
import numpy as np
import os
import sys

# Add scripts directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.utils.directional_control import wrap_directional_policy

def verify_rewards():
    env_id = "Walker2d-v5"
    config_path = "configs/walker2d_config.json"
    
    # Test Forward
    print("Testing Forward Reward...")
    env_fwd = wrap_directional_policy(env_id, direction="forward", config_path=config_path)
    obs, info = env_fwd.reset()
    action = env_fwd.action_space.sample()
    obs, reward, terminated, truncated, info = env_fwd.step(action)
    
    hr = info.get('reward_survive', 0.0)
    fr = info.get('reward_forward', 0.0)
    cc = abs(info.get('reward_ctrl', 0.0))
    expected_fwd = hr + fr - cc
    
    print(f"  Observed Reward: {reward}")
    print(f"  Expected Reward: {expected_fwd} (HR: {hr}, FR: {fr}, CC: {cc})")
    assert np.isclose(reward, expected_fwd), f"Forward reward mismatch! {reward} != {expected_fwd}"
    print("  Forward Success!")
    env_fwd.close()

    # Test Backward
    print("\nTesting Backward Reward...")
    env_bwd = wrap_directional_policy(env_id, direction="backward", config_path=config_path)
    obs, info = env_bwd.reset()
    action = env_bwd.action_space.sample()
    obs, reward, terminated, truncated, info = env_bwd.step(action)
    
    # In my implementation, I use weight * (-vx) for backward
    # Gymnasium's reward_forward is weight * vx
    hr = info.get('reward_survive', 0.0)
    fr = info.get('reward_forward', 0.0)
    cc = abs(info.get('reward_ctrl', 0.0))
    br = -fr # since br = weight * (-vx) and fr = weight * vx
    expected_bwd = hr + br - cc
    
    print(f"  Observed Reward: {reward}")
    print(f"  Expected Reward: {expected_bwd} (HR: {hr}, BR: {br}, CC: {cc})")
    assert np.isclose(reward, expected_bwd), f"Backward reward mismatch! {reward} != {expected_bwd}"
    print("  Backward Success!")
    env_bwd.close()

def verify_config():
    print("\nTesting Config Loading...")
    config_path = "configs/walker2d_config.json"
    # Change a value in config to verify it's loaded
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    original_hr = config['healthy_reward']
    test_hr = 5.0
    config['healthy_reward'] = test_hr
    
    temp_config = "configs/temp_walker2d_config.json"
    with open(temp_config, 'w') as f:
        json.dump(config, f)
        
    try:
        env = wrap_directional_policy("Walker2d-v5", direction="forward", config_path=temp_config)
        # For Walker2d-v5, healthy_reward directly influences reward_survive
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        hr = info.get('reward_survive', 0.0)
        print(f"  Requested healthy_reward: {test_hr}")
        print(f"  Observed reward_survive: {hr}")
        assert np.isclose(hr, test_hr), f"Config loading failed! {hr} != {test_hr}"
        print("  Config Loading Success!")
        env.close()
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)

if __name__ == "__main__":
    try:
        verify_rewards()
        verify_config()
        print("\nALL VERIFICATIONS PASSED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        sys.exit(1)

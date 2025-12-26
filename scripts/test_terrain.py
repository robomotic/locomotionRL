import gymnasium as gym
import numpy as np
from utils.terrain import TerrainCurriculumWrapper

def test_terrain():
    env = gym.make("Hopper-v5")
    total_steps = 1000
    env = TerrainCurriculumWrapper(env, total_timesteps=total_steps)
    
    print("Testing Terrain Curriculum Wrapper...")
    
    stages = [0, 250, 600, 850] # Representative steps for stages 1, 2, 3, 4
    for step in stages:
        env.update_curriculum(step)
        gravity = env.unwrapped.model.opt.gravity.copy()
        angle = env.current_angle_deg
        print(f"Step {step} ({step/total_steps*100}%): Angle = {angle:.2f} deg, Gravity = {gravity}")
        
        # Verify gravity rotation
        expected_gx = -9.81 * np.sin(np.radians(angle))
        expected_gz = -9.81 * np.cos(np.radians(angle))
        
        assert np.isclose(gravity[0], expected_gx, atol=1e-3)
        assert np.isclose(gravity[2], expected_gz, atol=1e-3)
        
    print("Test passed!")

if __name__ == "__main__":
    test_terrain()

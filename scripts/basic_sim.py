import gymnasium as gym
import time
import numpy as np

def run_basic_simulation():
    # Create the MuJoCo Ant environment
    # render_mode="human" allows us to see the simulation
    env = gym.make("Ant-v5", render_mode="human")
    
    observation, info = env.reset()
    
    print("Environment initialized.")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    for _ in range(1000):
        # Sample a random action
        action = env.action_space.sample()
        
        # Step the simulation
        observation, reward, terminated, truncated, info = env.step(action)
        
        # If the episode is over, reset
        if terminated or truncated:
            observation, info = env.reset()
            print("Episode finished, resetting environment...")

        # Sleep slightly to slow down rendering
        time.sleep(0.01)

    env.close()

if __name__ == "__main__":
    try:
        run_basic_simulation()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Note: If 'human' render mode fails, ensure you have a display or use 'rgb_array'.")

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, spaces

class DirectionalControlWrapper(Wrapper):
    """
    A wrapper that enables goal-conditioned locomotion.
    Adds a target direction (command) to the observation space and 
    modifies the reward to encourage following that command.
    """
    def __init__(self, env, change_goal_freq=200):
        super().__init__(env)
        
        # Original observation space is (27,)
        # We add 2 values for the target direction vector (dx, dy)
        low = np.concatenate([self.observation_space.low, [-1.0, -1.0]])
        high = np.concatenate([self.observation_space.high, [1.0, 1.0]])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        
        self.change_goal_freq = change_goal_freq
        self.steps_since_goal_change = 0
        self.current_goal = np.array([1.0, 0.0]) # Start with "Forward" (W)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps_since_goal_change = 0
        # Randomize goal on reset
        self._sample_new_goal()
        return self._get_obs(obs), info

    def _sample_new_goal(self):
        # Sample a random direction on the unit circle
        angle = np.random.uniform(0, 2 * np.pi)
        self.current_goal = np.array([np.cos(angle), np.sin(angle)])

    def _get_obs(self, obs):
        return np.concatenate([obs, self.current_goal])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate velocity-based reward
        # In Ant-v5, info contains 'x_velocity' and 'y_velocity'
        actual_vel = np.array([info.get('x_velocity', 0.0), info.get('y_velocity', 0.0)])
        
        # Reward = projection of actual velocity onto target direction
        # We want to maximize movement in the target direction
        directional_reward = np.dot(actual_vel, self.current_goal)
        
        # Penalty for moving in the wrong direction (perpendicular movement)
        # We can use the standard reward structure but replace the forward component
        # Standard Ant forward reward is usually just x_velocity. 
        # We replace it with our directional reward.
        
        # Note: Ant-v5's 'reward' already contains forward_reward + healthy_reward - costs
        # We'll subtract the original forward_reward and add our directional one.
        original_forward_reward = info.get('forward_reward', actual_vel[0])
        modified_reward = (reward - original_forward_reward) + directional_reward
        
        self.steps_since_goal_change += 1
        if self.steps_since_goal_change >= self.change_goal_freq:
            self._sample_new_goal()
            self.steps_since_goal_change = 0
            
        return self._get_obs(obs), modified_reward, terminated, truncated, info

def wrap_directional(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = DirectionalControlWrapper(env)
    return env

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
        info['current_goal'] = self.current_goal
        # Initial angular velocity is zero
        info['ang_velocity'] = np.zeros(3)
        return self._get_obs(obs), info

    def _sample_new_goal(self):
        # Sample a random direction on the unit circle
        angle = np.random.uniform(0, 2 * np.pi)
        self.current_goal = np.array([np.cos(angle), np.sin(angle)])

    def _get_obs(self, obs):
        return np.concatenate([obs, self.current_goal])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['current_goal'] = self.current_goal
        
        # 0. Extract Torso Angular Velocity from MuJoCo data
        # In Ant-v5, the first 6 elements of qvel are linear and angular velocities of the torso
        ang_vel = self.env.unwrapped.data.qvel[3:6]
        info['ang_velocity'] = ang_vel

        # 1. Coordinate tracking & Velocity
        actual_vel = np.array([info.get('x_velocity', 0.0), info.get('y_velocity', 0.0)])
        current_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)])
        
        # 2. Directional Reward (Progress toward goal)
        directional_reward = np.dot(actual_vel, self.current_goal)
        
        # 3. Cross-Track Error (CTE) Penalty
        # Distance from the line starting at (0,0) with direction self.current_goal
        # Formula for point (x0, y0) distance from line ax + by + c = 0
        # Line in direction (dx, dy) passing through origin: dy*x - dx*y = 0
        dx, dy = self.current_goal
        cte = np.abs(dy * current_pos[0] - dx * current_pos[1])
        cte_penalty = -0.5 * cte
        
        # 4. Heading Alignment Reward
        # Get the robot's current facing direction (from quaternions in qpos[3:7])
        # For simplicity in Ant-v5, we can approximate heading from the torso orientation info if available,
        # otherwise we use the velocity direction as a proxy for 'intent'.
        # However, a better way is to encourage velocity alignment with goal:
        vel_mag = np.linalg.norm(actual_vel)
        alignment_reward = 0.0
        if vel_mag > 0.1:
            alignment = np.dot(actual_vel / vel_mag, self.current_goal)
            alignment_reward = 0.5 * alignment
            
        # 5. Stability Penalty (Angular Velocity)
        # Ant-v5 info usually provides 'ang_velocity' or similar. 
        # We penalize 'yapping/spinning' to keep the gait stable.
        ang_vel = info.get('ang_velocity', np.zeros(3))
        stability_penalty = -0.1 * np.linalg.norm(ang_vel)
        
        # 6. Survival & Termination
        is_healthy = info.get('reward_survive', 0) > 0
        extra_healthy_reward = 1.0 if is_healthy else 0.0
        
        flip_penalty = 0.0
        if terminated and not truncated:
            flip_penalty = -100.0
            print("\r💥 FLIPPED! Applying penalty.      ", end="")
        
        # Assemble final reward
        # We strip the original forward_reward to avoid conflicting goals
        original_forward_reward = info.get('forward_reward', actual_vel[0])
        modified_reward = (reward - original_forward_reward) + \
                          directional_reward + \
                          cte_penalty + \
                          alignment_reward + \
                          stability_penalty + \
                          extra_healthy_reward + \
                          flip_penalty
        
        self.steps_since_goal_change += 1
        if self.steps_since_goal_change >= self.change_goal_freq:
            self._sample_new_goal()
            self.steps_since_goal_change = 0
            
        return self._get_obs(obs), modified_reward, terminated, truncated, info

def wrap_directional(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = DirectionalControlWrapper(env)
    return env

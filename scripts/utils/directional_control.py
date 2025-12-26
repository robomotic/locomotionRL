import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, spaces

class DirectionalControlWrapper(Wrapper):
    """
    A wrapper that enables goal-conditioned locomotion.
    Supports both 3D (Ant) and 2D (Hopper, Walker2d) environments.
    """
    def __init__(self, env, change_goal_freq=200):
        super().__init__(env)
        
        # Detect robot type: Ant is 3D (X,Y,Z), Hopper/Walker are 2D (X,Z)
        self.is_2d = "Ant" not in env.unwrapped.spec.id
        
        # Original observation space
        # We add 2 values for the target direction vector (dx, dy)
        low = np.concatenate([self.observation_space.low, [-1.0, -1.0]])
        high = np.concatenate([self.observation_space.high, [1.0, 1.0]])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        
        self.change_goal_freq = change_goal_freq
        self.steps_since_goal_change = 0
        self.current_goal = np.array([1.0, 0.0]) # Start with "Forward"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps_since_goal_change = 0
        self._sample_new_goal()
        info['current_goal'] = self.current_goal
        info['ang_velocity'] = np.zeros(3)
        return self._get_obs(obs), info

    def _sample_new_goal(self):
        # 10% chance of "Stationary" goal
        if np.random.rand() < 0.1:
            self.current_goal = np.array([0.0, 0.0])
            return

        if self.is_2d:
            # For 2D robots, goal is only Forward (+1) or Backward (-1) on X
            dir = 1.0 if np.random.rand() > 0.5 else -1.0
            self.current_goal = np.array([dir, 0.0])
        else:
            # Sample a random direction on the unit circle
            angle = np.random.uniform(0, 2 * np.pi)
            self.current_goal = np.array([np.cos(angle), np.sin(angle)])

    def _get_obs(self, obs):
        return np.concatenate([obs, self.current_goal])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['current_goal'] = self.current_goal
        
        # 0. Extract Torso Velocities and positions
        qvel = self.env.unwrapped.data.qvel
        if self.is_2d:
            # Hopper/Walker: [vx, vz, pitch_vel, ...]
            # We treat the robot's locomotion as 1D along X
            vx = qvel[0]
            actual_vel = np.array([vx, 0.0]) 
            ang_vel = np.array([0.0, qvel[2], 0.0]) # Pitch velocity
            current_pos = np.array([info.get('x_position', 0.0), 0.0])
        else:
            # Ant: [vx, vy, vz, wx, wy, wz, ...]
            actual_vel = np.array([qvel[0], qvel[1]])
            ang_vel = qvel[3:6]
            current_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)])
            
        info['ang_velocity'] = ang_vel

        # 2. Directional Reward (Progress toward goal)
        is_stationary = np.all(self.current_goal == 0)
        if is_stationary:
            # Penalty for moving when we should be stationary
            drift_penalty = -1.0 * np.linalg.norm(actual_vel)
            directional_reward = drift_penalty
        else:
            directional_reward = np.dot(actual_vel, self.current_goal)
        
        # 3. Cross-Track Error (CTE) Penalty
        if self.is_2d:
            cte_penalty = 0.0 
        else:
            dx, dy = self.current_goal
            cte = np.abs(dy * current_pos[0] - dx * current_pos[1])
            cte_penalty = -0.5 * cte
        
        # 4. Heading Alignment Reward
        vel_mag = np.linalg.norm(actual_vel)
        alignment_reward = 0.0
        if not is_stationary and vel_mag > 0.1:
            alignment = np.dot(actual_vel / vel_mag, self.current_goal)
            alignment_reward = 0.5 * alignment
            
        # 5. Stability Penalty (Angular Velocity)
        stability_penalty = -0.1 * np.linalg.norm(ang_vel)
        
        # 6. Survival & Termination
        is_healthy = info.get('reward_survive', 0) > 0
        extra_healthy_reward = 1.0 if is_healthy else 0.0
        
        flip_penalty = 0.0
        if terminated and not truncated:
            flip_penalty = -100.0
        
        # Assemble final reward
        # Map reward keys correctly: Gymnasium v5 uses 'reward_forward'
        forward_reward_component = info.get('reward_forward', info.get('forward_reward', 0.0))
        
        # We replace the environment's default forward reward with our goal-conditioned one
        modified_reward = (reward - forward_reward_component) + \
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

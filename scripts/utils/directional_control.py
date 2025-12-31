import gymnasium as gym
import numpy as np
import json
import os
from gymnasium import Wrapper, spaces
from typing import Literal, Optional

# Valid directions for each robot type
DIRECTIONS_3D = ["forward", "backward", "left", "right"]
DIRECTIONS_2D = ["forward", "backward"]

class DirectionalPolicyWrapper(Wrapper):
    """
    A wrapper for training direction-specific policies.
    Each policy is trained for a single fixed direction, eliminating
    the need for goal vectors in the observation space.
    
    For 3D robots (Ant): forward, backward, left, right
    For 2D robots (Hopper, Walker2d): forward, backward only
    """
    def __init__(self, env, direction: Literal["forward", "backward", "left", "right"] = "forward"):
        super().__init__(env)
        
        # Detect robot type: Ant is 3D (X,Y,Z), Hopper/Walker are 2D (X,Z)
        self.env_id = env.unwrapped.spec.id
        self.is_2d = "Ant" not in self.env_id
        self.is_walker2d = "Walker2d" in self.env_id
        
        # Validate direction
        valid_directions = DIRECTIONS_2D if self.is_2d else DIRECTIONS_3D
        if direction not in valid_directions:
            raise ValueError(f"Invalid direction '{direction}' for {'2D' if self.is_2d else '3D'} robot. "
                           f"Valid directions: {valid_directions}")
        
        self.direction = direction
        self.goal_vector = self._direction_to_vector(direction)
        
        # Observation space remains unchanged (no goal vector appended)
        # This is the key difference from the old wrapper
        
    def _direction_to_vector(self, direction: str) -> np.ndarray:
        """Convert direction name to unit vector."""
        direction_map = {
            "forward": np.array([1.0, 0.0]),
            "backward": np.array([-1.0, 0.0]),
            "left": np.array([0.0, 1.0]),
            "right": np.array([0.0, -1.0]),
        }
        return direction_map[direction]
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['direction'] = self.direction
        info['goal_vector'] = self.goal_vector
        info['ang_velocity'] = np.zeros(3)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['direction'] = self.direction
        info['goal_vector'] = self.goal_vector
        
        # Extract velocities from MuJoCo
        qvel = self.env.unwrapped.data.qvel
        if self.is_2d:
            # Hopper/Walker: [vx, vz, pitch_vel, ...]
            vx = qvel[0]
            actual_vel = np.array([vx, 0.0])
            ang_vel = np.array([0.0, qvel[2], 0.0])  # Pitch velocity
            current_pos = np.array([info.get('x_position', 0.0), 0.0])
        else:
            # Ant: [vx, vy, vz, wx, wy, wz, ...]
            actual_vel = np.array([qvel[0], qvel[1]])
            ang_vel = qvel[3:6]
            current_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)])
            
        info['ang_velocity'] = ang_vel
        info['velocity'] = actual_vel

        # Compute directional reward
        directional_reward = np.dot(actual_vel, self.goal_vector)
        
        # Cross-Track Error (CTE) Penalty - only for 3D robots
        cte_penalty = 0.0
        if not self.is_2d:
            dx, dy = self.goal_vector
            cte = np.abs(dy * current_pos[0] - dx * current_pos[1])
            cte_penalty = -0.5 * cte
        
        # Heading Alignment Reward
        vel_mag = np.linalg.norm(actual_vel)
        alignment_reward = 0.0
        if vel_mag > 0.1:
            alignment = np.dot(actual_vel / vel_mag, self.goal_vector)
            alignment_reward = 0.5 * alignment
            
        # Stability Penalty (Angular Velocity)
        stability_penalty = -0.1 * np.linalg.norm(ang_vel)
        
        # Survival & Termination
        is_healthy = info.get('reward_survive', 0) > 0
        extra_healthy_reward = 1.0 if is_healthy else 0.0
        
        flip_penalty = 0.0
        if terminated and not truncated:
            flip_penalty = -100.0
        
        # Gymnasium v5 uses 'reward_forward'
        forward_reward_component = info.get('reward_forward', info.get('forward_reward', 0.0))
        ctrl_cost = info.get('reward_ctrl', info.get('ctrl_cost', 0.0))
        # ensure ctrl_cost is positive for the subtraction in the formula if it came as a penalty (negative)
        # though gymnasium usually returns it as positive cost to be subtracted.
        ctrl_cost = abs(ctrl_cost)
        
        healthy_reward = info.get('reward_survive', 0.0)
        
        if self.is_walker2d:
            # Specific logic for Walker2d as requested:
            # forward: reward = healthy_reward bonus + forward_reward - ctrl_cost
            # backward: reward = healthy_reward bonus + backward_reward - ctrl_cost
            
            # backward_reward is based on forward_reward (w * dx/dt)
            # In Gymnasium, reward_forward = weight * (x_after - x_before) / dt
            # For backward, we want the same weight but negative displacement.
            
            if self.direction == "forward":
                directional_reward_component = forward_reward_component
            else:
                # backward_reward is based on forward_reward but for motion in the opposite direction
                # Since forward_reward = weight * vx, backward_reward should be weight * (-vx)
                directional_reward_component = -forward_reward_component
                
            modified_reward = healthy_reward + directional_reward_component - ctrl_cost
        else:
            # Generic directional reward for other robots (Ant, Hopper)
            directional_reward = np.dot(actual_vel, self.goal_vector)
            
            # Cross-Track Error (CTE) Penalty - only for 3D robots
            cte_penalty = 0.0
            if not self.is_2d:
                dx, dy = self.goal_vector
                cte = np.abs(dy * current_pos[0] - dx * current_pos[1])
                cte_penalty = -0.5 * cte
            
            # Heading Alignment Reward
            vel_mag = np.linalg.norm(actual_vel)
            alignment_reward = 0.0
            if vel_mag > 0.1:
                alignment = np.dot(actual_vel / vel_mag, self.goal_vector)
                alignment_reward = 0.5 * alignment
                
            # Stability Penalty (Angular Velocity)
            stability_penalty = -0.1 * np.linalg.norm(ang_vel)
            
            # Survival & Termination
            is_healthy = healthy_reward > 0
            extra_healthy_reward = 1.0 if is_healthy else 0.0
            
            flip_penalty = 0.0
            if terminated and not truncated:
                flip_penalty = -100.0
                
            modified_reward = (reward - forward_reward_component) + \
                              directional_reward + \
                              cte_penalty + \
                              alignment_reward + \
                              stability_penalty + \
                              extra_healthy_reward + \
                              flip_penalty
        
        return obs, modified_reward, terminated, truncated, info


def wrap_directional_policy(env_id: str, direction: str = "forward", config_path: Optional[str] = None, **kwargs):
    """
    Create an environment wrapped for a specific direction policy.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "Walker2d-v5", "Ant-v5")
        direction: One of "forward", "backward", "left", "right"
                   (left/right only valid for 3D robots like Ant)
        config_path: Path to a JSON configuration file for environment arguments
        **kwargs: Additional arguments passed to gym.make()
    
    Returns:
        Wrapped environment for the specified direction
    """
    # Load config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_kwargs = json.load(f)
            # In case of overlap, kwargs (command line/explicit) override file_kwargs
            merged_kwargs = {**file_kwargs, **kwargs}
    else:
        merged_kwargs = kwargs

    env = gym.make(env_id, **merged_kwargs)
    env = DirectionalPolicyWrapper(env, direction=direction)
    return env


def get_valid_directions(env_id: str) -> list:
    """
    Get valid directions for a given environment.
    
    Args:
        env_id: Gymnasium environment ID
        
    Returns:
        List of valid direction strings
    """
    is_2d = "Ant" not in env_id
    return DIRECTIONS_2D if is_2d else DIRECTIONS_3D


# Legacy wrapper for backward compatibility (deprecated)
class DirectionalControlWrapper(Wrapper):
    """
    DEPRECATED: Use DirectionalPolicyWrapper instead.
    
    This wrapper adds a goal vector to observations for goal-conditioned learning.
    The new approach trains separate policies per direction instead.
    """
    def __init__(self, env, change_goal_freq=200):
        import warnings
        warnings.warn(
            "DirectionalControlWrapper is deprecated. Use DirectionalPolicyWrapper "
            "and train separate policies for each direction instead.",
            DeprecationWarning
        )
        super().__init__(env)
        
        # Detect robot type
        self.is_2d = "Ant" not in env.unwrapped.spec.id
        
        # Add 2 values for the target direction vector (dx, dy)
        low = np.concatenate([self.observation_space.low, [-1.0, -1.0]])
        high = np.concatenate([self.observation_space.high, [1.0, 1.0]])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        
        self.change_goal_freq = change_goal_freq
        self.steps_since_goal_change = 0
        self.current_goal = np.array([1.0, 0.0])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.steps_since_goal_change = 0
        self._sample_new_goal()
        info['current_goal'] = self.current_goal
        info['ang_velocity'] = np.zeros(3)
        return self._get_obs(obs), info

    def _sample_new_goal(self):
        if np.random.rand() < 0.1:
            self.current_goal = np.array([0.0, 0.0])
            return

        if self.is_2d:
            dir = 1.0 if np.random.rand() > 0.5 else -1.0
            self.current_goal = np.array([dir, 0.0])
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            self.current_goal = np.array([np.cos(angle), np.sin(angle)])

    def _get_obs(self, obs):
        return np.concatenate([obs, self.current_goal])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['current_goal'] = self.current_goal
        
        qvel = self.env.unwrapped.data.qvel
        if self.is_2d:
            vx = qvel[0]
            actual_vel = np.array([vx, 0.0]) 
            ang_vel = np.array([0.0, qvel[2], 0.0])
            current_pos = np.array([info.get('x_position', 0.0), 0.0])
        else:
            actual_vel = np.array([qvel[0], qvel[1]])
            ang_vel = qvel[3:6]
            current_pos = np.array([info.get('x_position', 0.0), info.get('y_position', 0.0)])
            
        info['ang_velocity'] = ang_vel

        is_stationary = np.all(self.current_goal == 0)
        if is_stationary:
            drift_penalty = -1.0 * np.linalg.norm(actual_vel)
            directional_reward = drift_penalty
        else:
            directional_reward = np.dot(actual_vel, self.current_goal)
        
        if self.is_2d:
            cte_penalty = 0.0 
        else:
            dx, dy = self.current_goal
            cte = np.abs(dy * current_pos[0] - dx * current_pos[1])
            cte_penalty = -0.5 * cte
        
        vel_mag = np.linalg.norm(actual_vel)
        alignment_reward = 0.0
        if not is_stationary and vel_mag > 0.1:
            alignment = np.dot(actual_vel / vel_mag, self.current_goal)
            alignment_reward = 0.5 * alignment
            
        stability_penalty = -0.1 * np.linalg.norm(ang_vel)
        
        is_healthy = info.get('reward_survive', 0) > 0
        extra_healthy_reward = 1.0 if is_healthy else 0.0
        
        flip_penalty = 0.0
        if terminated and not truncated:
            flip_penalty = -100.0
        
        forward_reward_component = info.get('reward_forward', info.get('forward_reward', 0.0))
        
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
    """
    DEPRECATED: Use wrap_directional_policy() instead.
    """
    import warnings
    warnings.warn(
        "wrap_directional() is deprecated. Use wrap_directional_policy() instead.",
        DeprecationWarning
    )
    env = gym.make(env_id, **kwargs)
    env = DirectionalControlWrapper(env)
    return env

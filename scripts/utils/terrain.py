import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

class TerrainCurriculumWrapper(Wrapper):
    """
    A wrapper that simulates floor inclination by tilting the gravity vector.
    Supports a curriculum that changes the angle based on training progress.
    """
    def __init__(self, env, total_timesteps=1000000):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.current_angle_deg = 0.0
        
        # Original gravity magnitude (usually 9.81)
        self.g_mag = np.linalg.norm(self.unwrapped.model.opt.gravity)
        if self.g_mag == 0:
            self.g_mag = 9.81

    def set_total_timesteps(self, total_timesteps):
        self.total_timesteps = total_timesteps

    def update_curriculum(self, global_step):
        """
        Updates the floor inclination based on the global training step.
        Stages:
        1. 0% - 20%: Flat (0 deg)
        2. 20% - 50%: Increasing Uphill (0 to +15 deg)
        3. 50% - 70%: Flat (0 deg)
        4. 70% - 100%: Increasing Downhill (0 to -15 deg)
        """
        self.current_step = global_step
        progress = global_step / self.total_timesteps
        
        if progress < 0.2:
            # Stage 1: Flat
            angle = 0.0
        elif progress < 0.5:
            # Stage 2: Uphill climb (0 to 15)
            stage_progress = (progress - 0.2) / 0.3
            angle = stage_progress * 15.0
        elif progress < 0.7:
            # Stage 3: Flat
            angle = 0.0
        else:
            # Stage 4: Downhill (0 to -15)
            stage_progress = (progress - 0.7) / 0.3
            angle = stage_progress * -15.0
            
        self.current_angle_deg = angle
        self._apply_inclination(angle)

    def _apply_inclination(self, angle_deg):
        """
        Rotates the gravity vector to simulate a slope.
        Angle > 0 is uphill (robot walking in +X).
        """
        angle_rad = np.radians(angle_deg)
        
        # For a slope of alpha, the gravity vector in the world frame 
        # (where the floor is always Z=0) rotates by -alpha around Y.
        # g_x = -g * sin(alpha)
        # g_z = -g * cos(alpha)
        gx = -self.g_mag * np.sin(angle_rad)
        gz = -self.g_mag * np.cos(angle_rad)
        
        self.unwrapped.model.opt.gravity[0] = gx
        self.unwrapped.model.opt.gravity[2] = gz

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['terrain_angle_deg'] = self.current_angle_deg
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['terrain_angle_deg'] = self.current_angle_deg
        return obs, info

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

class DomainRandomizationWrapper(Wrapper):
    """
    A wrapper that randomizes physical parameters of a MuJoCo environment
    at each reset to improve policy robustness.
    """
    def __init__(self, env, 
                 friction_range=(0.5, 1.5), 
                 mass_range=(0.8, 1.2), 
                 actuator_range=(0.8, 1.2)):
        super().__init__(env)
        self.friction_range = friction_range
        self.mass_range = mass_range
        self.actuator_range = actuator_range
        
        # Store original values
        self.original_friction = self.unwrapped.model.geom_friction.copy()
        self.original_mass = self.unwrapped.model.body_mass.copy()
        self.original_gain = self.unwrapped.model.actuator_gainprm.copy()

    def reset(self, **kwargs):
        # Randomize friction
        friction_factor = np.random.uniform(*self.friction_range)
        self.unwrapped.model.geom_friction[:] = self.original_friction * friction_factor
        
        # Randomize mass
        mass_factor = np.random.uniform(*self.mass_range)
        self.unwrapped.model.body_mass[:] = self.original_mass * mass_factor
        
        # Randomize actuator gain
        gain_factor = np.random.uniform(*self.actuator_range)
        self.unwrapped.model.actuator_gainprm[:] = self.original_gain * gain_factor
        
        return self.env.reset(**kwargs)

def wrap_env(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = DomainRandomizationWrapper(env)
    return env

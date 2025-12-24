import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LocomotionMetricsCallback(BaseCallback):
    """
    A custom callback for tracking locomotion-specific metrics during training.
    Logs to TensorBoard and a local CSV for graphical display.
    """
    def __init__(self, verbose=0, log_dir="./logs/metrics/"):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, "locomotion_progress.csv")
        os.makedirs(log_dir, exist_ok=True)
        
        # Performance trackers
        self.total_flips = 0
        self.total_straight_meters = 0.0
        self.last_action = None
        self.jitter_window = []
        
        # Initialize CSV header
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("timesteps,flips,straight_meters,jitter,ang_vel\n")
        
        self.ang_vel_window = []

    def _on_step(self) -> bool:
        # 1. Track Jitter (Action delta)
        current_action = self.locals["actions"]
        if self.last_action is not None:
            jitter = np.mean(np.abs(current_action - self.last_action))
            self.jitter_window.append(jitter)
            if len(self.jitter_window) > 100:
                self.jitter_window.pop(0)
        self.last_action = current_action.copy()

        # 2. Track Flips, Progress, and Stability from Info
        for i, info in enumerate(self.locals["infos"]):
            # Straight meters
            vel = np.array([info.get('x_velocity', 0.0), info.get('y_velocity', 0.0)])
            goal = info.get('current_goal', np.array([1.0, 0.0]))
            progress = np.dot(vel, goal) * 0.02
            if progress > 0:
                self.total_straight_meters += progress

            # Tracking Angular Velocity (Stability)
            ang_vel = info.get('ang_velocity', np.zeros(3))
            self.ang_vel_window.append(np.linalg.norm(ang_vel))
            if len(self.ang_vel_window) > 1000:
                self.ang_vel_window.pop(0)

            # Termination (Flip)
            if self.locals["dones"][i]:
                if not info.get("TimeLimit.truncated", False):
                    self.total_flips += 1

        # 3. Periodically save to CSV for the dashboard
        if self.n_calls % 1000 == 0:
            avg_jitter = np.mean(self.jitter_window) if self.jitter_window else 0.0
            avg_ang_vel = np.mean(self.ang_vel_window) if self.ang_vel_window else 0.0
            with open(self.csv_path, "a") as f:
                f.write(f"{self.num_timesteps},{self.total_flips},{self.total_straight_meters:.4f},{avg_jitter:.6f},{avg_ang_vel:.4f}\n")
            
            # Also log to TensorBoard
            self.logger.record("locomotion/total_flips", self.total_flips)
            self.logger.record("locomotion/straight_meters", self.total_straight_meters)
            self.logger.record("locomotion/jitter", avg_jitter)
            self.logger.record("locomotion/avg_ang_vel", avg_ang_vel)
            
        return True

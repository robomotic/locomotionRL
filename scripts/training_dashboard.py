import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import argparse

def animate(i, csv_path):
    if not os.path.exists(csv_path):
        return

    try:
        data = pd.read_csv(csv_path)
    except:
        return # Handle file locks or empty files during writing

    if len(data) < 2:
        return

    plt.cla()
    
    # Create subplots
    fig.clear()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    # Plot Flips
    ax1.plot(data['timesteps'], data['flips'], color='red', label='Total Flips')
    ax1.set_title('Locomotion Performance Dashboard')
    ax1.set_ylabel('Flips')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left')

    # Plot Straight Meters
    ax2.plot(data['timesteps'], data['straight_meters'], color='green', label='Straight Meters')
    ax2.set_ylabel('Dist (m)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left')

    # Plot Jitter
    ax3.plot(data['timesteps'], data['jitter'], color='blue', label='Action Jitter')
    ax3.set_ylabel('Jitter')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='upper left')

    # Plot Angular Velocity
    ax4.plot(data['timesteps'], data['ang_vel'], color='purple', label='Avg. Ang. Vel (Stability)')
    ax4.set_xlabel('Timesteps')
    ax4.set_ylabel('Rad/s')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend(loc='upper left')

    plt.tight_layout()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Ant-v5", help="Environment ID (e.g., Hopper-v5)")
    parser.add_argument("--directional", action="store_true", help="Monitor directional logs")
    args = parser.parse_args()

    env_name_clean = args.env.replace("-v5", "").lower()
    log_suffix = "_dir" if args.directional else ""
    csv_path = f"./logs/ppo_{env_name_clean}{log_suffix}/locomotion_progress.csv"

    print(f"Starting dashboard monitoring: {csv_path}")
    print("Close the window to stop.")

    fig = plt.figure(figsize=(10, 8))
    ani = FuncAnimation(fig, animate, fargs=(csv_path,), interval=2000, cache_frame_data=False)
    plt.show()

#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting setup for LocomotionRL..."

# Define the virtual environment directory
VENV_DIR="venv"

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "📂 Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "🛠️ Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
echo "🔌 Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
# Core dependencies
pip install mujoco gymnasium[mujoco] stable-baselines3 sb3-contrib shimmy

# Optional/Helper dependencies
pip install moviepy tensorboard tqdm rich

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "Now you can run the simulation:"
echo "    python scripts/basic_sim.py"

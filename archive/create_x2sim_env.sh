#!/bin/bash

# This script will build a $SCRATCH/x2sim directory if not present and build a virtual environment named x2sim_env in the $SCRATCH/x2sim directory.
# Ensure you have the x2sim repository cloned in your $SCRATCH directory:
# Ensures PyTorch is built with CUDA support. Activates environment upon being built. Otherwise, activate the virtual environment with:
 
# cd $SCRATCH/x2sim
# source x2sim_env/bin/activate

# Define paths
REPO_DIR="$SCRATCH/x2sim"
ENV_DIR="$REPO_DIR/x2sim_env"

# Create repository directory if it doesn't exist
if [ ! -d "$REPO_DIR" ]; then
    echo "Creating repository directory at $REPO_DIR..."
    mkdir -p "$REPO_DIR"
fi

# Create virtual environment
echo "Creating virtual environment at $ENV_DIR..."
python3 -m venv "$ENV_DIR"

# Activate the environment
source "$ENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "Installing PyTorch with CUDA support..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu12

# Remove torch and torchvision from requirements before installing
echo "Installing remaining packages from requirements.txt..."
grep -v "torch\|torchvision" "$REPO_DIR/requirements.txt" | pip install -r /dev/stdin

echo "Environment setup complete!"
echo "To activate the environment, run: source $REPO_DIR/start_x2sim_env.sh"

# Create activation script
cat > "$REPO_DIR/start_x2sim_env.sh" << EOL
#!/bin/bash
source "$ENV_DIR/bin/activate"
which python
echo "x2sim_env activated. Run 'deactivate' to exit the environment."
EOL

chmod +x "$REPO_DIR/start_x2sim_env.sh"

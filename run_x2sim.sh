#!/bin/bash
#SBATCH -J X2Sim_Run                     # Job name
#SBATCH -o X2Sim_Run.o%j                 # Name of stdout output file
#SBATCH -e X2Sim_Run.e%j                 # Name of stderr error file
#SBATCH -p gpu-a100-dev                  # Queue (partition) name - using GPU for TRELLIS
#SBATCH -N 1                             # Total # of nodes
#SBATCH -n 1                             # Total # of mpi tasks
#SBATCH -t 01:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-type=all                  # Send email at begin and end of job
#SBATCH --mail-user=jgaucin@example.com  # Replace with your email
#SBATCH -A XXXXXXXX                      # Project/Allocation name (replace with yours)

# Exit on error
set -e

# Navigate to target directory
cd $SCRATCH/x2sim

# Activate virtual environment
source start_x2sim_env.sh

# Check if parameters are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch run_x2sim.sh --prompt \"Your prompt here\" [options]"
    echo "   or: sbatch run_x2sim.sh --video \"video_url_or_path\" [options]"
    echo ""
    echo "Options:"
    echo "  --prompt TEXT     Text prompt describing the object and simulation"
    echo "  --video URL/PATH  URL or path to a video file (YouTube URLs supported)"
    echo "  --direction TEXT  Direction of fluid impact (default: \"from the front\")"
    echo "  --object TEXT     Override object description (used with --video)"
    exit 1
fi

# Run the x2sim pipeline with provided arguments
python x2sim.py "$@"

# Print completion message
echo "X2Sim job completed at $(date)"
#!/bin/bash
#SBATCH -J X2Sim_Run                            # Job name
#SBATCH -o log.X2Sim_Run.o%j                    # Name of stdout output file
#SBATCH -e log.X2Sim_Run.e%j                    # Name of stderr error file
#SBATCH -p gpu-a100-dev                         # Queue (partition) name - using GPU for TRELLIS
#SBATCH -N 1                                    # Total # of nodes
#SBATCH -n 1                                    # Total # of mpi tasks
#SBATCH -t 01:00:00                             # Run time (hh:mm:ss)
#SBATCH --mail-type=all                         # Send email at begin and end of job
#SBATCH --mail-user=XXXX@cXXXX.XXX              # Replace with your email
#SBATCH -A XXXXXXXX                             # Project/Allocation name (replace with yours)

# Load requisite modules from host
module load gcc/11.2.0
module load cuda/12.2
module load python3
module load tacc-apptainer

# Exit on error
set -e

# Navigate to target directory
cd $SCRATCH/x2sim

# Load credentials
source credentials/credentials.sh

# Check if container exists, otherwise pull it
count=`ls -1 container/*.sif 2>/dev/null | wc -l`
if [ $count == 0 ]; then
    echo "Container not found, pulling from GitHub Container Registry..."
    apptainer pull container/x2sim.sif docker://ghcr.io/geoelements/x2sim:0.1
else
    echo "Using existing sif file found in ./container"
fi

# Help function to display usage
function show_help {
    echo "Usage: sbatch jobrun_x2sim.sh [options]"
    echo ""
    echo "Input options (choose one):"
    echo "  --prompt TEXT     Text prompt describing the object and simulation"
    echo "  --image PATH      Path to an image file"
    echo "  --video URL/PATH  URL or path to a video file (YouTube URLs supported)"
    echo ""
    echo "Additional options:"
    echo "  --direction TEXT  Direction of fluid impact (default: \"from the front\")"
    echo ""
    echo "Examples:"
    echo "  sbatch jobrun_x2sim.sh --prompt \"A cube being hit with water\""
    echo "  sbatch jobrun_x2sim.sh --image \$SCRATCH/x2sim/jobs/testinputs/building.png --direction \"from the side\""
    echo "  sbatch jobrun_x2sim.sh --video \$SCRATCH/x2sim/input_video.mp4 --direction \"from above\""
}

# Check if no parameters are provided
if [ "$#" -lt 1 ]; then
    show_help
    exit 1
fi

echo "Running x2sim with parameters: $@, at $(date). DEBUG: "$@""

# Run the simulation using the container
apptainer exec --nv \
    --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    --bind "/etc/pki:/etc/pki:ro" \
    ./container/x2sim.sif \
    python3 $SCRATCH/x2sim/x2sim/x2sim.py "$@"

# Print completion message
echo "X2Sim job completed at $(date)"

# TODO: Add virtual environment option if needed.

# # Check for virtual environment flag
# venv_flag=0
# if [[ $@ == *"--use-venv"* ]]; then
#     venv_flag=1
# else
#     venv_flag=0
# fi

# # Option 1: Run x2sim with Docker container (default)
# if [ $venv_flag == 0 ]; then
#     # Load credentials
#     source credentials/credentials.sh
    
#     # Check if container exists, otherwise pull it
#     count=`ls -1 container/*.sif 2>/dev/null | wc -l`
#     if [ $count == 0 ]; then
#         echo "Container not found, pulling from GitHub Container Registry..."
#         apptainer pull container/x2sim.sif docker://ghcr.io/geoelements/x2sim:0.1
#     else
#         echo "Using existing sif file found in ./container"
#     fi
    
#     echo "Running x2sim through docker container"
#     apptainer exec --nv \
#         --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
#         --bind "/etc/pki:/etc/pki:ro" \
#         ./container/x2sim.sif \
#         bash -c "python3 \$SCRATCH/x2sim/x2sim/x2sim.py $@"

# # Option 2: Run x2sim with local virtual environment
# else
#     # Load credentials
#     source credentials/credentials.sh
    
#     # Setup virtual environment
#     echo "Setting up virtual environment"
#     bash setup/setup.sh
    
#     # Activate virtual environment
#     source x2sim_env/bin/activate
    
#     echo "Running x2sim through virtual environment"
#     python3 x2sim/x2sim.py "$@"
# fi
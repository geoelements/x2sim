#!/bin/bash
#SBATCH -J X2Sim_Run                     	# Job name
#SBATCH -o log.X2Sim_Run.o%j                 	# Name of stdout output file
#SBATCH -p gpu-a100-dev                  	# Queue (partition) name - using GPU for TRELLIS
#SBATCH -N 1                             	# Total # of nodes
#SBATCH -n 1                             	# Total # of mpi tasks
#SBATCH -t 01:00:00                      	# Run time (hh:mm:ss)
#SBATCH --mail-type=all                  	# Send email at begin and end of job
#SBATCH --mail-user=XXXX  		# Replace with your email
#SBATCH -A XXXXXX                      	# Project/Allocation name (replace with yours)

# Load requisite modules from host
module load gcc/11.2.0
module load cuda/12.2
module load python3
module load tacc-apptainer

# Exit on error
set -e

# Check if parameters are provided
#if [ "$#" -lt 1 ]; then
#    echo "Usage: sbatch run_x2sim.sh --prompt \"Your prompt here\" [options]"
#    echo "   or: sbatch run_x2sim.sh --video \"video_url_or_path\" [options]"
#    echo ""
#    echo "Options:"
#    echo "  --prompt TEXT     Text prompt describing the object and simulation"
#    echo "  --video URL/PATH  URL or path to a video file (YouTube URLs supported)"
#    echo "  --direction TEXT  Direction of fluid impact (default: \"from the front\")"
#    echo "  --object TEXT     Override object description (used with --video)"
#    echo "  --use-venv        Create and/or use virtual environment for package management (default is to use docker container)
#    exit 1
#fi



# Check for virtual environment flag
venv_flag=0
if [[ $@ == *"--use-venv"* ]]; then
        venv_flag=1
else
        venv_flag=0
fi

# Option 1: Run x2sim with Docker container (default)
if [ $venv_flag == 0 ]; then
        source credentials/credentials.sh
	count=`ls -1 container/*.sif 2>/dev/null | wc -l`

        if [ $count == 0  ]; then
                apptainer pull container/x2sim.sif docker://ghcr.io/geoelements/x2sim:0.1
        else
                echo "Using existing sif file found in ./container"
        fi
        echo "Running x2sim through docker container"
        apptainer exec --nv --env OPEN_API_KEY=$OPEN_API_KEY container/x2sim.sif python3 x2sim/x2sim.py "$@"

# Option 2: Run x2sim with local virtual environment
else
        source credentials/credentials.sh
        echo "Setting up virtual environment"
        bash setup/setup.sh
	echo "Running x2sim through virtual environment"
        python3 x2sim/x2sim.py "$@"
fi



# Print completion message
echo "X2Sim job completed at $(date)"

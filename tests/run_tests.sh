#!/bin/bash
#SBATCH -J X2Sim_Run                            # Job name
#SBATCH -o log.X2Sim_Run.o%j                    # Name of stdout output file
#SBATCH -e log.X2Sim_Run.e%j                    # Name of stderr error file
#SBATCH -p gpu-a100-dev                         # Queue (partition) name - using GPU for TRELLIS
#SBATCH -N 1                                    # Total # of nodes
#SBATCH -n 1                                    # Total # of mpi tasks
#SBATCH -t 01:00:00                             # Run time (hh:mm:ss)
#SBATCH --mail-type=all                         # Send email at begin and end of job
#SBATCH --mail-user=smithl@tacc.utexas.edu              # Replace with your email
#SBATCH -A OTH24002                             # Project/Allocation name (replace with yours)

# Load requisite modules from host
module load gcc/11.2.0
module load cuda/12.2
module load python3
module load tacc-apptainer

# Exit on error
set -e

# Navigate to the proper working directory
cd $SCRATCH/x2sim

# Source OPEN_API credentials
source credentials/credentials.sh

# Test 1: Text Input
echo "Testing text input capability"
input_text=$(cat "tests/text/test_prompt_text.txt")
apptainer exec --nv \
    --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    ./container/x2sim.sif \
    python3 $SCRATCH/x2sim/x2sim/x2sim.py --prompt "$input_text" &> tests/text/log.test
echo "Text input test sucessful!"
mv outputs/* tests/text/outputs


# Test 2: Single Image Input
echo "Testing image input capability"
input_image_path="tests/image/test_prompt_image.png"
apptainer exec --nv \
    --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    ./container/x2sim.sif \
    python3 $SCRATCH/x2sim/x2sim/x2sim.py --image "$input_image_path" &> tests/image/log.test
echo "Image input test sucessful!"
mv outputs/* tests/image/outputs

# Test 3: Video (image directory) input
echo "Testing video (image directory) input capability"
input_video_path="tests/image_dir"
apptainer exec --nv \
    --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    ./container/x2sim.sif \
    python3 $SCRATCH/x2sim/x2sim/x2sim.py --image "$input_video_path" &> tests/image_dir/log.test
echo "Video (image directory) input test sucessful!"
mv outputs/* tests/image_dir/outputs

# Test 4: Video (URL) input
echo "Testing video (url) input capability"
input_url=$(cat "tests/url/test_prompt_url.txt")
apptainer exec --nv \
    --env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    ./container/x2sim.sif \
    python3 $SCRATCH/x2sim/x2sim/x2sim.py --video "$input_url" &> tests/url/log.test
echo "Video (url) input test sucessful!"
mv outputs/* tests/url/outputs

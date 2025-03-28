# If --env flag is set, setup and activate a virtual environment
python3 -m venv x2sim-env
source x2sim-env/bin/activate

# Requirements specific to x2sim
pip3 install -r x2sim/requirements.txt

# Requirements specific to taichi mpm
pip3 install -r taichi

# Requirements specific to trellis text-to-3D pipeline
cd trellis
bash setup.sh
cd ..

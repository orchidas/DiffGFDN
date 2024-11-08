#!/bin/bash -l
# change current working directory
#SBATCH --chdir=/scratch/users/%u/DiffGFDN/
# set output directory
#SBATCH --output=/scratch/users/%u/%j.out
#SBATCH --job-name=gpu
#SBATCH --gres=gpu
# Load required modules
module load python/3.11.6-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0 
module load cudnn/8.7.0.84-11.8-gcc-13.2.0
# Path to the virtual environment 
VENV_PATH=".venv" 
# Path to the Python script you want to run 
PYTHON_SCRIPT="src/run_model.py"
CONFIG_PATH="data/config/gpu_training_od.yml" 
# Activate the virtual environment 
source "$VENV_PATH/bin/activate" 
# Run the Python script 
python3 "$PYTHON_SCRIPT" -c "$CONFIG_PATH"
nvidia-debugdump -l
# Deactivate the virtual environment 
deactivate
echo “Done executing script”
module purge

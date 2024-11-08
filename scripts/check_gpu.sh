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
echo "Loaded required modules"
# Path to the virtual environment 
VENV_PATH=".venv"
source "$VENV_PATH/bin/activate"
ipython <<EOF
# Your Python code here
import sys
import torch
print("Running from within a bash script!")
print("Python version:", sys.version)
print("Is CUDA available?", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
EOF
deactivate
module purge

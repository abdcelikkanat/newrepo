#!/bin/bash
#SBATCH --job-name=MODEL
#SBATCH --output=%x_%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0-48:00:00
#SBATCH --partition=prioritized

nvidia-smi
set -eu

# Critical for PyTorch distributed
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NCCL_SOCKET_IFNAME=bond0
echo "port" ${MASTER_PORT} "==" ${MASTER_ADDR} "==" ${SLURM_JOB_NODELIST}

# Define the global variables
BASEFOLDER=~/workspace/kum
PYTHON="srun singularity exec --nv ${BASEFOLDER}/../kum_project/vib.sif python"
SCRIPT_PATH=${BASEFOLDER}/src/model.py
INPUT_PATH=$BASEFOLDER/../../kadir/dataset/train_2m.csv

# Run the bash script
source ${BASEFOLDER}/sh/run_model.sh "$BASEFOLDER" "$PYTHON" "$SCRIPT_PATH" "$INPUT_PATH"


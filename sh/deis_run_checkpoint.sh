#!/bin/bash
#SBATCH --job-name=ICML #Give your experiment a name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=naples  # turing --> gpu, dhabi, naples and rome ---> cpu
#SBATCH --mem=120G
#SBATCH --mail-type=NONE  # Type of email notification: BEGIN,END,FAIL,ALL,NONE
#SBATCH --mail-user=abce@cs.aau.dk
#SBATCH --time=72:00:00  #  time limit in dd:hh:mm:ss format. 


#nvidia-smi
set -eu

# Critical for PyTorch distributed
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NCCL_SOCKET_IFNAME=bond0
echo "port" ${MASTER_PORT} "==" ${MASTER_ADDR} "==" ${SLURM_JOB_NODELIST}

# Define the global variables
BASEFOLDER=/nfs/home/cs.aau.dk/zs74qz/workspace/icml
PYTHON=/nfs/home/cs.aau.dk/zs74qz/miniconda3/envs/kum/bin/python
SCRIPT_PATH=${BASEFOLDER}/src/model.py
INPUT_PATH=$BASEFOLDER/../../kadir/dataset/train_2m.csv

# Run the bash script
source ${BASEFOLDER}/sh/run_checkpoint.sh "$BASEFOLDER" "$PYTHON" "$SCRIPT_PATH" "$INPUT_PATH"

#!/bin/bash
#SBATCH --job-name=DIM #Give your experiment a name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=naples  # turing --> gpu, dhabi, naples and rome ---> cpu
#SBATCH --mem=96G
#SBATCH --mail-type=NONE  # Type of email notification: BEGIN,END,FAIL,ALL,NONE
#SBATCH --mail-user=abce@cs.aau.dk
#SBATCH --time=48:00:00  #  time limit in dd:hh:mm:ss format. 

#nvidia-smi
set -eu

# Critical for PyTorch distributed
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export NCCL_SOCKET_IFNAME=bond0
echo "port" ${MASTER_PORT} "==" ${MASTER_ADDR} "==" ${SLURM_JOB_NODELIST}

# Define the global variables
BASEFOLDER=/nfs/home/cs.aau.dk/zs74qz/workspace/icml_final
PYTHON=/nfs/home/cs.aau.dk/zs74qz/miniconda3/envs/kum/bin/python
SCRIPT_PATH=${BASEFOLDER}/src/model.py
INPUT_PATH=$BASEFOLDER/../../kadir/dataset/train_2m.csv
SEED=4
DIM=512 #512 #128 #64 #16 #4

# Train the model for mean vectors for 50 epochs
source ${BASEFOLDER}/sh/train_dim_analysis.sh "${BASEFOLDER}" "$PYTHON" "$SCRIPT_PATH" "$INPUT_PATH" "${SEED}" "50" "mean_only" "" "${DIM}"

# Train the model for variance vectors for 25 epochs
CHECKPOINT_NAME="k=4_d=${DIM}_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=mahalanobis_strategy=mean_only_seed=${SEED}"
source ${BASEFOLDER}/sh/train_dim_analysis.sh "${BASEFOLDER}" "$PYTHON" "$SCRIPT_PATH" "$INPUT_PATH" "${SEED}" "25" "var_only" "${CHECKPOINT_NAME}" "${DIM}"

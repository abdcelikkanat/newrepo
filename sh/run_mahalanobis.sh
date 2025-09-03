#!/bin/bash
#SBATCH --job-name=DEV4
#SBATCH --output=%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --partition=prioritized
#num_gpus=2
# Set the number of tasks and GPUs accordingly
####SBATCH --ntasks=$num_gpus
####SBATCH --gres=gpu:a10:$num_gpus

nvidia-smi
set -eu

# Critical for PyTorch distributed
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "port" ${MASTER_PORT} "==" ${MASTER_ADDR} "==" ${SLURM_JOB_NODELIST} "--" 

#export NCCL_DEBUG=INFO # For printing the debug related to NCCL
export NCCL_SOCKET_IFNAME=bond0
#export NCCL_P2P_LEVEL=PHB

# Define the global variables
NAME=scalable
BASEFOLDER=~/workspace/kum_project #/home/cs.aau.dk/zs74qz/workspace/kum_project
PYTHON="srun singularity exec --nv ${BASEFOLDER}/vib.sif python"
SCRIPT_PATH=${BASEFOLDER}/src/scalable.py

# Model Parameters
INPUT_PATH=$BASEFOLDER/../../kadir/dataset/train_2m.csv
LOSS_NAME="mahalanobis"
POSTFIX="_filter=0.01"
K=4
OUT_DIM=256
NEGSAMPLEPERPOS=200
MAXSEQNUM=100000
LR=0.001
BATCH_SIZE=10000
SAVE_EVERY=5
DISTRIBUTED=1
DEVICE=None
SEED=1
EPOCHNUM=100

# Define the output path
NAME=${NAME}_k=${K}_d=${OUT_DIM}_negsampleperpos=${NEGSAMPLEPERPOS}
NAME=${NAME}_maxseq=${MAXSEQNUM}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}
NAME=${NAME}_loss=${LOSS_NAME}_seed=${SEED}${POSTFIX}

OUTPUT_PATH=${BASEFOLDER}/models/${NAME}.model
LOG_DIR=${BASEFOLDER}/logs/${NAME}

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input $INPUT_PATH --output_path ${OUTPUT_PATH} --k ${K} --out_dim ${OUT_DIM}"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_seq_num ${MAXSEQNUM}"
CMD="${CMD} --epoch $EPOCHNUM --lr $LR --batch_size ${BATCH_SIZE} --save_every ${SAVE_EVERY} --distributed ${DISTRIBUTED}"
CMD="${CMD} --device ${DEVICE} --loss_name ${LOSS_NAME} --log_dir ${LOG_DIR} --seed ${SEED}"


if [ -e ${OUTPUT_PATH} ]; then
    echo "File exists!"
else
    # Run the command
    $CMD
fi


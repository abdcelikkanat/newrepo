#!/bin/bash
#SBATCH --job-name=FILTER  # Give your experiment a name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=naples  # turing --> gpu, dhabi, naples and rome ---> cpu
#SBATCH --mem=120G
#SBATCH --mail-type=NONE  # Type of email notification: BEGIN,END,FAIL,ALL,NONE
#SBATCH --mail-user=abce@cs.aau.dk
#SBATCH --time=48:00:00  #  time limit in dd:hh:mm:ss format. 


# Define the global variables
BASEFOLDER=/nfs/home/cs.aau.dk/zs74qz/workspace/icml_final
PYTHON=/nfs/home/cs.aau.dk/zs74qz/miniconda3/envs/kum/bin/python
SCRIPT_PATH=${BASEFOLDER}/analysis/ablation/filtering.py
RESULTS_FOLDER=${BASEFOLDER}/results/ablation/filtering
SPECIES_LIST=(reference plant marine)
SEED_LIST=(1 2 3 4 5)

# Create the folder if it doesn't exist
mkdir -p ${RESULTS_FOLDER}
export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

for SEED in ${SEED_LIST[@]}
do
for SPECIES in ${SPECIES_LIST[@]}
do

MODELFILE="k=4_d=256_negperpos=200_maxseq=100000_epoch=25_lr=0.001_batch=100000_loss=mahalanobis_strategy=var_only_seed=${SEED}.model"
MODEL_PATH=${BASEFOLDER}/models/${MODELFILE}

echo $MODEL_PATH
sbatch ${BASEFOLDER}/sh/filter.sh "${BASEFOLDER}" "${PYTHON}" "${SCRIPT_PATH}" "${RESULTS_FOLDER}" "${MODEL_PATH}" "${SPECIES}"

done
done

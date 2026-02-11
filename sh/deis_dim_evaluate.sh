#!/bin/bash
#SBATCH --job-name=EVALDIM  # Give your experiment a name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=rome #dhabi  # turing --> gpu, dhabi, naples and rome ---> cpu
#SBATCH --mem=120G
#SBATCH --mail-type=NONE  # Type of email notification: BEGIN,END,FAIL,ALL,NONE
#SBATCH --mail-user=abce@cs.aau.dk
#SBATCH --time=48:00:00  #  time limit in dd:hh:mm:ss format. 


# Define the global variables
BASEFOLDER=/nfs/home/cs.aau.dk/zs74qz/workspace/icml_final
PYTHON=/nfs/home/cs.aau.dk/zs74qz/miniconda3/envs/kum/bin/python
SCRIPT_PATH=${BASEFOLDER}/evaluation/binning.py
RESULTS_FOLDER=${BASEFOLDER}/results/ablation/dim
SPECIES_LIST=( reference plant marine)
SEED_LIST=( 1 2 3 4 5 )
DIM_LIST=( 4 16 64 128 512 )

mkdir -p ${RESULTS_FOLDER}
export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

for SEED in ${SEED_LIST[@]}
do
for DIM in ${DIM_LIST[@]}
do
for SPECIES in ${SPECIES_LIST[@]}
do

echo $SPECIES - $DIM - $SPECIES

MODELFILE="k=4_d=${DIM}_negperpos=200_maxseq=100000_epoch=25_lr=0.001_batch=100000_loss=mahalanobis_strategy=var_only_seed=${SEED}.model"
MODEL_PATH=${BASEFOLDER}/models/ablation/dim/${MODELFILE}

sbatch ${BASEFOLDER}/sh/dim_evaluate.sh "${BASEFOLDER}" "${PYTHON}" "${SCRIPT_PATH}" "${RESULTS_FOLDER}" "${MODEL_PATH}" "${SPECIES}"

done
done
done

#!/bin/bash
#SBATCH --job-name=EVALUATION  # Give your experiment a name
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
RESULTS_FOLDER=${BASEFOLDER}/results

export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

source ${BASEFOLDER}/sh/evaluate.sh "${BASEFOLDER}" "${PYTHON}" "${SCRIPT_PATH}" "${RESULTS_FOLDER}"



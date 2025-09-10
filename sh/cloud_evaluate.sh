#!/usr/bin/bash -l
#SBATCH --job-name=EVALUATION
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=prioritized
#SBATCH --cpus-per-task=4
#SBATCH --mem=720G
#SBATCH --time=0-12:00:00

# Define the global variables
BASEFOLDER=/home/cs.aau.dk/zs74qz/workspace/kum/
PYTHON="singularity exec --nv ${BASEFOLDER}/../../containers/nn python"
SCRIPT_PATH=${BASEFOLDER}/evaluation/binning.py
RESULTS_FOLDER=${BASEFOLDER}/results

export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

source ${BASEFOLDER}/sh/evaluate.sh "${BASEFOLDER}" "${PYTHON}" "${SCRIPT_PATH}" "${RESULTS_FOLDER}"



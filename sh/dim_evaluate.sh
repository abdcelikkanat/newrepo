#!/bin/bash
#SBATCH --job-name=EVALDIM  # Give your experiment a name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=rome #dhabi  # turing --> gpu, dhabi, naples and rome ---> cpu
#SBATCH --mem=120G
#SBATCH --mail-type=NONE  # Type of email notification: BEGIN,END,FAIL,ALL,NONE
#SBATCH --mail-user=abce@cs.aau.dk
#SBATCH --time=48:00:00  #  time limit in dd:hh:mm:ss format. 

export PYTHONUNBUFFERED=1

# Define the global variables
BASEFOLDER=$1
PYTHON=$2
SCRIPT_PATH=$3
RESULTS_FOLDER=$4
MODEL_PATH=$5
SPECIES=$6

export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

# Check if a folder exists or not
if ! [ -d ${RESULTS_FOLDER} ]; then
mkdir ${RESULTS_FOLDER}
fi
if ! [ -d ${RESULTS_FOLDER}/reference ]; then
mkdir ${RESULTS_FOLDER}/reference
fi
if ! [ -d ${RESULTS_FOLDER}/marine ]; then
mkdir ${RESULTS_FOLDER}/marine
fi
if ! [ -d ${RESULTS_FOLDER}/plant ]; then
mkdir ${RESULTS_FOLDER}/plant
fi

# Define the model name
MODELFILE="$(basename ${MODEL_PATH})"

# Define the evaluation parameters
#SPECIES_LIST=("marine") #("reference" "plant" "marine" )
DATA_DIR=${BASEFOLDER}/../../kadir/dataset/
FEATURE_FOLDER=${BASEFOLDER}/temp_features${POSTFIX}
#MODEL_PATH=${BASEFOLDER}/models/${MODELFILE}
MODELLIST=ours
METRIC="mahalanobis"
SAMPLE_IDS="5,6"
CHUNK_SIZE=100
#FILTER_STRATEGY="before"
#FILTER_RATIO="0.01"
POSTFIX="_metric-${METRIC}"

# Define output path
OUTPUT_PATH="${RESULTS_FOLDER}/${SPECIES}/${MODELFILE}${POSTFIX}.results"
#CONFUSION_MATRIX_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELFILE}${POSTFIX}.confusion"

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH}"
CMD=$CMD" --data_dir ${DATA_DIR}"
#CMD=$CMD" --feature_folder ${FEATURE_FOLDER}"
CMD=$CMD" --model_path ${MODEL_PATH}"
CMD=$CMD" --model_list ${MODELLIST}"
CMD=$CMD" --metric ${METRIC}"
CMD=$CMD" --chunk_size ${CHUNK_SIZE}"
#CMD=$CMD" --filter_strategy ${FILTER_STRATEGY}"
#CMD=$CMD" --filter_ratio ${FILTER_RATIO}"
#CMD=$CMD" --confusion_matrix_path ${CONFUSION_MATRIX_PATH}"
CMD=$CMD" --species ${SPECIES}"
CMD=$CMD" --sample_ids ${SAMPLE_IDS}"
CMD=$CMD" --output ${OUTPUT_PATH}"
$CMD

# Remove the feature folder
rm -rf ${FEATURE_FOLDER}




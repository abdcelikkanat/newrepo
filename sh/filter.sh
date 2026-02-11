#!/bin/bash
#SBATCH --job-name=FILTER  # Give your experiment a name
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=naples  # turing --> gpu, dhabi, naples and rome ---> cpu
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


# Define the evaluation parameters
DATA_DIR=${BASEFOLDER}/../../kadir/dataset/
MODELLIST=ours
METRIC="mahalanobis"
SAMPLE_IDS=( 6 ) #(5 6)
CHUNK_SIZE=100
POSTFIX=""

for ID in ${SAMPLE_IDS[@]}
do

# Define output path
FILENAME="$(basename ${MODEL_PATH})"
OUTPUT_PATH="${RESULTS_FOLDER}/${SPECIES}/${FILENAME}${POSTFIX}_id=${ID}.pkl"
mkdir -p ${RESULTS_FOLDER}/${SPECIES}

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH}"
CMD=$CMD" --data_dir ${DATA_DIR}"
CMD=$CMD" --model_path ${MODEL_PATH}"
CMD=$CMD" --model_list ${MODELLIST}"
CMD=$CMD" --metric ${METRIC}"
CMD=$CMD" --chunk_size ${CHUNK_SIZE}"
CMD=$CMD" --species ${SPECIES}"
CMD=$CMD" --sample_ids ${SAMPLE_IDS}"
CMD=$CMD" --output ${OUTPUT_PATH}"
$CMD

# Remove the feature folder
rm -rf ${FEATURE_FOLDER}

done

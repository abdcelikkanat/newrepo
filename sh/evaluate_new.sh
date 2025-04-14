#!/usr/bin/bash -l
#SBATCH --job-name=EVALUATION
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=0-12:00:00

# Define the global variables
BASEFOLDER=/home/cs.aau.dk/zs74qz/workspace/kum_project/
PYTHON="singularity exec --nv ${BASEFOLDER}/../../containers/nn python"
SCRIPT_PATH=${BASEFOLDER}/evaluation/binning_new.py
RESULTS_FOLDER=${BASEFOLDER}/results

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


# Define the model file name
MODELFILE="scalable_train_2m_k=4_d=256_negsampleperpos=200_maxseq=100000_epoch=5_LR=0.001_batch=10000_device=None_loss=bottle_seed=1_test12.model"

# Define the evaluation parameters
SPECIES_LIST=("reference") #("reference" "plant" "marine")
DATA_DIR=${BASEFOLDER}/../../kadir/dataset/
FEATURE_FOLDER=${BASEFOLDER}/temp_features
MODEL_PATH=${BASEFOLDER}/models/${MODELFILE}
MODELLIST=scalable_vib #nonlinear
METRIC="l2" #" this should be bottle after finishing the implementation.
SAMPLE_IDS="5" #"5,6"
CHUNK_SIZE=100

for SPECIES in ${SPECIES_LIST[@]}
do

# Define output path
OUTPUT_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELFILE}.txt"

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH}"
CMD=$CMD" --data_dir ${DATA_DIR}"
CMD=$CMD" --feature_folder ${FEATURE_FOLDER}"
CMD=$CMD" --model_path ${MODEL_PATH}"
CMD=$CMD" --model_list ${MODELLIST}"
CMD=$CMD" --metric ${METRIC}"
CMD=$CMD" --chunk_size ${CHUNK_SIZE}"
CMD=$CMD" --species ${SPECIES}"
CMD=$CMD" --sample_ids ${SAMPLE_IDS}"
CMD=$CMD" --output ${OUTPUT_PATH}"
$CMD

done





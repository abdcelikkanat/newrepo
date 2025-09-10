
export PYTHONUNBUFFERED=1

# Define the global variables
BASEFOLDER=$1
PYTHON=$2
SCRIPT_PATH=$3
RESULTS_FOLDER=$4

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
MODELFILE="scalable_k=4_d=256_negsampleperpos=200_maxseq=100000_epoch=100_LR=0.001_batch=10000_loss=mahalanobis_seed=1_multiview_nonoverlap_3parts.model.epoch_95.checkpoint"

# Define the evaluation parameters
POSTFIX="_test"
SPECIES_LIST=("reference")
DATA_DIR=${BASEFOLDER}/../../kadir/dataset/
FEATURE_FOLDER=${BASEFOLDER}/temp_features${POSTFIX}
MODEL_PATH=${BASEFOLDER}/../kum_project/models/${MODELFILE}
MODELLIST=ours
METRIC="squared_l2" #"l2"
SAMPLE_IDS="5" #"5,6"
CHUNK_SIZE=100
FILTER_STRATEGY="after"
FILTER_RATIO="0.05"

for SPECIES in ${SPECIES_LIST[@]}
do
# Define output path
OUTPUT_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELFILE}${POSTFIX}.txt"
CONFUSION_MATRIX_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELFILE}${POSTFIX}.txt"

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH}"
CMD=$CMD" --data_dir ${DATA_DIR}"
#CMD=$CMD" --feature_folder ${FEATURE_FOLDER}"
CMD=$CMD" --model_path ${MODEL_PATH}"
CMD=$CMD" --model_list ${MODELLIST}"
CMD=$CMD" --metric ${METRIC}"
CMD=$CMD" --chunk_size ${CHUNK_SIZE}"
CMD=$CMD" --filter_strategy ${FILTER_STRATEGY}"
CMD=$CMD" --filter_ratio ${FILTER_RATIO}"
CMD=$CMD" --confusion_matrix_path ${CONFUSION_MATRIX_PATH}"
CMD=$CMD" --species ${SPECIES}"
CMD=$CMD" --sample_ids ${SAMPLE_IDS}"
CMD=$CMD" --output ${OUTPUT_PATH}"
$CMD

done




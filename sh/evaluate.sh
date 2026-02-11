
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
MODELFILE="k=4_d=256_negperpos=200_maxseq=100000_epoch=25_lr=0.001_batch=100000_loss=mahalanobis_strategy=var_only_seed=1.model"

# Define the evaluation parameters
SPECIES_LIST=("reference" "plant" "marine" )
DATA_DIR=${BASEFOLDER}/../../kadir/dataset/
FEATURE_FOLDER=${BASEFOLDER}/temp_features${POSTFIX}
MODEL_PATH=${BASEFOLDER}/models/${MODELFILE}
MODELLIST=ours
METRIC="mahalanobis"
SAMPLE_IDS="5,6"
CHUNK_SIZE=100
POSTFIX="_metric-${METRIC}"

for SPECIES in ${SPECIES_LIST[@]}
do

# Define output path
OUTPUT_PATH="${RESULTS_FOLDER}/${SPECIES}/${MODELFILE}${POSTFIX}.results"

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

done




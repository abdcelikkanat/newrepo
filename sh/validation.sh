
export PYTHONUNBUFFERED=1

# Define the global variables
BASEFOLDER=$1
PYTHON=$2
SCRIPT_PATH=$3
RESULTS_FOLDER=$4
MODEL_PATH=$5

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
SPECIES_LIST=("reference")
EPOCH_LIST=(5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95)
DATA_DIR=~/Downloads/kum/dataset/ #-----${BASEFOLDER}/../../kadir/dataset/
MODELLIST=ours
METRIC="squared_l2"
CHUNK_SIZE=100
POSTFIX=""


for EPOCH in ${EPOCH_LIST[@]}
do
#Define model path
MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=100_lr=0.001_batch=10000_loss=mahalanobis_strategy=mean_only_seed=1_macbook.model"
MODELFILE="${MODELFILE}.epoch_${EPOCH}.checkpoint"
MODEL_PATH=${BASEFOLDER}/../kum/models/${MODELFILE}
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
CMD=$CMD" --output ${OUTPUT_PATH}"
$CMD

done

done


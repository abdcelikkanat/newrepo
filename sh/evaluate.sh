
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
POSTFIX="_metric-l2"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=mahalanobis_strategy=mean_only_seed=1.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_builtin_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_implementation_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v1_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=mahalanobis_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_builtin_strategy=mean_only_seed=2.model.epoch_40.checkpoint"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=25_lr=0.001_batch=100000_loss=l2_builtin_strategy=mean_only_seed=2_hingePLUSl2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_05builtin_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v2_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_025builtin_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v3_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v4_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v5_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v5_strategy=mean_only_seed=2.model.epoch_10.checkpoint"
MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v6_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=l2_constrained_v6_strategy=mean_only_seed=2.model.epoch_10.checkpoint"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=hinge_strategy=mean_only_seed=2.model"
#MODELFILE="model_k=4_d=256_negperpos=200_maxseq=100000_epoch=50_lr=0.001_batch=100000_loss=hinge_strategy=mean_only_seed=2.model.epoch_10.checkpoint"
#MODELFILE="${MODELFILE}.epoch_${TEMP_EPOCH}.checkpoint"

# Define the evaluation parameters
SPECIES_LIST=("marine") #("reference")
DATA_DIR=${BASEFOLDER}/../../kadir/dataset/
FEATURE_FOLDER=${BASEFOLDER}/temp_features${POSTFIX}
MODEL_PATH=${BASEFOLDER}/models/${MODELFILE}
MODELLIST=ours
METRIC="l2" #-----
SAMPLE_IDS="5,6"
CHUNK_SIZE=100
#FILTER_STRATEGY="before"
#FILTER_RATIO="0.01"

for SPECIES in ${SPECIES_LIST[@]}
do

# Define output path
OUTPUT_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELFILE}${POSTFIX}.results"
CONFUSION_MATRIX_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELFILE}${POSTFIX}.confusion"

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

rm -rf ${FEATURE_FOLDER}

done




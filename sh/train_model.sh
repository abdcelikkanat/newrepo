
export PYTHONUNBUFFERED=1

# Define the global variables
BASEFOLDER=$1
PYTHON=$2
SCRIPT_PATH=$3
INPUT_PATH=$4

# Model Parameters
POSTFIX=""
K=4
OUT_DIM=256
NEGSAMPLEPERPOS=200
MAXSEQNUM=100000
LR=0.001
BATCH_SIZE=100000
SAVE_EVERY=5
DISTRIBUTED=0
DEVICE=cpu
SEED=$5
LOSS_NAME="mahalanobis"
EPOCHNUM=$6
STRATEGY=$7 #mean_only #sequential #----together #----mean_only
CHECKPOINT_NAME=${8-""}

# Define the model name
MODELNAME="k=${K}_d=${OUT_DIM}_negperpos=${NEGSAMPLEPERPOS}_maxseq=${MAXSEQNUM}_epoch=${EPOCHNUM}_lr=${LR}"
MODELNAME="${MODELNAME}_batch=${BATCH_SIZE}_loss=${LOSS_NAME}_strategy=${STRATEGY}_seed=${SEED}${POSTFIX}"

# Define output and log path
OUTPUT_PATH=${BASEFOLDER}/models/${MODELNAME}.model
LOG_DIR=${BASEFOLDER}/logs/${MODELNAME}

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input $INPUT_PATH --output ${OUTPUT_PATH} --k ${K} --out_dim ${OUT_DIM}"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_seq_num ${MAXSEQNUM}"
CMD="${CMD} --epoch $EPOCHNUM --lr $LR --batch_size ${BATCH_SIZE} --save_every ${SAVE_EVERY} --distributed ${DISTRIBUTED}"
CMD="${CMD} --device ${DEVICE} --loss_name ${LOSS_NAME}  --strategy ${STRATEGY} --log_dir ${LOG_DIR} --seed ${SEED}"
if [[ ${CHECKPOINT_NAME} != "" ]]; then
CMD="${CMD} --checkpoint ${BASEFOLDER}/models/${CHECKPOINT_NAME}.model"
fi

if [ -e ${OUTPUT_PATH} ]; then
    echo "File exists!"
else
    $CMD
fi



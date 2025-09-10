
# Define the global variables
#BASEFOLDER=$1
#PYTHON=$2
#SCRIPT_PATH=$3
#INPUT_PATH=$4

# Model Parameters
NAME=model
LOSS_NAME="mahalanobis"
POSTFIX=""
K=4
OUT_DIM=256
NEGSAMPLEPERPOS=200
MAXSEQNUM=1000 #100000
LR=0.001
BATCH_SIZE=10000
SAVE_EVERY=5
DISTRIBUTED=1
DEVICE=None
SEED=1
EPOCHNUM=10 #100

# Define the output path
NAME=${NAME}_k=${K}_d=${OUT_DIM}_negperpos=${NEGSAMPLEPERPOS}
NAME=${NAME}_maxseq=${MAXSEQNUM}_epoch=${EPOCHNUM}_lr=${LR}_batch=${BATCH_SIZE}
NAME=${NAME}_loss=${LOSS_NAME}_seed=${SEED}${POSTFIX}

OUTPUT_PATH=${BASEFOLDER}/models/${NAME}.model
LOG_DIR=${BASEFOLDER}/logs/${NAME}

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input $INPUT_PATH --output ${OUTPUT_PATH} --k ${K} --out_dim ${OUT_DIM}"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_seq_num ${MAXSEQNUM}"
CMD="${CMD} --epoch $EPOCHNUM --lr $LR --batch_size ${BATCH_SIZE} --save_every ${SAVE_EVERY} --distributed ${DISTRIBUTED}"
CMD="${CMD} --device ${DEVICE} --loss_name ${LOSS_NAME} --log_dir ${LOG_DIR} --seed ${SEED}"


if [ -e ${OUTPUT_PATH} ]; then
    echo "File exists!"
else
    $CMD
fi



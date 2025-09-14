# Define the global variables
BASEFOLDER=/Users/zs74qz/Downloads/kum
PYTHON=/Users/zs74qz/Downloads/kum/envs/kum/bin/python
SCRIPT_PATH=${BASEFOLDER}/src/model.py
INPUT_PATH=$BASEFOLDER/dataset/train_2m.csv

# Run the bash script
source ${BASEFOLDER}/sh/run_model.sh "$BASEFOLDER" "$PYTHON" "$SCRIPT_PATH" "$INPUT_PATH"


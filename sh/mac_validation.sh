
# Define the global variables
BASEFOLDER=/Users/zs74qz/Downloads/kum
PYTHON=/Users/zs74qz/Downloads/kum/envs/kum/bin/python
SCRIPT_PATH=${BASEFOLDER}/evaluation/clustering.py
RESULTS_FOLDER=${BASEFOLDER}/validation_results

export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

source ${BASEFOLDER}/sh/validation.sh "${BASEFOLDER}" "${PYTHON}" "${SCRIPT_PATH}" "${RESULTS_FOLDER}"



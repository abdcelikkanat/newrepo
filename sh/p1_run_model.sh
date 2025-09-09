#!/bin/sh
### General options
### -- specify queue --
#BSUB -q p1
### -- set the job Name --
#BSUB -J MODEL
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=32GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
#BSUB -M 32GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u abce@cs.aau.dk
### -- send notification at start --
###BSUB -B
### -- send notification at completion --
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o MODEL_%J.out
#BSUB -e MODEL_%J.out

module load cuda/12.6.2
nvidia-smi
set -eu

# Define the global variables
BASEFOLDER=~/workspace/kum
PYTHON=~/workspace/semibin3_project/envs/kum/bin/python
SCRIPT_PATH=${BASEFOLDER}/src/model.py
INPUT_PATH=$BASEFOLDER/../../datasets/revisitingkmers/train/train_2m.csv

source ${BASEFOLDER}/sh/run_model.sh $BASEFOLDER $PYTHON $SCRIPT_PATH $INPUT_PATH



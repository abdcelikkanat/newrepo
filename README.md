## K-mers for Uncertainty Modelling (KUM)

### Setting up the pyhon environment
You can install all the required packages by running the following commands:
```
conda create -n ENVIRONMENTAL_NAME python=3.9
conda activate ENVIRONMENTAL_NAME
pip install -r requirements.txt
```

### Datasets
We first need the ```gdown``` package to get the data. 
```
pip install gdown
```

Then, please run the following commands to download the training dataset:
```
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp
unzip dnabert-s_train.zip
```
and the evaluation datasets can be retrieved similarly.
```
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c
unzip dnabert-s_eval.zip
```

### Training
You can train the model by setting the required parameters. An example is given below:
```
python src/nonlinear.py --input DATA_FOLDER_PATH/train_2m.csv --output EMB_FILE_PATH --max_read_num 100 --device cpu
```
Note that the training set contains two million sequence pairs, but you can randomly sample desired amount of sequences
by setting the ```max_read_num``` parameter. 

For more details, you can run the following command:
```
python src/nonlinear.py --help
```

### Evaluation
```
python evaluation/binning.py --data_dir DATA_FOLDER --model_list kum --metric symmetric_kl --species SPECIES --test_model_dir EMB_FILE_PATH --output OUTPUT"
```

Here, the ```--data_dir``` parameter must be set to the path of the folder containing the evaluation datasets, 
```--species``` the name of the evaluation data file ( i.e. "reference" "plant" "marine") and 
```test_model_dir``` the path of the trained model file. The last parameter, ```--output```, might be set to
a text file that will be used to store the number of high quality bins.
import os.path

import transformers
import torch
import tqdm
import numpy as np
import time
import sys
from src.model import Model


def get_model(model_name, model_name_or_path):

    # if the model is UncertainGen
    if model_name == "ours":

        model_path = model2info[model_name]["path"]
        kwargs, model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model = Model(**kwargs)
        model.load_state_dict(model_state_dict)

    # if the model is NT
    elif model_name == "nt":
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    elif model_name == "dnabert2":

        config = transformers.models.bert.configuration_bert.BertConfig.from_pretrained(model_name_or_path)
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            config=config,
        )
    else:
        model = transformers.AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    # Move the model to CPU
    model.to("cpu")

    return model

def get_average_time(model2info, train_loader):

    # Get the average time over all the training data
    average_time = 0
    #
    for batch in train_loader:

        if model_name != "ours":

            # Get the model max length
            model_max_length = model2info[model_name]["max_length"]

            # Define the tokenizer
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=None,
                model_max_length=model_max_length,
                padding_side="right",
                use_fast=True,
                trust_remote_code=True,
            )

        with torch.no_grad():

            # Get the initial time
            init_time = time.time()

            if model_name == "ours":

                embedding, cov = model.seq2emb(dna_sequences)

            else:

                token_feat = tokenizer.batch_encode_plus(
                    batch,
                    max_length=model_max_length,
                    return_tensors='pt',
                    padding='longest',
                    truncation=True
                )
                input_ids = token_feat['input_ids']
                # Get the embedding
                emb = model.forward(input_ids=input_ids)[0].detach().cpu()

            # Get the elapsed time
            elapsed_time = time.time() - init_time

            average_time += elapsed_time

    average_time = average_time / len(train_loader)

    return average_time

model2info = {
    "hyenadna": {"path": "LongSafari/hyenadna-medium-450k-seqlen-hf", "max_length": 20000},
    "dnabert2": {"path": "zhihan1996/DNABERT-2-117M", "max_length": 5000},
    "nt": {"path": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "max_length": 2048},
    "dnaberts": {"path": "zhihan1996/DNABERT-S", "max_length": 5000},
    "ours": {"path": os.path.join(os.getcwd(), "rebuttal", "k=4_d=256_negperpos=200_maxseq=100000_epoch=25_lr=0.001_batch=100000_loss=mahalanobis_strategy=var_only_seed=1.model")}
}

if __name__ == "__main__":

    data_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    subsample_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    batch_size = 10

    # Read the data file
    dna_sequences = []
    with open(data_file_path, 'r') as f:
        for line in f.readlines():
            left_seq, right_seq = line.strip().split(',')
            dna_sequences.append(left_seq)
            dna_sequences.append(right_seq)

    # Randomly choose 'subsample_size' sequences
    dna_sequences = [dna_sequences[i] for i in np.random.randint(0, len(dna_sequences), subsample_size)]

    # Define the train loader
    train_loader = torch.utils.data.DataLoader(
        dna_sequences, batch_size=batch_size, shuffle=False, num_workers=1
    )

    model2time = {}
    for model_name in model2info:

        # Get the path
        model_name_or_path = model2info[model_name]["path"]
        # Get the model
        model = get_model(model_name, model_name_or_path)

        # Get the average time
        model2time[model_name] = get_average_time(model2info=model2info, train_loader=train_loader)

    # Save the output file
    with open(output_file_path, 'w') as f:
        for model_name in model2time:
            f.write(f"{model_name}: {model2time[model_name]}\n")

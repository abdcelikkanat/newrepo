# This script has been adapted from the file available at the following address:
# https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/eval_binning.py
import csv
import argparse
import os
import sys
import numpy as np
import sklearn.metrics
import time
from datetime import datetime
from evaluation.utils import align_labels_via_hungarian_algorithm
from evaluation.utils import get_embedding, KMedoid, compute_class_center_medium_similarity
from evaluation.utils import filter_sequences, convert_labels2int, clean_by_variance

MAX_SEQ_LEN = 20000
MIN_SEQ_LEN = 2500
MIN_ABUNDANCE_VALUE = 10


def main(args):

    model_list = args.model_list.split(",")
    for model_name in model_list:
        for species in args.species.split(","):
            for sample_id in map(int, args.sample_ids.split(",")):

                # Define the metric for the given method
                metric = args.metric
                if metric == None:
                    raise ValueError(f"Metric was not chosen for the model {model_name}")

                print("Computing the clustering results")
                print(f"+ Model: {model_name} | Species: {species} | Sample ID: {sample_id} | Metric: {metric} | Chunk size: {args.chunk_size}")
                print(f"+ Model path: {args.model_path}")

                ########################################################################################################
                # =================== Compute the threshold value required for the k-Medoid algorithm ==================
                ########################################################################################################
                # Load the clustering data to compute similarity threshold
                init_time = time.time()
                clustering_data_file_path = os.path.join(args.data_dir, species, f"clustering_0.tsv")

                dna_sequences, labels = filter_sequences(
                    data_path=clustering_data_file_path, shorten=MAX_SEQ_LEN, min_len = 0, abundance = 0,
                )
                # Convert the string labels to integer labels
                labels = convert_labels2int(labels)
                print(f"+ Initial filtering of clustering_0.tsv was done in {time.time() - init_time:.2f} seconds")

                # Get embeddings (and covariances) of the sequences in the dataset named clustering
                init_time = time.time()
                features = get_embedding(
                    dna_sequences=dna_sequences, model_name=model_name, species=species, sample_id=0,
                    k=args.k, model_path=args.model_path, embedding_file_path="",
                )
                print(f"\t- Computing the clustering file embeddings (done in {time.time() - init_time:.2f} seconds")

                # We need to compute the similarity threshold that is required for the k-medoid algorithm
                init_time = time.time()
                percentile_values = compute_class_center_medium_similarity(
                    features, labels, metric=metric, chunk_size=args.chunk_size
                )
                threshold = percentile_values[-3]
                print(f"\t- Threshold value: {threshold} ({time.time() - init_time:.2f} seconds)")

                ########################################################################################################
                # ========================== Evaluate the model for the validation set =================================
                ########################################################################################################

                # Run the k-medoid algorithm
                init_time = time.time()
                binning_results = KMedoid(
                    features, min_similarity=threshold, min_bin_size=10,
                    max_iter=1000, metric=metric, chunk_size=args.chunk_size
                )
                print(f"\t- Computing the clustering results (done in {time.time() - init_time:.2f} seconds)")

                # Get the number of true labels and predicted labels
                true_labels_bin = labels[binning_results != -1]
                predicted_labels = binning_results[binning_results != -1]
                print("\t- Number of predicted labels: ", len(predicted_labels))

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(true_labels_bin, predicted_labels)
                predicted_labels_bin = [alignment_bin[label] for label in predicted_labels]

                # Calculate the precision
                precision_bin = sklearn.metrics.precision_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                precision_bin.sort()

                # Calculate the recall
                recall_bin = sklearn.metrics.recall_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                recall_bin.sort()

                # calculate the f1 score
                f1_bin = sklearn.metrics.f1_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                f1_bin.sort()

                # Define the threshold values
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                precision_results, recall_results, f1_results = [], [], []
                for threshold in thresholds:
                    precision_results.append(precision_bin[precision_bin > threshold].shape[0])
                    recall_results.append(len(np.where(recall_bin > threshold)[0]))
                    f1_results.append(len(np.where(f1_bin > threshold)[0]))

                print(f"+ f1_results: {f1_results}")
                print(f"+ precision_results: {precision_results}")
                print(f"+ recall_results: {recall_results} \n")

                with open(args.output, 'a+') as f:
                    f.write("\n")
                    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    f.write(f"model: {model_name}, species: {species}, sample_id: {sample_id}, binning\n")
                    f.write(f"recall_results: {recall_results}\n")
                    f.write(f"precision_results: {precision_results}\n")
                    f.write(f"f1_results: {f1_results}\n")
                    f.write(f"threshold: {threshold}\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument(
        '--species', type=str, default="reference,marine,plant", help='Species to evaluate'
    )
    parser.add_argument(
        '--sample_ids', type=str, default="5,6",
        help='Species to evaluate'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output file path'
    )
    parser.add_argument(
        '--model_path', type=str, default="",
        help='Path to the pretrained model (if required by the method)'
    )
    parser.add_argument(
        '--model_list', type=str, default="dnaberts",
        help='List of models to evaluate, separated by comma. Currently support [ours]'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Data directory'
    )
    parser.add_argument(
        '--k', type=int, default=4,
        help="k Value for the kmerprofile method"
    )
    parser.add_argument(
        '--metric', type=str, default=None,
        help="Metric to measure the similarities among embeddings"
    )
    parser.add_argument(
        '--chunk_size', type=int, default=0,
        help="Defines the size of the chunk to be used for the similarity matrix computation"
    )

    args = parser.parse_args()
    main(args)

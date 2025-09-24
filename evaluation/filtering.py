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
import pickle

MAX_SEQ_LEN = 20000
MIN_SEQ_LEN = 2500
MIN_ABUNDANCE_VALUE = 10


def remove_sequences(filter_ratios, thresholds, sequence_num, sorted_seq_indices, true_labels_bin, predicted_labels_bin):

    # Define the recall, precision and f1 lists to store the results
    recall_list, precision_list, f1_list = [], [], []

    for filtering_ratio in filter_ratios:
        print(f"\t- Filtering ratio: {filtering_ratio:.2f}")
        # Define the number of sequences to be filtered
        num_sequences_to_filter = int(sequence_num * filtering_ratio)
        chosen_indices = sorted_seq_indices[:num_sequences_to_filter]

        # Construct copies of the predicted and ground truth labels
        current_pred_labels = np.asarray(predicted_labels_bin).copy()
        current_ground_truth_labels = np.asarray(true_labels_bin).copy()

        # # Define the indices to be removed based on the random strategy
        # chosen_indices = np.random.choice(len(current_pred_labels), size=num_sequences_to_filter, replace=False)
        # current_ground_truth_labels[chosen_indices] = -1 # We do not need to change the ground truth labels
        current_pred_labels[chosen_indices] = -1
        # Construct the total distinct labels
        all_labels = np.sort(np.unique(np.concatenate((current_ground_truth_labels, current_pred_labels))))

        current_class_recall_scores = sklearn.metrics.recall_score(
            y_true=current_ground_truth_labels, y_pred=current_pred_labels, labels=all_labels, average=None
        )[1:]  # We do not consider the garbage class
        current_class_recall_scores.sort()

        current_class_precision_scores = sklearn.metrics.precision_score(
            y_true=current_ground_truth_labels, y_pred=current_pred_labels, labels=all_labels, verage=None
        )[1:]  # We do not consider the garbage class
        current_class_precision_scores.sort()

        current_class_f1_scores = sklearn.metrics.f1_score(
            y_true=current_ground_truth_labels, y_pred=current_pred_labels, labels=all_labels, verage=None
        )[1:]  # We do not consider the garbage class
        current_class_f1_scores.sort()

        # Calculate the recall scores for different thresholds
        recall_list.append(
            [sum(current_class_recall_scores > threshold) for threshold in thresholds]
        )
        # Calculate the precision scores for different thresholds
        precision_list.append(
            [sum(current_class_precision_scores > threshold) for threshold in thresholds]
        )
        # Calculate the f1 scores for different thresholds
        f1_list.append(
            [sum(current_class_f1_scores > threshold) for threshold in thresholds]
        )

    return recall_list, precision_list, f1_list

def main(args):


    scores = {}
    model_list = args.model_list.split(",")
    for model_name in model_list:
        for species in args.species.split(","):
            for sample_id in map(int, args.sample_ids.split(",")):

                # Define the metric for the given method
                metric = args.metric
                if metric == None:
                    raise ValueError(f"Metric was not chosen for the model {model_name}")

                print(f"+ Model: {model_name} | Species: {species} | Sample ID: {sample_id} | Metric: {metric} | Chunk size: {args.chunk_size}")

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
                # ========================================= Evaluate the model =========================================
                ########################################################################################################
                # Load binning data
                init_time = time.time()
                data_file = os.path.join(args.data_dir, species, f"binning_{sample_id}.tsv")

                # Filter the sequences based on the minimum length and abundance
                dna_sequences, labels_bin = filter_sequences(
                    data_path=data_file, shorten=MAX_SEQ_LEN, min_len=MIN_SEQ_LEN, abundance=MIN_ABUNDANCE_VALUE,
                )
                # # Relabel the string labels to integer labels
                labels_bin = convert_labels2int(labels_bin)

                print(f"+ Initial filtering of binning_{sample_id}.tsv was done in {time.time() - init_time:.2f} seconds")

                # Generate embeddings for the binning set
                init_time = time.time()
                features = get_embedding(
                    dna_sequences=dna_sequences, model_name=model_name, species=species, sample_id=sample_id,
                    k=args.k, model_path=args.model_path, embedding_file_path="",
                )
                print(f"\t- Embeddings of binning_{sample_id}.tsv were computed in {time.time() - init_time:.2f} seconds")
                # Run the k-medoid algorithm
                init_time = time.time()
                binning_results = KMedoid(
                    features, min_similarity=threshold, min_bin_size=10,
                    max_iter=1000, metric=metric, chunk_size=args.chunk_size
                )

                print(f"\t- Computing the binning results (done in {time.time() - init_time:.2f} seconds)")
                # Get the number of true labels and predicted labels
                true_labels_bin = labels_bin[binning_results != -1]
                predicted_labels = binning_results[binning_results != -1]
                print("\t- Number of predicted labels: ", len(predicted_labels))

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(true_labels_bin, predicted_labels)
                predicted_labels_bin = [alignment_bin[label] for label in predicted_labels]

                # Define the threshold values
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                # Define filter ratios to be used
                filter_ratios = np.arange(0, 1.0, 0.01)


                scores = {}
                for strategy in ["random", "cov"]:
                    print(f"\t- Strategy: {strategy}")

                    if strategy == "random":
                        sorted_seq_indices = np.random.permutation(len(true_labels_bin))
                    elif strategy == "cov":
                        _, covs = features
                        # means = means[binning_results != -1]
                        covs = covs[binning_results != -1]
                        sorted_seq_indices = np.sort(covs, axis=-1)[:, int(covs.shape[1]*0.5)]

                    # Get the recall, precision and f1 lists after filtering the sequences
                    recall_list, precision_list, f1_list = remove_sequences(
                        filter_ratios=filter_ratios, thresholds=thresholds, sequence_num=len(true_labels_bin),
                        sorted_seq_indices=sorted_seq_indices,
                        true_labels_bin=true_labels_bin, predicted_labels_bin=predicted_labels_bin
                    )

                    scores[model_name][species][sample_id][strategy] = {
                        "filter_ratios": filter_ratios,
                        "thresholds": thresholds,
                        "recall": recall_list,
                        "precision": precision_list,
                        "f1": f1_list
                    }

    # Save the scores dictionary to a file
    with open(args.output_file, "wb") as f:
        pickle.dump(scores, f)


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
        help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, dnaberts, kmerprofile, ours]'
    )
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
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

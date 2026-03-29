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

    # Check if the filtering strategy and ratio are consistent
    if (args.filter_strategy == "" and args.filter_ratio != 0) or (args.filter_strategy != "" and args.filter_ratio == 0):
        raise ValueError("If filter_strategy is chosen, filter_ratio must be > 0 and vice versa")

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

                # # Define the embedding/feature file path. If file exist, we don't need to recompute the dna embeddings
                cluster_embs_file_path = ""
                if args.feature_folder != "":
                    filename = os.path.basename(args.model_path) + ".features"
                    cluster_embs_file_path = os.path.join(args.feature_folder, species, f"clustering_{sample_id}",filename)
                # Get embeddings (and covariances) of the sequences in the dataset named clustering
                init_time = time.time()
                features = get_embedding(
                    dna_sequences=dna_sequences, model_name=model_name, species=species, sample_id=0,
                    k=args.k, model_path=args.model_path, embedding_file_path=cluster_embs_file_path,
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
                # Clean the sequences that have the largest minimum covariances
                if args.filter_ratio != 0:
                    selected_idx = clean_by_variance(
                        dna_sequences, model_path=args.model_path, filter_ratio=args.filter_ratio
                    )

                # Clean the sequences if the strategy is before the clustering step
                if args.filter_ratio != 0 and args.filter_strategy == "before":
                    print(f"\t+ Cleaning the sequences before the clustering step (Ratio: {args.filter_ratio})")
                    print(f"\t- {len(selected_idx)} sequences left out of {len(dna_sequences)}")
                    dna_sequences = [dna_sequences[i] for i in selected_idx]
                    labels_bin = np.asarray(labels_bin)[selected_idx]
                # Relabel the string labels to integer labels
                labels_bin = convert_labels2int(labels_bin)

                print(f"+ Initial filtering of binning_{sample_id}.tsv was done in {time.time() - init_time:.2f} seconds")

                # Define the embedding/feature file path. If the file exist, we don't recompute the dna embeddings
                embedding_file_path = ""
                if args.feature_folder != "":
                    filename = os.path.basename(args.model_path) + ".features"
                    embedding_file_path = os.path.join(args.feature_folder, species, f"binning_{sample_id}", filename)
                # Generate embeddings for the binning set
                init_time = time.time()
                features = get_embedding(
                    dna_sequences=dna_sequences, model_name=model_name, species=species, sample_id=sample_id,
                    k=args.k, model_path=args.model_path, embedding_file_path=embedding_file_path,
                )
                print(f"\t- Embeddings of binning_{sample_id}.tsv were computed in {time.time() - init_time:.2f} seconds")
                # Run the k-medoid algorithm
                init_time = time.time()
                binning_results = KMedoid(
                    features, min_similarity=threshold, min_bin_size=10,
                    max_iter=1000, metric=metric, chunk_size=args.chunk_size
                )
                # Clean the sequences if the strategy is after the clustering step
                if args.filter_ratio != 0 and args.filter_strategy == "after":
                    idx = 0
                    sorted_selected_indices = sorted(selected_idx)
                    for i in range(len(binning_results)):
                        if i == sorted_selected_indices[idx]:
                            idx += 1
                        else:
                            binning_results[i] = -1

                        if idx == len(sorted_selected_indices):
                            binning_results[i:] = [-1] * (len(binning_results) - i)
                            break

                    print(f"\t+ Cleaning the sequences after the clustering step (Ratio: {args.filter_ratio})")
                    print(f"\t- In total, {len(binning_results) - len(selected_idx)} sequences have been determined as outliers by variance")
                    print(f"\t- {sum(binning_results == -1) - len(selected_idx)} sequences among them have been also determined as outliers by the clustering algorithm")


                print(f"\t- Computing the binning results (done in {time.time() - init_time:.2f} seconds)")
                # Get the number of true labels and predicted labels
                true_labels_bin = labels_bin[binning_results != -1]
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

                # calculate the NMI score
                nmi = sklearn.metrics.normalized_mutual_info_score(
                    true_labels_bin, predicted_labels_bin
                )

                # calculate the AMI score
                ami = sklearn.metrics.normalized_mutual_info_score(
                    true_labels_bin, predicted_labels_bin
                )

                # calculate the micro precision, recall and f1
                micro_precision, micro_recall, micro_f1, _ = sklearn.metrics.precision_recall_fscore_support(
                    true_labels_bin, predicted_labels_bin, average='micro'
                )
                # calculate the macro precision, recall and f1
                macro_precision, macro_recall, macro_f1, _ = sklearn.metrics.precision_recall_fscore_support(
                    true_labels_bin, predicted_labels_bin, average='macro'
                )

                # calculate the average completeness and contamination
                average_completeness = macro_recall
                average_contamination = 1 - macro_precision

                # If the confusion matrix path is given, save the confusion matrix and true/predicted labels
                if args.confusion_matrix_path != "":
                    confusion_matrix = sklearn.metrics.confusion_matrix(true_labels_bin, predicted_labels_bin)
                    np.savetxt(args.confusion_matrix_path, confusion_matrix, delimiter="\t")
                    # Save the true and predicted labels as numpy arrays
                    np.savez(
                        args.confusion_matrix_path + "_true_predicted_labels_model.npz",
                        true_labels=true_labels_bin, predicted_labels=predicted_labels_bin
                    )

                # Compute the NMI/ARI score
                with open(args.output, 'a+') as f:
                    f.write("\n")
                    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    f.write(f"model: {model_name}, species: {species}, sample_id: {sample_id}, binning\n")
                    f.write(f"nmi: {nmi}\n")
                    f.write(f"ami: {ami}\n")
                    f.write(f"micro_f1: {micro_f1}\n")
                    f.write(f"macro_f1: {macro_f1}\n")
                    f.write(f"avg_completeness: {average_completeness}\n")
                    f.write(f"avg_contamination: {average_contamination}\n\n")


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
        '--feature_folder', type=str, default="",
        help='Folder path to store the features/embeddings of the sequences in order to avoid recomputing them'
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
    parser.add_argument(
        '--filter_strategy', type=str, default="",
        help="Defines the filtering strategy to be used (before or after clustering)"
    )
    parser.add_argument(
        '--filter_ratio', type=float, default=0,
        help="Filter ratio to be used (0 means no filtering)"
    )
    parser.add_argument(
        '--confusion_matrix_path', type=str, default="",
        help="Path to save the confusion matrix (if given)"
    )


    args = parser.parse_args()
    main(args)

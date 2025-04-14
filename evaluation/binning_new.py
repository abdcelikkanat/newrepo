# This script has been adapted from the file available at the following address:
# https://github.com/MAGICS-LAB/DNABERT_S/blob/main/evaluate/eval_binning.py
import csv
import argparse
import os
import sys
import collections
import numpy as np
import sklearn.metrics
import time
from datetime import datetime
from evaluation.utils_new import align_labels_via_hungarian_algorithm
from evaluation.utils_new import get_embedding, KMedoid, compute_class_center_medium_similarity
from evaluation.utils_new import filter_sequences

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxsize)
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

                print(f"+ Model: {model_name} | Species: {species} | Sample ID: {sample_id} | Metric: {metric} | Chunk size: {args.chunk_size}")

                ########################################################################################################
                # =================== Compute the threshold value required for the k-Medoid algorithm ==================
                ########################################################################################################
                # Load the clustering data to compute similarity threshold
                # init_time = time.time()
                clustering_data_file_path = os.path.join(args.data_dir, species, f"clustering_0.tsv")
                # with open(clustering_data_file_path, "r") as f:
                #     reader = csv.reader(f, delimiter="\t")
                #     data = list(reader)[1:]
                # print(f"\t- Reading clustering data file (done in {time.time() - init_time:.2f} seconds)")
                #
                # # Shorten the sequences if they are longer than the MAX_SEQ_LEN value
                # dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                # labels = [d[1] for d in data]
                #
                # # Convert the sequence labels to numeric values
                # label2id = {l: i for i, l in enumerate(set(labels))}
                # labels = np.array([label2id[l] for l in labels])
                # num_clusters = len(label2id)
                # print(f"\t- Clustering data contains {len(dna_sequences)} sequences with {num_clusters} clusters.")
                keep_ratio = 0.95
                min_max_normalization = True
                dna_sequences, labels = filter_sequences(
                    data_path=clustering_data_file_path, model_path=args.model_path,
                    keep_ratio=keep_ratio, min_max_normalization=min_max_normalization
                )

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
                # init_time = time.time()
                data_file = os.path.join(args.data_dir, species, f"binning_{sample_id}.tsv")
                # with open(data_file, "r") as f:
                #     reader = csv.reader(f, delimiter="\t")
                #     data = list(reader)[1:]
                # print(f"+ Reading the binning file (binning_{sample_id}.tsv) ({time.time() - init_time:.2f} seconds)")

                # # Get the dna sequences nd labels, and shorten the sequences whose length is longer than 'MAX_SEQ_LEN'
                # dna_sequences = [d[0][:MAX_SEQ_LEN] for d in data]
                # labels_bin = [d[1] for d in data]
                #
                # # Filter sequences with length < 2500
                # filterd_idx = [i for i, seq in enumerate(dna_sequences) if len(seq) >= MIN_SEQ_LEN]
                # dna_sequences = [dna_sequences[i] for i in filterd_idx]
                # labels_bin = [labels_bin[i] for i in filterd_idx]
                #
                # # Filter sequences with low abundance labels (less than 10)
                # label_counts = collections.Counter(labels_bin)
                # filterd_idx = [i for i, l in enumerate(labels_bin) if label_counts[l] >= MIN_ABUNDANCE_VALUE]
                # dna_sequences = [dna_sequences[i] for i in filterd_idx]
                # labels_bin = [labels_bin[i] for i in filterd_idx]
                #
                # # Convert labels to numeric values
                # label2id = {l: i for i, l in enumerate(set(labels_bin))}
                # labels_bin = np.array([label2id[l] for l in labels_bin])
                # num_clusters = len(label2id)
                # print(f"\t- Get {len(dna_sequences)} sequences, {num_clusters} clusters")
                dna_sequences, labels_bin = filter_sequences(
                    data_path=data_file, model_path=args.model_path, keep_ratio=keep_ratio, min_max_normalization=min_max_normalization
                )

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
                print(f"\t- Computing the binning results (done in {time.time() - init_time:.2f} seconds)")
                # Get the number of true labels and predicted labels
                true_labels_bin = labels_bin[binning_results != -1]
                predicted_labels = binning_results[binning_results != -1]
                print("\t- Number of predicted labels: ", len(predicted_labels))

                # Align labels
                alignment_bin = align_labels_via_hungarian_algorithm(true_labels_bin, predicted_labels)
                predicted_labels_bin = [alignment_bin[label] for label in predicted_labels]

                # Calculate purity, completeness, recall, and ARI
                recall_bin = sklearn.metrics.recall_score(
                    true_labels_bin, predicted_labels_bin, average=None, zero_division=0
                )
                recall_bin.sort()

                f1_bin = sklearn.metrics.f1_score(true_labels_bin, predicted_labels_bin, average=None, zero_division=0)
                f1_bin.sort()
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                recall_results = []
                f1_results = []
                for threshold in thresholds:
                    recall_results.append(len(np.where(recall_bin > threshold)[0]))
                    f1_results.append(len(np.where(f1_bin > threshold)[0]))

                print(f"+ f1_results: {f1_results}")
                print(f"+ recall_results: {recall_results} \n")

                with open(args.output, 'a+') as f:
                    f.write("\n")
                    f.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                    f.write(f"model: {model_name}, species: {species}, sample_id: {sample_id}, binning\n")
                    f.write(f"recall_results: {recall_results}\n")
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
        help='Directory to save trained models to test'
    )
    parser.add_argument(
        '--feature_folder', type=str, default="",
        help='Folder path to store the features/embeddings of the sequences in order to avoid recomputing them'
    )
    parser.add_argument(
        '--model_list', type=str, default="dnaberts",
        help='List of models to evaluate, separated by comma. Currently support [tnf, tnf-k, dnabert2, hyenadna, nt, dnaberts, kmerprofile, vib]'
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

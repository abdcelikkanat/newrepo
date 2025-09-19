import os
import numpy as np
import matplotlib.pyplot as plt
# Read the result files and write them into a csv file

# Define the dataset name
dataset_name = "marine"
batch_size = "100000"
postfix = f"_batch={batch_size}"
# Define the model name
model_name = f"model_k=4_d=256_negperpos=200_maxseq=100000_epoch=100_lr=0.001_batch={batch_size}_loss=mahalanobis_strategy=mean_only_seed=1_main_cpu.model"

# Define epoch list
epoch_list = list(range(5, 100, 5))

# Define the folder
results_dir = os.path.join(os.path.dirname(__file__), "..", "deis_evaluation_results", dataset_name)

recall_list, precision_list, f1_list = [], [], []
for epoch in epoch_list:
    current_results_file = os.path.join( results_dir, f"{model_name}.epoch_{epoch}.checkpoint_main_cpu.results")
    with open(current_results_file, "r") as f:
        # Skip the headers
        for _ in range(2):
            f.readline()
        # Read recall scores
        current_recall_scores = list(map(int, f.readline().strip()[17:-1].split(', ')))
        # Read precision scores
        current_precision_scores = list(map(int, f.readline().strip()[20:-1].split(', ')))
        # Read f1 scores
        current_f1_scores = list(map(int, f.readline().strip()[13:-1].split(', ')))

        # Append the scores
        recall_list.append(current_recall_scores)
        precision_list.append(current_precision_scores)
        f1_list.append(current_f1_scores)

# Define the csv file path
csv_file_path = os.path.join( results_dir, f"compiled_results{postfix}.csv")
# Write the results to a csv file
with open(csv_file_path, "w") as f:

    f.write(f"{dataset_name}, Clustering 0\n")

    for epoch, recall_scores, precision_scores, f1_scores in zip(epoch_list, recall_list, precision_list, f1_list):
        f.write("\n")
        f.write(f"Epoch={epoch}\n")
        f.write(f",0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9\n")
        f.write(f"Recall,{','.join(map(str, recall_scores))}\n")
        f.write(f"Precision,{','.join(map(str, recall_scores))}\n")
        f.write(f"F1,{','.join(map(str, f1_scores))}\n")
        f.write("\n")

# Plot the top 3 high quality bin recall/precision/f1 scores side by side
recall_list, precision_list, f1_list = np.asarray(recall_list), np.asarray(precision_list), np.asarray(f1_list)

# Plot top 3 high quality scores for recall, precision and f1 side by side
plt.figure(figsize=(18, 8))
plt.xticks([])   # remove x ticks
plt.yticks([])   # remove y ticks
plt.title(f"{dataset_name} Evaluation Results")

plt.subplot(1, 3, 1)
plt.plot(epoch_list, recall_list[:, -3:])
plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.xticks(ticks=epoch_list, labels=[str(e) for e in epoch_list], rotation=45)

plt.subplot(1, 3, 2)
plt.plot(epoch_list, precision_list[:, -3:],)
plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.xticks(ticks=epoch_list, labels=[str(e) for e in epoch_list], rotation=45)

plt.subplot(1, 3, 3)
plt.plot(epoch_list, f1_list[:, -3:], )
plt.grid(True)
plt.xlabel("Epoch")
plt.ylabel("F1")
plt.xticks(ticks=epoch_list, labels=[str(e) for e in epoch_list], rotation=45)

#
# Add a global legend with labels, 'a', 'b', and 'c'
plt.legend([">0.7", ">0.8", ">0.9"], loc='center right', ncol=1, bbox_to_anchor=(1.25, 0.5) )
plt.tight_layout()

# plt.show()
plt.savefig(os.path.join(results_dir, f"{dataset_name}_evaluation_results{postfix}.png"))
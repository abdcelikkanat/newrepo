from xml.etree.ElementInclude import include

import torch
import math
import time
import sys
import random
import pickle as pkl
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from functools import partial
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import re
import os

def set_seed(seed):
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)


class PairDataset(Dataset):
    def __init__(self, file_path, transform_func, neg_sample_per_pos=1000, max_read_num=0, verbose=True, seed=0):

        # Set the parameters
        self.__both_kmer_profiles = None
        self.__transform_func = transform_func
        self.__neg_sample_per_pos = neg_sample_per_pos
        self.__seed = seed

        # Set the seed
        set_seed(seed)

        # Get the number of lines
        with open(file_path, 'r') as f:
            lines_num = sum(1 for _ in f)
        # If the max_read_num is set, then sample the line number to read
        if max_read_num > 0:
            chosen_lines = random.sample(range(lines_num), max_read_num)
            chosen_lines.sort()

        # Read the file
        chosen_line_idx = 0
        left_kmer_profiles, right_kmer_profiles = [], []
        with open(file_path, 'r') as f:
            for current_line_idx, line in enumerate(f):

                if max_read_num > 0:
                    if chosen_line_idx == len(chosen_lines):
                        break

                    if current_line_idx != chosen_lines[chosen_line_idx]:
                        continue
                    else:
                        chosen_line_idx += 1

                # Remove the newline character and commas
                left_read, right_read = line.strip().split(',')
                left_kmer_profiles.append(self.__transform_func(left_read))
                right_kmer_profiles.append(self.__transform_func(right_read))

        # Combine the left and right k-mer profiles
        self.__both_kmer_profiles = torch.from_numpy(
            np.asarray(left_kmer_profiles + right_kmer_profiles)
        ).to(torch.float)

        if verbose:
            print(f"The data file was read successfully!")
            print(f"\t+ Total number of read pairs: {lines_num}")
            if max_read_num > 0:
                print(f"\t+ Number of read pairs used: {max_read_num}")

        # Temporary variables
        self.__ones = torch.ones((len(self.__both_kmer_profiles),))

    def __len__(self):

        return len(self.__both_kmer_profiles) // 2

    def __getitem__(self, idx):

        # Sample negative sample_indices
        negative_sample_indices = torch.multinomial(
            self.__ones, replacement=True, num_samples=2*self.__neg_sample_per_pos
        )

        # Define the positive and negative k-mer profile pairs
        left_kmer_profiles = torch.concatenate((
            self.__both_kmer_profiles[idx].unsqueeze(0),
            self.__both_kmer_profiles[negative_sample_indices[:self.__neg_sample_per_pos]]
        ))
        right_kmer_profiles = torch.concatenate((
            self.__both_kmer_profiles[idx+self.__len__()].unsqueeze(0),
            self.__both_kmer_profiles[negative_sample_indices[self.__neg_sample_per_pos:]]
        ))
        # Define the labels
        labels = torch.tensor([1] + [0] * self.__neg_sample_per_pos, dtype=torch.float)

        return left_kmer_profiles, right_kmer_profiles, labels


class VIBModel(torch.nn.Module):
    def __init__(self, k, dim=256, device=torch.device("cpu"), verbose=False, seed=0):
        super(VIBModel, self).__init__()

        # Set the parameters
        self.__device = device
        self.__verbose = verbose

        # Define the letters, k-mer size, and the base complement
        self.__k = k
        self.__dim = dim
        self.__letters = ['A', 'C', 'G', 'T']
        self.__kmer2id = {''.join(kmer): i for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))}
        self.__kmers_num = len(self.__kmer2id)

        # Set the seed
        set_seed(seed)

        # Define the layers
        self.linear1_mean = torch.nn.Linear(self.__kmers_num, 512, dtype=torch.float, device=self.__device)
        self.batch1_mean = torch.nn.BatchNorm1d(512, dtype=torch.float, device=self.__device)
        self.activation1_mean = torch.nn.Sigmoid()
        self.dropout1_mean = torch.nn.Dropout(0.2)
        self.linear2_mean = torch.nn.Linear(512, self.__dim, dtype=torch.float, device=self.__device)

        self.linear1_std = torch.nn.Linear(self.__kmers_num, 512, dtype=torch.float, device=self.__device)
        self.batch1_std = torch.nn.BatchNorm1d(512, dtype=torch.float, device=self.__device)
        self.activation1_std = torch.nn.Sigmoid()
        self.dropout1_std = torch.nn.Dropout(0.2)
        self.linear2_std = torch.nn.Linear(512, self.__dim, dtype=torch.float, device=self.__device)

    def mean_encoder(self, kmer_profile):

        mean = self.linear1_mean(kmer_profile)
        mean = self.batch1_mean(mean)
        mean = self.activation1_mean(mean)
        mean = self.dropout1_mean(mean)
        mean = self.linear2_mean(mean)

        return mean

    def std_encoder(self, kmer_profile):

        std = self.linear1_mean(kmer_profile)
        std = self.batch1_mean(std)
        std = self.activation1_mean(std)
        std = self.dropout1_mean(std)
        std = self.linear2_mean(std)
        std = torch.nn.functional.softplus(std) + 1e-6

        return std

    def forward(self, left_kmer_profile, right_kmer_profile):

        left_mean, right_mean = self.mean_encoder(left_kmer_profile), self.mean_encoder(right_kmer_profile)
        left_std, right_std = self.std_encoder(right_kmer_profile), self.std_encoder(right_kmer_profile)

        return left_mean, left_std, right_mean, right_std

    def get_k(self):
        return self.__k

    def get_dim(self):
        return self.__dim

    def get_device(self):
        return self.__device

    def read2kmer_profile(self, read, normalized=True):

        # Get the k-mer profile
        kmer2id = [self.__kmer2id[read[i:i + self.__k]] for i in range(len(read) - self.__k + 1)]
        kmer_profile = np.bincount(kmer2id, minlength=self.__kmers_num)

        if normalized:
            kmer_profile = kmer_profile / kmer_profile.sum()

        return kmer_profile

    def read2emb(self, reads, normalized=True):

        with torch.no_grad():
            self.eval()
            kmer_profiles = []
            for read in reads:
                kmer_profiles.append(self.read2kmer_profile(read, normalized=normalized))

            kmer_profiles = torch.from_numpy(np.asarray(kmer_profiles)).to(torch.float)
            means, stds = self.mean_encoder(kmer_profiles), self.std_encoder(kmer_profiles)
            means = means.detach().numpy()
            stds = stds.detach().numpy()

        return means, stds

def loss_func(left_mean, left_std, right_mean, right_std, labels, name="vib", include_std=True):

    # Helper functions
    def log1mexp(p):
        """
        Compute `log(1 - exp(-x))` elementwise in a numerically stable way.
        """
        log2 = torch.log(torch.tensor(2.0, dtype=p.dtype, device=p.device))
        return torch.where(
            p < log2, torch.log(-torch.expm1(-p)), torch.log1p(-torch.exp(-p))
        )

    if name == "vib-nonstable":

        if include_std:
            k = 8
            epsilon = (2**0.5)*torch.randn(size=(k, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)

            p = torch.mean(
                torch.exp(-((left_mean - right_mean) + (left_std - right_std)*epsilon )**2), keepdim=False
            )
        else:
            p = torch.exp(-(left_mean - right_mean)**2)

        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')

    elif name == "vib":

        z_i, z_j = left_mean, right_mean  # shape (batch, d)

        if include_std:
            k = 8
            epsilon_i = torch.randn(size=(k, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)  # dim = (k, batch, d)
            epsilon_j = torch.randn(size=(k, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)  # dim = (k, batch, d)
            z_i = z_i + left_std*epsilon_i # shape (k, batch, d)
            z_j = z_j + right_std*epsilon_j  # shape (k, batch, d)
        
            distance = torch.sum((z_i - z_j)**2, dim=2)  # shape (k, batch)

            # Get log probabilities, \sum_k exp(-dist) / K (In fact, we don't need the normalization)
            log_p_yij_ri_rj = torch.logsumexp(-distance, dim=0) - torch.log(torch.tensor(k, dtype=torch.float))

        else:
            log_p_yij_ri_rj = -torch.sum((z_i - z_j)**2, dim=1)

        # Define logits, log(p / (1-p))
        logits = log_p_yij_ri_rj - log1mexp(-log_p_yij_ri_rj)

        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
    else:
        raise ValueError(f"Unknown loss function name: {name}")


def single_epoch(rank, model, loss_func, optimizer, training_loader, loss_name="bern", include_std=True):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epoch_loss = 0.
    for data in training_loader:
        left_kmer_profile, right_kmer_profile, labels = data

        # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
        optimizer.zero_grad()
        left_kmer_profile = left_kmer_profile.reshape(-1, left_kmer_profile.shape[-1]).to(rank)
        right_kmer_profile = right_kmer_profile.reshape(-1, right_kmer_profile.shape[-1]).to(rank)
        labels = labels.reshape(-1).to(rank)

        # Make predictions for the current epoch
        left_mean, left_std, right_mean, right_std = model(left_kmer_profile, right_kmer_profile)

        # Compute the loss and backpropagate
        batch_loss = loss_func(left_mean, left_std, right_mean, right_std, labels, name=loss_name, include_std=include_std)
        batch_loss.backward()

        # Update the model parameters
        optimizer.step()

        # Get the epoch loss for reporting
        epoch_loss += batch_loss.item()
        #del batch_loss, left_kmer_profile, right_kmer_profile, labels, left_output, right_output
        torch.cuda.empty_cache()

    return epoch_loss / len(training_loader)

def set_gradients(parameters, requires_grad):

    for param in parameters:
        param.requires_grad = requires_grad

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    torch.distributed.destroy_process_group()

def run(rank, world_size, model, training_dataset, batch_size, workers_num, learning_rate, epoch_num, loss_name="bern", model_save_path=None, loss_file_path=None, checkpoint=0, verbose=True):

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model.to(torch.device("cuda:0"))
    setup(rank, world_size)

    # Distributed Sampler: Ensures data is divided among GPUs using DistributedSampler.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        training_dataset, num_replicas=world_size, rank=rank
    )
    # Data Loader: Creates a DataLoader that iterates through batches of data with distributed sampling and batching.
    training_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers_num, pin_memory=True, sampler=train_sampler
    )
    # Model Initialization: Initializes ResNet-50 model (net) and distributes it across GPUs using DistributedDataParallel.
    print(f'Rank {rank}: Building model..')
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Get the model parameters and set the gradients to False
    mean_parameters = [param for name, param in model.named_parameters() if 'mean' in name]
    std_parameters = [param for name, param in model.named_parameters() if 'std' in name]

    # Define the optimizers
    mean_optimizer = torch.optim.Adam(mean_parameters, lr=learning_rate)
    all_optimizer = torch.optim.Adam(mean_parameters + std_parameters, lr=learning_rate)

    # We will update the mean parameters in the first half of the epochs
    include_std = False
    set_gradients(mean_parameters, True)
    set_gradients(std_parameters, False)
    optimizer = mean_optimizer

    # Define the loss curve writer
    if loss_file_path is not None:
        writer = SummaryWriter(loss_file_path)
    if verbose:
        print("Training has just started.")

    for epoch in range(epoch_num):
        if verbose:
            print(f"\t+ Epoch {epoch + 1}.")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        # For the second half of the epochs, we will update the mean and std parameters together
        if epoch >= epoch_num // 2:
            include_std = True
            set_gradients(mean_parameters + std_parameters, True)
            optimizer = all_optimizer

        avg_loss = single_epoch(
            rank, model, loss_func, optimizer, training_loader, loss_name, include_std=include_std
        )

        if verbose:
            print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

        if loss_file_path is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch + 1)
            writer.flush()

        if model_save_path is not None and checkpoint > 0 and (epoch + 1) % checkpoint == 0:

            # model_save_path contains the substring epoch=ID, so change the ID to the current epoch
            temp_model_save_path = re.sub('epoch.*_LR', f"epoch={epoch + 1}_LR", model_save_path)

            torch.save([{'k': model.get_k(), 'device': model.get_device()}, model.state_dict()], temp_model_save_path)
            if verbose:
                print(f"Model is saving.")
                print(f"\t- Target path: {temp_model_save_path}")

    writer.close()
    # Cleanup: Releases distributed training resources after completion.
    cleanup()

    if model_save_path is not None:
        # If the model is a DataParallel object, then save the model.module
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        torch.save([{'k': model.get_k(), 'dim': model.get_dim(), 'device': model.get_device()}, model.state_dict()], model_save_path)

        if verbose:
            print(f"Model is saving.")
            print(f"\t- Target path: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--input', type=str, help='Input sequence file')
    parser.add_argument('--k', type=int, default=2, help='k value')
    parser.add_argument('--dim', type=int, default=256, help='dimension value')
    parser.add_argument('--neg_sample_per_pos', type=int, default=1000, help='Negative sample ratio')
    parser.add_argument('--max_read_num', type=int, default=10000, help='Maximum number of reads to get from the file')
    parser.add_argument('--epoch', type=int, default=1000, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=0, help='Batch size (0: no batch)')
    parser.add_argument('--device', type=str, default="cpu", help='Device (cpu or cuda)')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    parser.add_argument('--workers_num', type=int, default=1, help='Number of workers for data loader')
    parser.add_argument('--loss_name', type=str, default="vib", help='Loss function (vib or vib-nonstable)')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--seed', type=int, default=26042024, help='Seed for random number generator')
    parser.add_argument('--checkpoint', type=int, default=0, help='Save the model for every checkpoint epoch')
    args = parser.parse_args()

    # Define the model
    model = VIBModel(
        k=args.k, dim=args.dim, device=torch.device(args.device), verbose=True, seed=args.seed
    )

    # Read the dataset
    training_dataset = PairDataset(
        file_path=args.input, transform_func=model.read2kmer_profile, neg_sample_per_pos=args.neg_sample_per_pos,
        max_read_num=args.max_read_num, seed=args.seed
    )
    # # Define the training data loader
    # training_loader = DataLoader(
    #     training_dataset, batch_size=args.batch_size if args.batch_size else len(training_dataset),
    #     shuffle=True, num_workers=args.workers_num
    # )

    # Run the model
    # run(model, training_dataset, args.batch_size, args.workers_num, args.lr, args.epoch, args.loss_name, args.output, args.output + ".loss", args.checkpoint, verbose=True)
    torch.multiprocessing.spawn(
        run, nprocs=args.world_size, join=True,
        args=(
            args.world_size, model, training_dataset, args.batch_size, args.workers_num, args.lr, args.epoch,
            args.loss_name, args.output, args.output + ".loss", args.checkpoint, True
        ),
    )
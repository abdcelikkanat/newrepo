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

from sympy.physics.quantum.gate import normalized
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import re
import os
import csv

# Helper functions
def set_seed(seed: int):
    """
    Set the seed value for reproducibility
    """
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)

def log1mexp(p: torch.Tensor):
    """
    Compute `log(1 - exp(-x))` elementwise in a numerically stable way.
    """
    log2 = torch.log(torch.tensor(2.0, dtype=p.dtype, device=p.device))
    return torch.where(
        p < log2, torch.log(-torch.expm1(-p)), torch.log1p(-torch.exp(-p))
    )

class KmerConverter:
    def __init__(self, k):

        # Define the letters, k-mer size, and the base complement
        self.__k = k
        self.__letters = ['A', 'C', 'G', 'T']
        self.__kmer2id = {''.join(kmer): i for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))}
        self.__kmers_num = len(self.__kmer2id)

    def get_k(self):
        return self.__k

    def get_kmers_num(self):
        return self.__kmers_num

    def seq2kmer(self, seq:str, normalized:bool=True):

        # Get the k-mer profile
        kmer2id = [self.__kmer2id[seq[i:i + self.__k]] for i in range(len(seq) - self.__k + 1)]
        kmer_profile = np.bincount(kmer2id, minlength=self.__kmers_num)

        if normalized:
            kmer_profile = kmer_profile / kmer_profile.sum()

        return kmer_profile


class PairDataset(Dataset):
    def __init__(self, file_path, k, neg_sample_per_pos=1000, max_seq_num=0, verbose=True, seed=0):
        """
        PairDataset constructor
        """
        # Set the parameters
        self.__kmers = None # A tensor of shape N x D, storing the k-mer profiles
        self.__indices = None # A tensor of shape L x 2, storing the indices of the positive and negative pairs
        self.__labels = None  # A tensor of shape L, storing the labels of the pairs
        self.__kc = KmerConverter(k)
        self.__neg_sample_per_pos = neg_sample_per_pos

        # Set the seed
        set_seed(seed)

        if verbose:
            print(f"+ Reading the data file.")
            print(f"\t- File path: {file_path}")
            init_time = time.time()

        # Get the number of lines
        num_of_lines = sum(1 for _ in open(file_path, 'r'))

        # If the max_read_num is set, then sample the line numbers to read
        if max_seq_num > 0:
            chosen_lines = random.sample(range(num_of_lines), max_seq_num)
            chosen_lines.sort()
        # Otherwise, read all the lines
        else:
            chosen_lines = list(range(num_of_lines))

        # Read the file
        left_kmer_profiles, right_kmer_profiles = [], []
        with open(file_path, 'r') as f:
            reader = list(csv.reader(f, delimiter=','))
        for chosen_line_idx in chosen_lines:
            line = reader[chosen_line_idx]
            left_seq, right_seq = line[0], line[1]
            left_kmer_profiles.append(self.__kc.seq2kmer(left_seq))
            right_kmer_profiles.append(self.__kc.seq2kmer(right_seq))

        # Combine the left and right k-mer profiles.
        # The first half of the profiles are the left ones and the second half are the right ones
        self.__kmers = torch.from_numpy(
            np.asarray(left_kmer_profiles + right_kmer_profiles)
        ).to(torch.float)

        if verbose:
            print(f"\t- Completed in {time.time() - init_time:.2f} seconds.")
            # print the time information in 2 decimal points
            print(f"\t- The input file contains: {num_of_lines} sequence pairs.")
            print(f"\t- For training, {max_seq_num} sequence pairs will be used.")

        # Construct the indices storing the positive and negative pairs
        if verbose:
            print(f"+ Constructing the indices for positive and negative sample pairs")
            init_time = time.time()

        # Sample 'pos_indices' positive pair indices
        pos_indices = torch.vstack((
            torch.arange(self.__kmers.shape[0]//2, dtype=torch.long),
            torch.arange(self.__kmers.shape[0]//2, dtype=torch.long)+self.__kmers.shape[0]//2
        ))
        # Sample 'neg_sample_per_pos' negative pair indices for each positive pair
        temp_weights = torch.ones((self.__kmers.shape[0],), dtype=torch.float)
        neg_indices = torch.vstack((
            torch.multinomial(temp_weights, pos_indices.shape[1]*self.__neg_sample_per_pos, replacement=True),
            torch.multinomial(temp_weights, pos_indices.shape[1]*self.__neg_sample_per_pos, replacement=True),
        ))
        # Concatenate the positive and negative indices
        self.__indices = torch.hstack((pos_indices, neg_indices))
        self.__labels = torch.hstack((
            torch.ones((pos_indices.shape[1],), dtype=torch.float),
            torch.zeros((neg_indices.shape[1],), dtype=torch.float))
        )

        if verbose:
            print(f"\t- Completed in {time.time() - init_time:.2f} seconds.")
            print(f"\t- Training dataset contains {pos_indices.shape[1]} positive pairs.")
            print(f"\t- Training dataset contains {neg_indices.shape[1]} negative pairs.")

    def __len__(self):
        """
        Return the number of pairs
        """
        return self.__indices.shape[1]

    def __getitem__(self, idx):

        return self.__kmers[self.__indices[0, idx]], self.__kmers[self.__indices[1, idx]], self.__labels[idx]


class VIBModel(torch.nn.Module):
    def __init__(self, k:int, out_dim:int=256, seed:int=0):
        """
        Initialize the VIB model
        """
        super(VIBModel, self).__init__()

        # Set the parameters
        self.__k = k
        self.__out_dim = out_dim
        # Define the k-mer converter object
        self.__kc = KmerConverter(self.__k)
        # Set the seed
        set_seed(seed)

        # Define the layers
        self.linear1_common = torch.nn.Linear(self.__kc.get_kmers_num(), 512, dtype=torch.float)
        self.batch1_common = torch.nn.BatchNorm1d(512, dtype=torch.float)
        self.activation1_common = torch.nn.Sigmoid()
        self.dropout1_common = torch.nn.Dropout(0.2)
        self.linear2_mean = torch.nn.Linear(512, self.__out_dim, dtype=torch.float)
        self.linear2_std = torch.nn.Linear(512, self.__out_dim, dtype=torch.float)

    def encoder(self, kmers: torch.Tensor):
        """
        For given kmers, return the mean and std
        """
        h = self.linear1_common(kmers)
        h = self.batch1_common(h)
        h = self.activation1_common(h)
        h = self.dropout1_common(h)
        mean = self.linear2_mean(h)
        std = torch.nn.functional.softplus(self.linear2_std(h)) + 1e-6

        return mean, std

    def forward(self, left_kmers: torch.Tensor, right_kmers:torch.Tensor):
        """
        Forward pass
        """
        left_mean, left_std = self.encoder(left_kmers)
        right_mean, right_std = self.encoder(right_kmers)

        return left_mean, left_std, right_mean, right_std

    def get_k(self):
        """
        Return the k value
        """
        return self.__k

    def get_out_dim(self):
        """
        Return the output dimension
        """
        return self.__out_dim

    def get_seed(self):
        """
        Return the seed value
        """
        return self.__seed

    def seq2emb(self, sequences:list, normalized:bool=True):
        """
        Get the embeddings for the given list of DNA sequences
        """

        kmers = torch.asarray([self.__kc.seq2kmer(seq, normalized=normalized) for seq in sequences], dtype=torch.float)

        with torch.no_grad():
            self.eval()

            if normalized:
                kmers = kmers / kmers.sum(dim=1, keepdim=True)

            means, stds = self.encoder(kmers)

        return means.detach().numpy(), stds.detach().numpy()

    def save(self, path:str):
        """
        Save the model
        """
        torch.save(
            [{'k': self.get_k(), 'out_dim': self.get_out_dim()}, self.state_dict()], path
        )


def loss_func(
        left_mean: torch.Tensor, left_std: torch.Tensor, right_mean: torch.Tensor, right_std: torch.Tensor,
        labels: torch.Tensor, loss_name:str = "vib", include_std:bool = True, samples_num = 8
):
    """
    Definition of the loss functions
    """

    if loss_name == "bern":

        p = torch.exp(-torch.norm(left_mean - right_mean, dim=1)**2)
        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')

    elif loss_name == "vib":

        if include_std:

            epsilon_left = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            epsilon_right = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            z_i = left_mean + left_std * epsilon_left  # shape (k, batch, d)
            z_j = right_mean + right_std * epsilon_right  # shape (k, batch, d)
            # Compute the distance
            distance = torch.sum((z_i - z_j) ** 2, dim=2)  # shape (k, batch)
            # Get log probabilities, \sum_k exp(-dist) / K (In fact, we don't need the normalization)
            log_p_yij_ri_rj = torch.logsumexp(-distance, dim=0) - torch.log(torch.tensor(samples_num, dtype=torch.float))

        else:

            log_p_yij_ri_rj = -torch.sum((left_mean - right_mean) ** 2, dim=1)

        # Define logits, log(p / (1-p))
        logits = log_p_yij_ri_rj - log1mexp(-log_p_yij_ri_rj)

        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='mean')

    elif loss_name == "non_stable_vib":

        if include_std:

            epsilon_left = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            epsilon_right = torch.randn((samples_num, left_mean.shape[0], left_mean.shape[1]), device=left_mean.device)
            z_left = left_mean + left_std * epsilon_left  # shape (samples, batch, d)
            z_right = right_mean + right_std * epsilon_right  # shape (samples, batch, d)

        else:

            z_left, z_right = left_mean, right_mean

        # Define the success probability
        p = torch.mean(torch.exp(-torch.sum((z_left - z_right) ** 2, dim=-1)), dim=0, keepdim=False)

        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')

    elif loss_name == "vib_without_sampling":
        """
        Computation of the E_p(z_i|r_i)p(z_j|r_j)[p(y_ij|z_i,z_j)] without applying any sampling
        """

        var_left, var_right = left_std**2, right_std**2
        #
        # var_left, var_right = var_left*0, var_right*0

        # Compute the terms log(1+2(s_i^2 + s_j^2))
        log1p_cov = torch.log1p(2*(var_left + var_right))
        # Compute the term m_i - m_j
        mean_squared_diff = (left_mean - right_mean)**2
        # Compute the log expectation
        log_expectation = -0.5*log1p_cov.sum(dim=1) - (mean_squared_diff / torch.exp(log1p_cov)).sum(dim=1)
        # Get the success probability
        p = torch.exp(log_expectation)

        return torch.nn.functional.binary_cross_entropy(p, labels, reduction='mean')


    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")

class Trainer:
    def __init__(
            self, model: torch.nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer,
            criterion, distributed: bool, rank: int, save_every: int = 0, output_file: str = None
    ):
        self.__model = DistributedDataParallel(model.to(rank), device_ids=[rank]) if distributed else model
        self.__data_loader = data_loader
        # self.__lr = lr
        self.__current_optimizer = optimizer #torch.optim.Adam(self.__model.parameters(), lr=self.__lr)
        self.__criterion = criterion
        self.__distributed = distributed
        self.__rank = rank
        self.__save_every = save_every
        self.__output_file = output_file

    def __train_batch(self, left_kmers: torch.Tensor, right_kmers: torch.Tensor, labels: torch.Tensor):
        # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
        self.__current_optimizer.zero_grad()

        # Make predictions for the current epoch
        left_mean, left_std, right_mean, right_std = self.__model(left_kmers, right_kmers)

        # Compute the loss and backpropagate
        batch_loss = self.__criterion(left_mean, left_std, right_mean, right_std, labels)
        batch_loss.backward()

        # Update the model parameters
        self.__current_optimizer.step()

        return batch_loss

    def __train_single_epoch(self, epoch: int):

        # Set the epoch for the sampler
        # TODO: Check if this is necessary
        if self.__distributed:
            self.__data_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.
        for data in self.__data_loader:

            left_kmer_profile, right_kmer_profile, labels = data
            left_kmer_profile = left_kmer_profile.reshape(-1, left_kmer_profile.shape[-1])
            right_kmer_profile = right_kmer_profile.reshape(-1, right_kmer_profile.shape[-1])
            labels = labels.reshape(-1)
            if self.__distributed:
                left_kmer_profile = left_kmer_profile.to(self.__rank)
                right_kmer_profile = right_kmer_profile.to(self.__rank)
                labels = labels.to(self.__rank)

            # Run the training for the current batch
            batch_loss = self.__train_batch(left_kmer_profile, right_kmer_profile, labels)

            # Get the epoch loss for reporting
            epoch_loss += batch_loss

        # Get the average epoch loss
        average_epoch_loss = epoch_loss.item() / len(self.__data_loader)


        return average_epoch_loss

    def start(self, epoch_num: int):

        # Switch the model to training mode
        self.__model.train()

        ## Start training
        print(f"+ Training started.")
        for current_epoch in range(epoch_num):

            # Run the training for the current epoch
            loss = self.__train_single_epoch(epoch=current_epoch)
            print(f"\t- Epoch: {current_epoch}/{epoch_num} GPU:{self.__rank} - Loss: {loss}")

            # Save the checkpoint if necessary
            if self.__save_every > 0 and (current_epoch + 1) % self.__save_every == 0:
                if self.__rank == 0:
                    # Get the folder path of the output file
                    output_folder = os.path.dirname(self.__output_file)
                    # Define the checkpoint path
                    checkpoint_path = os.path.join(output_folder, f"checkpoint_epoch_{current_epoch}.model")
                    if self.__distributed:
                        self.__model.module.save(checkpoint_path)
                    else:
                        self.__model.save(checkpoint_path)

            # To ensure that all processes have finished the current epoch
            if self.__distributed:
                torch.distributed.barrier()

        print(self.__output_file)
        if self.__distributed:
            self.__model.module.save(self.__output_file)
        else:
            self.__model.save(self.__output_file)

        print(f"\t- Completed.")



def load_training_objects(
        input_file_path: str, k: int, out_dim:int, lr:float, neg_sample_per_pos:int, max_seq_num:int, seed:int
):
    # Read the dataset
    training_dataset = PairDataset(
        file_path=input_file_path, k=k,
        neg_sample_per_pos=neg_sample_per_pos, max_seq_num=max_seq_num, seed=seed
    )
    # Define the model
    model = VIBModel(k=k, out_dim=out_dim, seed=seed)
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    return training_dataset, model, optimizer


def ddp_setup(rank:torch.device, world_size:int, distributed:bool):

    if distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        backend = 'nccl'
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        raise NotImplemented


def ddp_cleanup():
    # Wait for all processes to complete
    torch.distributed.barrier()
    # Release the distributed training resources
    torch.distributed.destroy_process_group()

def main_worker(
        rank, world_size, input_file_path: str, k: int, out_dim: int, lr:float, neg_sample_per_pos: int,
        max_seq_num: int, epoch_num:int, batch_size:int, save_every: int, loss_name: str, output_path: str,
        distributed:bool, seed: int, verbose:bool=True
):
    # Setup the distributed training
    ddp_setup(rank, world_size, distributed=distributed)

    run(
        rank, world_size, input_file_path, k, out_dim, lr, neg_sample_per_pos, max_seq_num, epoch_num, batch_size,
        save_every, loss_name, output_path, distributed, seed, verbose
    )

    # Cleanup: Releases distributed training resources after completion.
    ddp_cleanup()

def run(
        rank, world_size, input_file_path: str, k: int, out_dim: int, lr:float, neg_sample_per_pos: int,
        max_seq_num: int, epoch_num:int, batch_size:int, save_every: int, loss_name: str, output_path: str,
        distributed:bool, seed: int, verbose:bool=True
):
    # Load the training objects
    training_dataset, model, optimizer = load_training_objects(
        input_file_path=input_file_path, k=k, out_dim=out_dim, lr=lr,
        neg_sample_per_pos=neg_sample_per_pos, max_seq_num=max_seq_num, seed=seed
    )

    # Define the loss function
    criterion = partial(loss_func, loss_name=loss_name, include_std=False)

    if distributed:
        training_sampler = DistributedSampler(training_dataset, num_replicas=world_size, rank=rank)
    else:
        training_sampler = None

    # Define a DataLoader that iterates through batches of data with distributed sampling and batching.
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True if distributed is None else False,
        num_workers=world_size, pin_memory=True, sampler=training_sampler
    )

    # Initializes Trainer object with model, data_loader, optimizer, loss function, rank, and save_every.
    trainer = Trainer(
        model, training_loader, optimizer, criterion, distributed=distributed, rank=rank,
        save_every=save_every, output_file=output_path
    )
    trainer.start(epoch_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--input', type=str, help='Input sequence file', required=True)
    parser.add_argument('--output_path', type=str, help='Model file path to save', required=True)
    parser.add_argument('--k', type=int, default=4, help='k value')
    parser.add_argument('--out_dim', type=int, default=256, help='dimension value')
    parser.add_argument('--neg_sample_per_pos', type=int, default=1000, help='Negative sample ratio')
    parser.add_argument('--max_seq_num', type=int, default=10000, help='Maximum number of sequences to get from the file')
    parser.add_argument('--epoch', type=int, default=300, help='Epoch number')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size (0: no batch)')
    parser.add_argument('--save_every', type=int, default=0, help='Save checkpoint every x epochs (0 for no checkpoint)')
    parser.add_argument('--distributed', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    parser.add_argument('--loss_name', type=str, default="vib", help='Loss function (bern, vib, non_stable_vib and vib_without_sampling)')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generator')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"+ Information")
    print(f"\t- Number of GPUs: {world_size}")
    print(f"\t- Loss name: {args.loss_name}")

    arguments = (
        world_size, args.input, args.k, args.out_dim, args.lr, args.neg_sample_per_pos, args.max_seq_num,
        args.epoch, args.batch_size, args.save_every, args.loss_name, args.output_path, args.distributed, args.seed, True
    )
    if world_size > 1 or args.distributed:
        torch.multiprocessing.spawn(main_worker, nprocs=world_size, join=True, args=arguments)
    else:
        print(f"\t- Running on the CPU")
        run(*((0,) + arguments))
    print("Completed.")
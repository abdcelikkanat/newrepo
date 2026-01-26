import sys
import numpy as np
from evaluation.utils import filter_sequences

# Define the global parameters
MAX_SEQ_LEN = 20000
MIN_SEQ_LEN = 2500
MIN_ABUNDANCE_VALUE = 10

# Define the data file
data_file = sys.argv[1]

# Filter the sequences based on the minimum length and abundance
dna_sequences, labels_bin = filter_sequences(
    data_path=data_file, shorten=MAX_SEQ_LEN, min_len=MIN_SEQ_LEN, abundance=MIN_ABUNDANCE_VALUE,
)

print( len(set(labels_bin)) )

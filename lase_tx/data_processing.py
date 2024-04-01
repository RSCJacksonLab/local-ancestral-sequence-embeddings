'''
Processing of sequence data for masked langauge modelling.
'''

import numpy as np
import tensorflow as tf

from Bio import SeqIO
from numpy.typing import ArrayLike
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple

tokenizer = {
    'A': 1, 'S': 2, 'G': 3, 'T': 4, 'L': 5, 'P': 6, 'R': 7, 'I': 8, 'N': 9,
    'V': 10, 'D': 11, 'Y': 12, 'E': 13, 'F': 14, 'K': 15, 'Q': 16, 'H': 17,
    'W': 18, 'M': 19, 'C': 20
}

def prepare_seqs(
    fasta_path: Path=None,
    seq_ls: list=None,
    ) -> ArrayLike:
    '''
    Prepare sequences for MLM. Sequences are padded and tokenized.

    Arguments:
    ----------
    fasta_path : Path, default `None`
        Path to .fasta file containing sequences for MLM.
    seq_ls : list, default `None`
        List of sequences for MLM.

    Returns:
    --------
    padded_token_arr : ArrayLike
        NumPy array containing tokenized and padded protein sequences.
    '''
    # load provided fasta file
    if fasta_path is not None:
        fasta_parser = SeqIO.parse(fasta_path, "fasta")
        seq_ls = [str(seq.seq).upper().rstrip() for seq in fasta_parser]
    elif seq_ls is None:
        raise ValueError("Must provide .fasta file or sequence list.")
    
    # length for padding sequences
    max_seq_len = max([len(seq) for seq in seq_ls])

    # tokenize each sequence
    token_ls = []
    for seq in seq_ls:
        token_seq = [tokenizer[aa] for aa in seq]
        token_ls.append(token_seq)
    
    # pad tokenized sequences
    padded_token_arr = tf.keras.preprocessing.sequence.pad_sequences(
        token_ls, 
        max_seq_len
    )
    return padded_token_arr

def process_mask(
    token_arr: ArrayLike, 
    vocab_size: int, 
    mask_pr: float,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    '''
    Mask the a tokenized sequence array. 

    Arguments:
    ----------
    token_arr : ArrayLike
        Array of tokenized and padded protein sequences.
    vocab_size : int
        Size of the vocabulary used for tokenizing.
    mask_pr : float
        Probability of masking a site.

    Returns:
    --------
    masked_tokens : ArrayLike
        Masked tokenized sequence array.
    y_ohe : ArrayLike
        Array of OHE embeddings of the true amino acid at masked sites.
        OHE index corresponds to the tokenization.
    sample_weights : ArrayLike
        Array with binary weightings such that the loss function is 
        only impacted by masked sites.
    '''
    # determine sites to mask
    mask_idx = np.random.rand(*token_arr.shape) < mask_pr
    if np.any(token_arr == 0):
        mask_idx[token_arr == 0] = False

    # prepare labels
    labels = -1 * np.ones(token_arr.shape, dtype=int)
    labels[mask_idx] = token_arr[mask_idx]
    y_labels = np.copy(token_arr)
    y_ohe = tf.one_hot(y_labels, vocab_size).numpy()

    # prepare masked input
    masked_tokens = np.copy(token_arr)
    masked_tokens[mask_idx] = 21

    # prepare sample weights
    sample_weights = np.ones(labels.shape)
    sample_weights[labels == -1] = 0

    return masked_tokens, y_ohe, sample_weights.astype('float32')

def process_data(
    fasta_path: Path,
    mask_pr: float,
    batch_size: int,
    train_split: Tuple=(0.3, 0.5),
    vocab_size: int=20,
) -> Tuple:
    '''
    Convert fasta sequences into train, validation and test datasets.

    Arguments:
    ----------
    fasta_path : Path
        Path to .fasta file containing sequences for MLM.
    mask_pr : float
        Probability of masking a site.
    batch_size : int
        Batch size for training.
    train_split : Tuple, default `(0.3, 0.5)`
        Tuple containing two floats. The first being the proportion of
        data to exclude from training for validation and testing. The
        second being the proportion remaining to use for testing.
    vocab_size : int, default `20`
        Size of the vocabulary used for tokenizing.

    Returns:
    --------
    mlm_trn : Dset
        Train dataset for MLM.
    mlm_val : Dset
        Validation dataset for MLM.
    mlm_tst : Dset
        Test dataset for MLM.
    '''
    # prepare sequences
    token_arr = prepare_seqs(fasta_path=fasta_path)
    # make data splits
    trn_token_arr, tst_token_arr = train_test_split(
        token_arr, 
        test_size=train_split[0]
    )
    tst_token_arr, val_token_arr = train_test_split(
        tst_token_arr,
        test_size=train_split[1]
    )
    # make datasets
    mlm_trn = tf.data.Dataset.from_tensor_slices(
        process_mask(trn_token_arr, vocab_size, mask_pr)
    ).batch(batch_size).shuffle(buffer_size=trn_token_arr.shape[0])
    mlm_val = tf.data.Dataset.from_tensor_slices(
        process_mask(val_token_arr, vocab_size, mask_pr)
    ).batch(batch_size).shuffle(buffer_size=val_token_arr.shape[0])
    mlm_tst = tf.data.Dataset.from_tensor_slices(
        process_mask(tst_token_arr, vocab_size, mask_pr)
    ).batch(batch_size).shuffle(buffer_size=tst_token_arr.shape[0])
    return mlm_trn, mlm_val, mlm_tst
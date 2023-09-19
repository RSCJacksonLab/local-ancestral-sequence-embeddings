
from Bio import SeqIO
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

tokenizer = {
    'A': 1, 'S': 2, 'G': 3, 'T': 4, 'L': 5, 'P': 6, 'R': 7, 'I': 8, 'N': 9,
    'V': 10, 'D': 11, 'Y': 12, 'E': 13, 'F': 14, 'K': 15, 'Q': 16, 'H': 17,
    'W': 18, 'M': 19, 'C': 20
}

def prepare_fasta(fasta_path: Path):
    '''
    
    '''
    fasta_parser = SeqIO.parse(fasta_path, "fasta")
    seq_ls = [str(seq.seq).upper().rstrip() for seq in fasta_parser]
    token_ls= []
    max_seq_len = max([len(seq) for seq in seq_ls])
    for seq in seq_ls:
        token_seq = [tokenizer[aa] for aa in seq]
        token_ls.append(token_seq)
    padded_token_arr = tf.keras.preprocessing.sequence.pad_sequences(
        token_ls, 
        max_seq_len
    )
    return padded_token_arr

def process_mask(
        token_arr: np.ndarray, 
        vocab_size: int, 
        mask_pr: float,
    ):
    '''
    
    '''
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
        train_split: tuple=(0.3, 0.5),
        vocab_size: int=20,
    ):
    '''
    
    '''
    token_arr = prepare_fasta(fasta_path)
    trn_token_arr, tst_token_arr = train_test_split(
        token_arr, 
        test_size=train_split[0]
    )
    tst_token_arr, val_token_arr = train_test_split(
        tst_token_arr,
        test_size=train_split[1]
    )
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
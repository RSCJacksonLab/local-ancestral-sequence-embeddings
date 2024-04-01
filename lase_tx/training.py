'''
Training loop for LASE model.
'''

import numpy as np
import tensorflow as tf

from lase_tx.data_processing import process_data
from lase_tx.lase_model import Encoder
from pathlib import Path
from tqdm import tqdm

def build_encoder(
    n_layers: int, 
    hidden_dim: int, 
    num_heads: int, 
    max_seq_len: int, 
    dropout_pr: float, 
    vocab_size: int=20,
):
    '''
    Build the TensorFlow LASE encoder model.

    Arguments:
    ----------
    n_layers : int
        Number of transformer layers in the encoder.
    hidden_dim : int
        Number of hidden dimensions.
    num_heads : int
        Number of attention heads.
    dropout_pr : float 
        Dropout probability.
    max_seq_len : int
        Maximum sequence length.
    vocab_size : int
        Size of vocabulary. 
    
    Returns:
    --------
    encoder 
        Compiled TensorFlow encoder model.
    '''
    encoder = Encoder(
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout_pr=dropout_pr,
    )
    encoder.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.CategoricalAccuracy(),
        optimizer=tf.keras.optimizers.Adam()
    )
    encoder.build(input_shape=(None, max_seq_len))
    return encoder

def train_encoder(
    encoder, 
    fasta_path: Path, 
    weight_dir: Path, 
    init_seed: int=0,
    max_epochs: int=10,
    patience: int=2,
):
    '''
    Train a pre-built TensorFlow LASE encoder.

    Arguments:
    ----------
    encoder 
    fasta_path : Path
        Path to .fasta file containing sequences for MLM.
    weight_dir : Path
        Path to directory for saving weight files.
    init_seed : int, default `0`
        Seed for initializing random weights.
    max_epochs : int, default `10`
        Maximum number of epochs allowed for model training.
    patience : int, default `2`
        Number of epochs without improvement before early stopping
        commences.

    '''
    # random seed for weight initialization
    tf.random.set_seed(init_seed)
    tf.keras.utils.set_random_seed(init_seed)

    # accuracy metrics
    cat_acc = tf.keras.metrics.CategoricalAccuracy()
    val_cat_acc = tf.keras.metrics.CategoricalAccuracy()
    tst_cat_acc = tf.keras.metrics.CategoricalAccuracy()

    # loss function and optimizer initialization
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    # load encoder summary
    encoder.summary()

    # process MLM data
    mlm_trn, mlm_val, mlm_tst = process_data(
        fasta_path=fasta_path,
        mask_pr=0.05,
        batch_size=32,
        train_split=(0.3, 0.5),
        vocab_size=20
    )

    # start early stopping
    best_val_loss = float('inf')
    early_stopping = 0

    # training loop
    for epoch in range(max_epochs):
        print(f"Start of epoch {epoch}")

        # training
        for step, data in tqdm(enumerate(mlm_trn)):
            x_batch_trn, y_batch_trn, sample_weight = data
            with tf.GradientTape() as tape:
                logits = encoder(x_batch_trn,training=True)[0]
                loss = loss_fn(
                    y_batch_trn, 
                    logits, 
                    sample_weight=sample_weight
                )
            grads = tape.gradient(loss, encoder.trainable_weights)
            optimizer.apply_gradients(zip(grads, encoder.trainable_weights))
            cat_acc.update_state(
                y_batch_trn,
                logits,
                sample_weight=sample_weight
            )
            if step % 100 == 0:
                print(f"Training loss (for one batch) at step {step}: {loss}")
                print(
                    f"Training accuracy (for one batch) at step {step}: ",
                    f"{cat_acc.result().numpy()}"
                )

        # validation
        val_loss = []
        for step, data in enumerate(mlm_val):
            x_batch_val, y_batch_val, sample_weight = data
            logits = encoder(x_batch_val, training=False)[0]
            val_loss.append(
                loss_fn(y_batch_val, logits, sample_weight=sample_weight)
            )
            val_cat_acc.update_state(y_batch_val,logits,sample_weight=sample_weight)
        val_loss = np.mean(val_loss)
        print(
            f"Validation accuracy (for one epoch) at epoch {epoch}: ",
            f"{val_cat_acc.result().numpy()}"
        )
        
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping = 0
        else:
            early_stopping += 1
        if early_stopping >= patience:
            print(f"Early stopping at epoch: {epoch}")
            break            

    # testing
    for step, data in enumerate(mlm_tst):
        x_batch_tst, y_batch_tst, sample_weight = data
        logits = encoder(x_batch_tst, training=False)[0]
        loss = loss_fn(y_batch_tst, logits, sample_weight=sample_weight)
        tst_cat_acc.update_state(
            y_batch_tst,
            logits,
            sample_weight=sample_weight
        )
    print(
        f"Test accuracy at epoch {epoch}: {tst_cat_acc.result().numpy()}"
    )
    
    # save final model
    encoder.save_weights(weight_dir, save_format='tf')
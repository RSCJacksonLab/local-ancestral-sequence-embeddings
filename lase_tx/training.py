from lase_tx.lase_model import Encoder
from lase_tx.data_processing import process_data

import tensorflow as tf
from tqdm import tqdm
import numpy as np
from pathlib import Path

def build_encoder(n_layers, hidden_dim, num_heads, max_seq_len, dropout_pr, vocab_size=20):
    '''
    
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
        init_seed: int,
        max_epochs: int=10,
        patience: int=2,
    ):
    '''
    
    '''
    tf.random.set_seed(init_seed)
    tf.keras.utils.set_random_seed(init_seed)
    cat_acc = tf.keras.metrics.CategoricalAccuracy()
    val_cat_acc = tf.keras.metrics.CategoricalAccuracy()
    tst_cat_acc = tf.keras.metrics.CategoricalAccuracy()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    encoder.summary()
    mlm_trn, mlm_val, mlm_tst = process_data(
        fasta_path=fasta_path,
        mask_pr=0.05,
        batch_size=32,
        train_split=(0.3, 0.5),
        vocab_size=20
    )
    best_val_loss = float('inf')
    early_stopping = 0

    for epoch in range(max_epochs):
        print(f"Start of epoch {epoch}")

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping = 0
        else:
            early_stopping += 1
        if early_stopping >= patience:
            print(f"Early stopping at epoch: {epoch}")
            break            

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
    
    encoder.save_weights(weight_dir, save_format='tf')
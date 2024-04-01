'''
In silico evolution module.
'''

import json
import numpy as np
import os
import pandas as pd
import pickle as pkl
import tensorflow as tf
import time

from lase_tx import data_processing
from numpy.typing import ArrayLike
from pathlib import Path
from sklearn.utils import shuffle
from typing import Tuple

def load_mut_dict(
    json_path: str,
) -> dict:
    '''
    Load a mutation dictionary from a JSON file. Expected the dictionary
    to be in the form: {site_idx: [allowed amino acids]}.
    
    Arguments:
    ----------
    json_path : str
        Path to the JSON file.

    Returns:
    --------
    mut_dict : dict
        Mutation dictionary object.
    '''
    mut_dict = json.load(open(json_path, "r"))
    mut_dict = {int(key): mut_dict[key] for key in mut_dict.keys()}
    return mut_dict

def embed_seqs(
    seq_ls: list[str],
    model, 
    batch_size: int,
) -> Tuple[ArrayLike, float]:
    '''
    Embed sequences from a sequence list using a pre-loaded LASE model.

    Arguments:
    ----------
    seq_ls : list
        List containing sequences to embed. The list must be 
        pre-cleaned containing only canonical amino acids.
    model
        A pre-loaded LASE model.
    batch_size : int
        Batch size for embedding sequences using a pre-trained model.

    Returns:
    --------
    representation_arr : ArrayLike
        Representation array for the given sequences. Returns size of
        (n_sequences, embedding_dim)
    time_taken : float
        Time taken to embed all sequences.
    '''
    # prepare dataset
    tokens = data_processing.prepare_seqs(seq_ls=seq_ls)
    emb_dset = tf.data.Dataset.from_tensor_slices(tokens).batch(batch_size)

    # extract representations
    representation_ls = []
    start_time = time.time()
    for x in emb_dset:
        out = model(x)[1].numpy()
        representation_ls.append(out)
    representation_arr = np.concatenate(representation_ls)
    end_time = time.time()
    time_taken = end_time - start_time
    return representation_arr, time_taken

def predict_fitness(
    model, 
    representation_arr: ArrayLike,
) -> Tuple[ArrayLike, float]:
    '''
    Given a pre-trained Sklearn model, predict the fitness of sequences
    embedded in the corresponding embedding/representation scheme.

    Arguments:
    ----------
    model
        Pre-trained Sklearn model
    representation_arr : ArrayLike
        Sequences embedding in a NumPy array containing sequences
        representations.

    Returns:
    --------
    fitness_arr : ArrayLike
        Array containing predicted fitness values.
    time_taken : float
        Time taken to predict the fitness of all sequences. This does
        not include embedding time.
    '''
    start_time = time.time()
    fitness_arr = model.predict(representation_arr)
    end_time = time.time()
    time_taken = end_time - start_time
    return fitness_arr, time_taken

def mutate_seqs(
    seq_ls: list[str], 
    mutation_dict: dict[int, list[str]], 
    seq_hist: list[str]
) -> Tuple[list[str], list[str]]:
    '''
    Produce all possible mutations for each sequence in a list of
    sequences given a mutation dictionary containing all allowed 
    amino acids for each site. Log the history of each sequence
    (i.e. what mutations have been accumulated since the beggining)
    of in silico evolution. 

    Arguments:
    ----------
    seq_ls : list
        List containing sequences to mutate. The list must be 
        pre-cleaned containing only canonical amino acids.
    mutation_dict : dict
        Mutation dictionary object.
    seq_hist : list
        List containing the history of mutations accumulated for each 
        sequence.

    Returns:
    --------
    mut_set_seqs_ls : list
        List containing mutated sequences.
    mut_seq_hist_ls : list
        List containing a list for each sequence of the accumulated
        mutations obtained over evolution.
    '''
    seq_hist = list(seq_hist)
    mut_seq_hist = []
    mut_seqs = []
    # for each sequence
    for i, seq in enumerate(seq_ls):
        # for each site with variation
        for key in mutation_dict.keys():
            # for each possible amino acid
            for val in mutation_dict[key]:
                mut_seq = np.array(list(seq)) 
                # if amino acid is not already at site, mutate
                if mut_seq[key] != val:
                    mut_seq[key] = val 
                    mut_seq = "".join(mut_seq)
                    mut_seqs.append(mut_seq)
                    mut_seq_hist.append(seq_hist[i])
    # remove duplicates
    mut_set_seqs_ls = list(set(mut_seqs))
    mut_seq_hist_ls = []
    # find original sequence for tracking of mutation history
    for seq in mut_set_seqs_ls:
        idx = mut_seqs.index(seq)
        mut_seq_hist_ls.append(mut_seq_hist[idx])
    return mut_set_seqs_ls, mut_seq_hist_ls

def greedy_search(
    seq_ls: list, 
    seq_hist: list, 
    saved_seqs: list, 
    skl_model, 
    n: int, 
    model, 
    batch_size: int,
    mut_dict: dict, 
    max_seqs: int=None,
) -> Tuple[list, list, list, float, float, float, int]:
    '''
    Perform a greedy search over the predicted sequence space.

    Arguments:
    ----------
    seq_ls : list
        List containing start sequences for this iteration of 
        evolution.
    seq_hist : list
        List containing the accumulated mutations for each sequence.
    saved_seqs : list
        Previously assessed sequences - used to remove duplicates.
    skl_model : sklearn model
        Already loaded pre-trained sklearn model for predicting 
        fitness.
    n : int
        The number of mutations to keep for the top and randomized
        datapoints. i.e. for each iteration, keep the n best variants
        as well as n random variants.
    model
        Pre-trained LASE model for embedding the sequences in. 
    batch_size : int
        Batch size for extracting LASE for each sequence.
    mut_dict : dict
        Dictionary containing allowed mutations for each site.
    max_seqs : int, default `None`
        The maximum amount of sequences to assess for this iteration.

    Returns:
    --------
    final_seq_ls : list
        List containing sequences to move forward into the next round
        of mutations.
    final_pred_hist_ls : list
        List containing the mutation history of each sequence being
        moved forward.
    final_pred_ls : list
        List of predicted fitness values of each sequence being moved
        forward.
    other_time : float
        Time taken to perform one in silico iteration excluding the 
        time to embed sequences and make predictions.
    embedding_time : float
        Time taken to embed all sequences.
    prediction_time : float
        Time taken to predict the fitness of all sequences. This does
        not include embedding time.
    seqs_assessed : list
        A list of all sequences assessed including those pruned out.
    '''
    seq_ls, seq_hist = mutate_seqs(seq_ls, mut_dict, seq_hist)
    if max_seqs==None:
        max_seqs = len(seq_ls)
    seq_ls, seq_hist = zip(*[(seq_ls[i], seq_hist[i]) for i in range(max_seqs) if not seq_ls[i] in saved_seqs])
    seq_ls = list(seq_ls)
    seqs_assessed = len(seq_ls)
    start_time = time.time()
    esm_representations, embedding_time = embed_seqs(seq_ls, model, batch_size)
    predictions, prediction_time = predict_fitness(skl_model, esm_representations)
    end_time = time.time()
    upd_seq_hist = []
    for i in range(len(seq_hist)):
        old_seq_hist = seq_hist[i].copy()
        old_seq_hist.append(predictions[i])
        upd_seq_hist.append(old_seq_hist)
    df = pd.DataFrame({
        "Sequence": seq_ls,
        "Prediction": predictions,
        "Prediction_history": upd_seq_hist,
    })
    df = df.sort_values(by="Prediction", ascending=False)
    top_df = df[:n]
    df = df[n:]
    df = shuffle(df)
    rand_df = df[:n]
    final_df = pd.concat([top_df, rand_df])
    other_time = (end_time - start_time) - (embedding_time + prediction_time)

    final_seq_ls = final_df.Sequence.tolist()
    final_pred_hist_ls = final_df.Prediction_history.tolist()
    final_pred_ls = final_df.Prediction.tolist()

    return (
        final_seq_ls,
        final_pred_hist_ls,
        final_pred_ls,
        other_time,
        embedding_time, 
        prediction_time, 
        seqs_assessed,
    )

def in_silico(
    start_seqs: list, 
    model, 
    batch_size: int, 
    skl_model, 
    gens: int,
    seqs_per_gen_method: int,
    mut_dict: dict, 
    save_dir: Path,
) -> pd.DataFrame:
    '''
    Perform multiple generations of in silico evolution on a list of
    start sequences. Both writes and returns results.

    Arguments:
    ----------
    start_seqs : list
        Pool of pre-cleaned sequences to perform in silico on. Assumes
        sequences are the same length.
    model
        Pre-trained embedding model.
    batch_size : int
        Batch size for extracting sequence embeddings.
    skl_model 
        Pre-trained sklearn model for making predictions. 
    gens : int
        Number of in silico generations/iterations to run evolution
        over.
    seqs_per_gen_method : int
        Number of sequences to keep from both the top predicted
        sequences and random sequences for each round of evolution.
    mut_dict : dict
        Dictionary containing allowed mutations for each site.
    save_dir : Path
        Path to directory for saving in silico output.

    Returns:
    --------
    final_df : pd.DataFrame
        DataFrame containing all key in silico results.
    '''
    current_seqs = start_seqs
    seq_hist = [[None]] * len(start_seqs)
    all_seqs = start_seqs
    timing_dict = {
        "gen": [], "n_seqs": [], 
        "other_time": [], "emb_time": [], "pred_time": [], 
        "top_pred": []
    }
    for gen in range(gens):
        # run a single iteration of in silico
        iter_res = greedy_search(
            current_seqs,
            seq_hist,
            all_seqs,
            skl_model,
            seqs_per_gen_method,
            model,
            batch_size,
            mut_dict
        )
        # extract iteration results
        current_seqs = iter_res[0]
        seq_hist = iter_res[1]
        preds = iter_res[2]
        other_time = iter_res[3]
        emb_time = iter_res[4]
        pred_time = iter_res[5]
        n_seq = iter_res[6]

        # update in silico results
        seq_hist = [list(seq_tup) for seq_tup in seq_hist]
        all_seqs += current_seqs
        timing_dict["gen"].append(gen)
        timing_dict["n_seqs"].append(n_seq)
        timing_dict["other_time"].append(other_time)
        timing_dict["emb_time"].append(emb_time)
        timing_dict["pred_time"].append(pred_time)
        timing_dict["top_pred"].append(np.max(preds))
        gen_res = {
            "sequences": current_seqs,
            "predictions": preds,
            "history": seq_hist
        }
        pd.DataFrame(gen_res).to_csv(os.path.join(save_dir, f"LASE_insilico_predictions_{gen}.csv"))
        pd.DataFrame(timing_dict).to_csv(os.path.join(save_dir, f"LASE_insilico_timing_{gen}.csv"))
    
    # save final results
    final_df = pd.DataFrame(timing_dict)
    final_df.to_csv(os.path.join(save_dir, f"LASE_insilico.csv"))
    return final_df

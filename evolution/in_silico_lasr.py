import tensorflow as tf
import numpy as np
import json
import pickle as pkl
import time
import pandas as pd
from lase_tx import (
    data_processing,
    lase_model
)
from sklearn.utils import shuffle

def load_mut_dict(json_path):
    mut_dict = json.load(open(json_path, "r"))
    mut_dict = {int(key): mut_dict[key] for key in mut_dict.keys()}
    return mut_dict

def embed_seqs(seq_ls, model, batch_size):
    '''
    
    '''
    # prepare dataset
    tokens = data_processing.prepare_seqs(seq_ls)
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

def predict_fitness(model, representation_arr):
    '''
    
    '''
    start_time = time.time()
    fitness_arr = model.predict(representation_arr)
    end_time = time.time()
    time_taken = end_time - start_time
    return fitness_arr, time_taken

def mutate_seqs(seq_ls, mutation_dict, seq_hist):
    seq_hist = list(seq_hist)
    mut_seq_hist = []
    mut_seqs = []
    for i, seq in enumerate(seq_ls):
        for key in mutation_dict.keys():
            for val in mutation_dict[key]:
                mut_seq = np.array(list(seq)) 
                if mut_seq[key] != val:
                    mut_seq[key] = val 
                    mut_seq = "".join(mut_seq)
                    mut_seqs.append(mut_seq)
                    mut_seq_hist.append(seq_hist[i])
    mut_set_seqs_ls = list(set(mut_seqs))
    mut_seq_hist_ls = []
    for seq in mut_set_seqs_ls:
        idx = mut_seqs.index(seq)
        mut_seq_hist_ls.append(mut_seq_hist[idx])
    return mut_set_seqs_ls, mut_seq_hist_ls

def greedy_search(seq_ls, seq_hist, saved_seqs, skl_model, n, model, batch_size, mut_dict, max_seqs=None):
    '''
    
    '''
    seq_ls, seq_hist = mutate_seqs(seq_ls, mut_dict, seq_hist)
    if max_seqs==None:
        max_seqs=len(seq_ls)
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
    return final_df.Sequence.tolist(), final_df.Prediction_history.tolist(), final_df.Prediction.tolist(), other_time, embedding_time, prediction_time, seqs_assessed


def in_silico(start_seqs, model, batch_size, skl_model, gens, mut_dict):
    current_seqs = start_seqs
    seq_hist = [[None]] * len(start_seqs)
    all_seqs = start_seqs
    timing_dict = {"gen": [], "n_seqs": [], "other_time": [], "emb_time": [], "pred_time": [], "top_pred": []}
    for gen in range(gens):
        current_seqs, unprocced_seq_hist, predictions, other_time, emb_time, pred_time, n_seq = greedy_search(current_seqs, seq_hist, all_seqs, skl_model, 250, model, batch_size, mut_dict)
        seq_hist = [list(seq_tup) for seq_tup in unprocced_seq_hist]
        all_seqs += current_seqs
        timing_dict["gen"].append(gen)
        timing_dict["n_seqs"].append(n_seq)
        timing_dict["other_time"].append(other_time)
        timing_dict["emb_time"].append(emb_time)
        timing_dict["pred_time"].append(pred_time)
        timing_dict["top_pred"].append(np.max(predictions))
        gen_res = {
            "sequences": current_seqs,
            "predictions": predictions,
            "history": seq_hist
        }
        pd.DataFrame(gen_res).to_csv(f"/home/dana/Documents/2023_PTE_LASR/2022_PTE_LASR/pte-lasr_storage_v2/2023_PTE_LASR/in_silico/lasr_data_wt_v2/LASR_insilico_predictions_{gen}.csv")
        pd.DataFrame(timing_dict).to_csv(f"/home/dana/Documents/2023_PTE_LASR/2022_PTE_LASR/pte-lasr_storage_v2/2023_PTE_LASR/in_silico/lasr_data_wt_v2/LASR_insilico_timing_{gen}.csv")
    pd.DataFrame(timing_dict).to_csv("/home/dana/Documents/2023_PTE_LASR/2022_PTE_LASR/pte-lasr_storage_v2/2023_PTE_LASR/in_silico/lasr_data_wt_v2/LASR_insilico_timing.csv")
        
device = "cuda"
sklearn_path = "/home/dana/Documents/2023_PTE_LASR/2022_PTE_LASR/pte-lasr_storage_v2/2023_PTE_LASR/in_silico/rf_lasr005s4_seed4.sav"
start_seqs = ["MQTRRVVLKSAAAAGTLLGGLAGCASVAGSIGTGDRINTVRGPITISEAGFTLTHEHICGSSAGFLRAWPEFFGSRKALAEKAVRGLRRARAAGVRTIVDVSTFDLGRDVSLLAEVSRAADVHIVAATGLWLDPPLSMRLRSVEELTQFFLREIQYGIEDTGIRAGIIKVATTGKVTPFQELVLRAAARASLATGVPVTTHTAASQRGGEQQAAIFESEGLSPSRVCIGHSDDTDDLSYLTALAARGYLIGLDHIPHSAIGLEDNASASALLGIRSWQTRALLIKALIDQGYMKQILVSNDWLFGFSSYVTNIMDVMDSVNPDGMAFIPLRVIPFLREKGIPQETLAGITVTNPARFLSPTLRAS"]

model = TF2_BERTv3.build_Encoder("/home/dana/Documents/2023_PTE_LASR/2022_PTE_LASR/pte-lasr_storage_v2/2023_PTE_LASR/in_silico/models/4/All_combined_processed_ancs_NR100.fasta")
model.load_weights("/home/dana/Documents/2023_PTE_LASR/2022_PTE_LASR/pte-lasr_storage_v2/2023_PTE_LASR/in_silico/models/4/weights/PTE_015")

# load sklearn model
reg_model = pkl.load(open(sklearn_path, "rb"))


mut_dict = load_mut_dict("/home/dana/Documents/2023_PTE_LASR/2022_PTE_LASR/pte-lasr_storage_v2/2023_PTE_LASR/in_silico/mutation_dict.json")

csv_res = in_silico(start_seqs, model, 512, reg_model, 25, mut_dict)
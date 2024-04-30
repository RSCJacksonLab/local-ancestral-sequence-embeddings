'''
LoRA fine-tuning of ESM with HuggingFace
'''

from tomlkit import value
import evaluate
import numpy as np
import os
import torch

from Bio import SeqIO
from datasets import Dataset
from pathlib import Path
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (
    DataCollatorForLanguageModeling,
    EsmForMaskedLM,
    EsmModel,
    EsmTokenizer,
    Trainer,
    TrainingArguments
)

def accuracy_fn(lm_output):
    '''
    Calculate accuracy given ESM predictions.
    '''
    metric = evaluate.load("accuracy")

    logits, inputs = lm_output
    preds = np.argmax(logits, axis=-1)
    targets = inputs[0]
    mask = inputs[2]

    # only consider masked amino acids
    mask = mask != -100
    targets = targets[mask]
    preds = preds[mask]

    return metric.compute(predictions=preds, references=targets)


def finetune_esm(
    model_name: str,
    seq_path: Path,
    output_dir: Path,
):
    '''
    Fine tune an ESM model on a provided sequence corpora. 

    Arguments:
    ----------
    model_name : str
        Name of ESM model to train.
    seq_path : Path
        Path to .fasta file containing sequences to train ESM on.
    output_dir : Path
        Path for saving results and final model to.
    '''
    # load esm model
    model = EsmForMaskedLM.from_pretrained(model_name)

    # add LoRA
    config = LoraConfig(
        r=4,
        alpha=8,
        target_modules=[
            "query",
            "key",
            "value"
        ],
        inference_mode=False,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)

    # ensure LM head is not frozen
    for param in model.lm_head.parameters():
        param.requires_grad = True
    
    # load tokenizer
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    # load data
    seq_ls = [str(s.seq) for s in SeqIO.parse(seq_path, "fasta")]
    seq_ls = [s for s in seq_ls if len(s) < 600]
    
    # train/test dataset
    trn_seq_ls, tst_seq_ls = train_test_split(seq_ls, test_size=0.1)

    trn_inputs = tokenizer(trn_seq_ls, padding=True)
    trn_dset = Dataset.from_dict(trn_inputs)

    tst_inputs = tokenizer(tst_seq_ls, padding=True)
    tst_dset = Dataset.from_dict(tst_inputs)

    # data collator for MLM - default mask pr = 0.15
    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=True
    )

    # training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        evaluation_strategy="steps",
        num_train_epochs=2,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        logging_steps=50,
        label_names=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=trn_dset,
        eval_dataset=tst_dset,
        compute_metrics=accuracy_fn,
    )
    
    # evaluate model at step==0
    trainer.evaluate()

    # train model
    trainer.train()

    # save final model
    torch.save(
        model.state_dict(),
        os.path.join(output_dir, "state_dict.pt")
    )

def get_reps(
    model_name: str,
    seq_path: Path,
    state_dict_path: Path,
):
    '''
    
    '''
    # process sequences
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    seq_ls = [str(s.seq) for s in SeqIO.parse(seq_path, "fasta")]
    inputs = tokenizer(seq_ls, padding=True, return_tensors="pt")
    dset = Dataset.from_dict(inputs)

    # prepare model
    model = EsmModel.from_pretrained(
        model_name
    )
    config = LoraConfig(
        r=4,
        alpha=8,
        target_modules=[
            "query",
            "key",
            "value"
        ],
        inference_mode=True,
        lora_dropout=0,
        bias="none",
    )
    model = get_peft_model(model, config)

    state_dict = torch.load(state_dict_path)
    state_dict = {
        "base_model.model." + key: value
        for key, value in state_dict.items()
    }

    model.load_state_dict(d=state_dict)

    
# Learned ancestral sequence embeddings

Learned ancestral sequence embeddings (LASE) are protein embeddings derived from training a light-weight language model on ancestral protein sequence data.

This repository contains the script used for training a transformer for LASE. It contains raw script for training the language model, training scikit learn regressors on saved representations and evaluating the ruggedness of landscapes resulting from these representations. 

## Python environment
All scripts (with exception of LoRA fine-tuning) were run using Python 3.8.17 and packages listed in requirements.txt. 

Code for LoRA fine-tuning of ESM was run using Python 3.11.4 and packages listed in lora_requirements.txt.

## File overview

### Data - `data`
Contains the mutation dictionary used for evolution as well as the regressor dataset. Data used in ancestral sequence reconstruction,for PTE and His3p, including extant sequences, ancestral sequences and phylogenetic trees are stored in the directory `phylogenetic_data`.

### Evolution - `evolution`
Contains Python modules for timed in silico evolution using LASE. 

### Protein Language modelling - `lase_tx`
Contains code required for processing of data for model training/representation extraction, the transformer model used, and for training the model. 

### Regressor top models - `sklearn_regressors`
Contains modules for training the regressors (SklearnRegressorOptimisation.py), a script for executing a full run-through of the training process (sklearn_Regressor_optimisation.py) as well as bash script used to run this workflow in parallel for different top models. 

### Ruggedness - `spectral_decomposition`
Contains code for determine the global and local dirichlet energy as well as eigenvalues, eigenvectors and Graph Fourier Transform.

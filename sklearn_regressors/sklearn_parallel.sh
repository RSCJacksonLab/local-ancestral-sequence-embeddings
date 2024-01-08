#!/bin/bash

rep=$1
save_dir=$2
label_path=$3

source ../../PyEnvs/lase/bin/activate

rep_name=$(basename $rep)
for i in 0 1 2 3 4
do
    mkdir $save_dir/
    mkdir $save_dir/$i
    mkdir $save_dir/$i/gp
    mkdir $save_dir/$i/rf

    parallel ::: "python3 ./sklearn_regressors/sklearn_regressor_optimisation.py \
        --X $rep \
        --y $label_path \
        --model_name gp \
        --enc_name $rep_name \
        --save_dir $save_dir/$i/gp \
        --holdout_seed $i" "python3 ./sklearn_regressors/sklearn_regressor_optimisation.py \
        --X $rep \
        --y $label_path \
        --model_name rf \
        --enc_name $rep_name \
        --save_dir $save_dir/$i/rf \
        --holdout_seed $i" 
done
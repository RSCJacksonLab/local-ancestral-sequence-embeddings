#!/bin/bash

rep_dir=$1
save_dir=$2
label_path=$3

source ../../../environments/pytorch-env/bin/activate

for rep in $rep_dir/*
do
    rep_name=$(basename $rep)
    for i in 0 1 2 3 4
    do
        mkdir $save_dir/$rep_name
        mkdir $save_dir/$rep_name/$i
        mkdir $save_dir/$rep_name/$i/gp
        mkdir $save_dir/$rep_name/$i/svm

        parallel ::: "python3 ./sklearn_regression_script.py \
            --X $rep_dir/$(basename $rep) \
            --y $label_path \
            --model_name gp \
            --enc_name $rep_name \
            --save_dir $save_dir/$rep_name/$i/gp \
            --random_seed $i" "python3 ./sklearn_regression_script.py \
            --X $rep_dir/$(basename $rep) \
            --y $label_path \
            --model_name svm \
            --enc_name $rep_name \
            --save_dir $save_dir/$rep_name/$i/svm \
            --random_seed $i"
    done
done
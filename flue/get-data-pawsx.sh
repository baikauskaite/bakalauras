#!/usr/bin/env bash
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

set -e

# data paths
BASE_DIR=/home/viktorija/bakalaurinis/flue
DATA_DIR=$BASE_DIR/data/pawsx
DATA_RAW=$DATA_DIR/raw
DATA_PROC=$DATA_RAW/processed
f=x-final

URLPATH=https://storage.googleapis.com/paws/pawsx/"$f.tar.gz"

# mkdir -p $DATA_RAW

# # Download data
# wget -c $URLPATH -P $DATA_RAW

# # unzip data
# tar -zxvf $DATA_RAW/"$f.tar.gz" --directory $DATA_RAW

# # Preprocess data
# python flue/extract_pawsx.py --indir "${DATA_RAW}/x-final" \
#                              --outdir $DATA_PROC \
#                              --use_hugging_face

model_type=camembert
model_fname=camembert-base

task_name=MRPC
batch_size=8
lr=5e-6
epochs=30

# experiment name
exp_name="pawsx_hf_"$model_fname
exp_id="lr_"$lr

model_name_or_path=$model_fname
output_dir=$BASE_DIR/experiments/$exp_name/$exp_id

save_steps=50000

python ./flue/run_glue.py \
                                        --train_file "${DATA_PROC}\train.tsv" \
                                        --test_file "${DATA_PROC}\test.tsv" \
                                        --model_name_or_path $model_name_or_path \
                                        --task_name $task_name \
                                        --output_dir $output_dir \
                                        --max_seq_length 512 \
                                        --do_train \
                                        --do_eval \
                                        --learning_rate $lr \
                                        --num_train_epochs $epochs \
                                        --save_steps $save_steps \
                                        |& tee output.log

# python ./flue/run_glue.py \
#                                         --train_file "${DATA_PROC}\train.tsv" \
#                                         --test_file "${DATA_PROC}\test.tsv" \
#                                         --model_name_or_path $model_name_or_path \
#                                         --task_name $task_name \
#                                         --output_dir $output_dir \
#                                         --max_seq_length 512 \
#                                         --do_train \
#                                         --do_eval \
#                                         --learning_rate $lr \
#                                         --num_train_epochs $epochs \
#                                         --save_steps $save_steps \
#                                         --fp16 \
#                                         --fp16_opt_level O1 \
#                                         |& tee output.log
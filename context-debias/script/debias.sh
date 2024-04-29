#!/bin/bash

set -e

MODEL_VERSION=$1

if [ -z "$MODEL_VERSION" ]; then
    echo "Provide model version ('camembert-base' or 'uklfr/gottbert-base') or path to model"
    exit 1
fi

if [[ $MODEL_VERSION == *"/"* && -d $MODEL_VERSION ]]; then
    MODEL_TYPE=$(basename $MODEL_VERSION)
else
    if [[ "$MODEL_VERSION" == */* ]]; then
        MODEL_TYPE="${MODEL_VERSION##*/}"
    else
        MODEL_TYPE=$MODEL_VERSION
    fi
fi

gpu=0
debias_layer=all # first last all
loss_target=token # token sentence
dev_data_size=1000
seed=42
alpha=0.2
beta=0.8

BASE_DIR="/home/viktorija/bakalaurinis/context-debias"

TRAIN_DATA="${BASE_DIR}/preprocess/$seed/$MODEL_TYPE/data.bin"
OUTPUT_DIR="${BASE_DIR}/../models/${MODEL_TYPE}-debiased"

rm -r $OUTPUT_DIR

echo $model_type $seed

CUDA_VISIBLE_DEVICES=$gpu python ../src/run_debias_mlm.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=$MODEL_VERSION \
    --do_train \
    --data_file=$TRAIN_DATA \
    --do_eval \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --num_train_epochs 3 \
    --block_size 128 \
    --loss_target $loss_target \
    --debias_layer $debias_layer \
    --seed $seed \
    --evaluate_during_training \
    --weighted_loss $alpha $beta \
    --dev_data_size $dev_data_size \
    --square_loss \
    --line_by_line \
    --mlm

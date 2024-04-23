#!/bin/bash

set -e

MODEL_VERSION=$1

if [ -z "$MODEL_VERSION" ]; then
    echo "Provide model version (for ex. camembert-base) or path to model"
    exit 1
fi

if [[ $MODEL_VERSION == *"/"* ]]; then
    MODEL_TYPE=$(basename $MODEL_VERSION)
else
    MODEL_TYPE=$MODEL_VERSION
fi

# data paths
BASE_DIR=/home/viktorija/bakalaurinis/pawsx
DATA_DIR=$BASE_DIR/data/pawsx
DATA_RAW=$DATA_DIR/raw
DATA_PROC=$DATA_RAW/processed
f=x-final

URLPATH=https://storage.googleapis.com/paws/pawsx/"$f.tar.gz"

mkdir -p $DATA_RAW

# Download data
wget -c $URLPATH -P $DATA_RAW

# unzip data
tar -zxvf $DATA_RAW/"$f.tar.gz" --directory $DATA_RAW

# Preprocess data
python $BASE_DIR/extract_pawsx.py   --indir "${DATA_RAW}/x-final" \
                                    --outdir $DATA_PROC \
                                    --use_hugging_face

task_name=MRPC
batch_size=8
lr=2e-5
epochs=4

OUTPUT_DIR="${BASE_DIR}/../models/${MODEL_TYPE}-finetuned-pawsx"

save_steps=50000
					# --per_device_train_batch_size $batch_size \
                                        # --do_train \
                                        # --num_train_epochs $epochs \

python $BASE_DIR/run_glue.py \
                                        --train_file "${DATA_PROC}\train.tsv" \
                                        --test_file "${DATA_PROC}\test.tsv" \
                                        --model_name_or_path $MODEL_VERSION \
                                        --task_name $task_name \
                                        --output_dir $OUTPUT_DIR \
                                        --max_seq_length 512 \
					--per_device_eval_batch_size $batch_size \
					--use_cpu \
                                        --do_eval \
                                        --learning_rate $lr \
                                        --save_steps $save_steps \
                                        |& tee output.log

#!/bin/bash

set -e

MODEL_VERSION=$1
LANGUAGE=$2

if [ -z "$MODEL_VERSION" ]; then
    echo "Provide model version ('camembert-base' or 'uklfr/gottbert-base') or path to model"
    exit 1
fi

if [ -z "$LANGUAGE" ]; then
    echo "Provide language ('french' or 'german')"
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

if [ "$language" = "german" ]; then
    country_code="de"
elif [ "$language" = "french" ]; then
    country_code="fr"
else
    echo "Unsupported language"
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
python $BASE_DIR/extract_pawsx.py   --indir "${DATA_RAW}/x-final/${country_code}" \
                                    --outdir $DATA_PROC/${LANGUAGE} \
                                    --use_hugging_face 

task_name=MRPC
batch_size=8
lr=5e-6
epochs=4

OUTPUT_DIR="${BASE_DIR}/../models/${MODEL_TYPE}-finetuned-pawsx"

save_steps=50000

python $BASE_DIR/run_glue.py \
                                        --train_file "${DATA_PROC}/${LANGUAGE}/train.tsv" \
                                        --test_file "${DATA_PROC}/${LANGUAGE}/test.tsv" \
                                        --model_name_or_path $MODEL_VERSION \
                                        --task_name $task_name \
                                        --output_dir $OUTPUT_DIR \
                                        --max_seq_length 512 \
                                        --do_train \
                                        --do_eval \
                                        --learning_rate $lr \
                                        --num_train_epochs $epochs \
                                        --fp16 \
                                        --fp16_opt_level O1 \
                                        --save_steps $save_steps \
                                        |& tee output.log

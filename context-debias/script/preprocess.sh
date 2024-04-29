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

seed=42
block_size=128
OUTPUT_DIR=../preprocess/$seed/$MODEL_TYPE

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python -u ../src/preprocess.py --input ../data/${LANGUAGE}/news-commentary-v18.${LANGUAGE} \
                        --stereotypes ../data/${LANGUAGE}/stereotypes.txt \
                        --attributes ../data/${LANGUAGE}/female.txt,../data/${LANGUAGE}/male.txt \
                        --output $OUTPUT_DIR \
                        --seed $seed \
                        --block_size $block_size \
                        --model_type $MODEL_TYPE \
                        --model_name_or_path $MODEL_VERSION


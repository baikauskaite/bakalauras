#!/bin/bash

set -e

MODEL_VERSION=$1
LANGUAGE=$2
TESTS=weat0,weat0b,weat1,weat2,weat3,weat3b,weat4,weat4b,weat5,weat5b,weat6

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

BASE_DIR="/home/viktorija/bakalaurinis/log-probability-bias"

OUTPUT_DIR="${BASE_DIR}/results/${MODEL_TYPE}"
OUT_FILE_SCORES="${OUTPUT_DIR}/${MODEL_TYPE}_scores.tsv"
OUT_FILE_SIGNIFICANCE="${OUTPUT_DIR}/${MODEL_TYPE}_significance.txt"

mkdir -p "$OUTPUT_DIR"

python $BASE_DIR/scripts/log_probability_bias_scores.py \
    --model ${MODEL_TYPE} \
    --model_version ${MODEL_VERSION} \
    --data_dir "${BASE_DIR}/tests/${LANGUAGE}" \
    --tests ${TESTS} \
    --output_dir ${OUTPUT_DIR} 

# For each file in the output directory, calculate statistical significance
for file in ${OUTPUT_DIR}/*-scores.tsv; do
    testname=$(basename $file)
    testname=${testname%%-scores.tsv}
    python $BASE_DIR/scripts/statistical_significance.py "${file}" > "${OUTPUT_DIR}/${testname}-significance.txt"
done

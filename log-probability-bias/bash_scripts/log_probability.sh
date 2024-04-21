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

MODEL_NAME="camembert"
LANGUAGE="french"
TESTS="weat0"

BASE_DIR="/home/viktorija/bakalaurinis/log-probability-bias"

OUTPUT_DIR="${BASE_DIR}/results/${MODEL_TYPE}"
OUT_FILE_SCORES="${OUTPUT_DIR}/${MODEL_NAME}_scores.tsv"
OUT_FILE_SIGNIFICANCE="${OUTPUT_DIR}/${MODEL_NAME}_significance.txt"

mkdir -p "$OUTPUT_DIR"

python $BASE_DIR/scripts/log_probability_bias_scores.py \
    --model ${MODEL_NAME} \
    --model_version ${MODEL_VERSION} \
    --demographic 'GEND' \
    --data_dir "${BASE_DIR}/tests/${LANGUAGE}" \
    --tests ${TESTS} \
    --out_file ${OUT_FILE_SCORES} 
 
python $BASE_DIR/scripts/statistical_significance.py ${OUT_FILE_SCORES} > ${OUT_FILE_SIGNIFICANCE}


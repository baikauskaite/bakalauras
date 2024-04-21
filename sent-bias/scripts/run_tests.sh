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

# TESTS=sent-weat1,sent-weat2,sent-weat3b,sent-weat5b,sent-weat6b,sent-weat7,sent-weat8,sent-weat9
TESTS=sent-weat0

LANGUAGE=french
SEED=42
BASE_DIR="/home/viktorija/bakalaurinis/sent-bias"
OUTPUT_DIR="${BASE_DIR}/results/${MODEL_TYPE}"
DATA_DIR="${BASE_DIR}/tests/${LANGUAGE}"
LOCAL_DIR="${BASE_DIR}/data"

mkdir "$OUTPUT_DIR"

python $BASE_DIR/scripts/main.py \
    --log_file ${OUTPUT_DIR}/log.log \
    -t ${TESTS} -m camembert \
    --camembert_version ${MODEL_VERSION} \
    --exp_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR} \
    -s ${SEED} \
    --ignore_cached_encs


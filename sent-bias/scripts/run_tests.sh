#!/bin/bash
mamba activate sent-bias
set -e

echo 'Note: this script should be called from the root of the repository' >&2

# TESTS=sent-weat1,sent-weat2,sent-weat3b,sent-weat5b,sent-weat6b,sent-weat7,sent-weat8,sent-weat9
TESTS=updated
TYPE=pre

LANGUAGE=french
SEED=1111
formatted_date=$(date "+%m.%d-%H.%M")
BASE_DIR="/home/viktorija/bakalaurinis/sent-bias"
OUTPUT_DIR="${BASE_DIR}/results/${TYPE}-${formatted_date}"
DATA_DIR="${BASE_DIR}/tests/${LANGUAGE}"
LOCAL_DIR="${BASE_DIR}/data"

mkdir "$OUTPUT_DIR"

# CamemBERT
# before debiasing
python sentbias/main.py --log_file ${OUTPUT_DIR}/log.log -t ${TESTS} -m camembert --camembert_version camembert-base --exp_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR} -s ${SEED} --ignore_cached_encs

# after debiasing
# python sentbias/main.py --log_file ${OUTPUT_DIR}/log.log -t ${TESTS} -m camembert --camembert_version camembert-base --exp_dir ${SAVE_DIR} --data_dir ${DATA_DIR} -s ${SEED} --ignore_cached_encs --local_path ${LOCAL_DIR}
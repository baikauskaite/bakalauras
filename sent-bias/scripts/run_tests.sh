#!/bin/bash
mamba activate sent-bias

echo 'Note: this script should be called from the root of the repository' >&2

# TESTS=sent-weat1,sent-weat2,sent-weat3b,sent-weat5b,sent-weat6b,sent-weat7,sent-weat8,sent-weat9
TESTS=updated
set -e

LANGUAGE="french"
SEED=1111
timestamp=$(date +%s)
BASE_DIR="/home/viktorija/bakalaurinis/sent-bias"
SAVE_DIR="${BASE_DIR}/results/${timestamp}"
DATA_DIR="${BASE_DIR}/tests/${LANGUAGE}"
LOCAL_DIR="${BASE_DIR}/data"
mkdir "$SAVE_DIR"

# CamemBERT
# before debiasing
python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m camembert --camembert_version camembert-base --exp_dir ${SAVE_DIR} --data_dir ${DATA_DIR} -s ${SEED} --ignore_cached_encs

# after debiasing
# python sentbias/main.py --log_file ${SAVE_DIR}/log.log -t ${TESTS} -m camembert --camembert_version camembert-base --exp_dir ${SAVE_DIR} --data_dir ${DATA_DIR} -s ${SEED} --ignore_cached_encs --local_path ${LOCAL_DIR}
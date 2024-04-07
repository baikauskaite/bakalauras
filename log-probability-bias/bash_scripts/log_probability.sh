mamba activate sent-bias

set -e

MODEL_NAME="camembert"
LANGUAGE="french"
TESTS="weat0"
TYPE="pre"

formatted_date=$(date "+%m.%d-%H.%M")
BASE_DIR="/home/viktorija/bakalaurinis/log-probability-bias"
OUTPUT_DIR="${BASE_DIR}/results/${TYPE}-${formatted_date}"
OUT_FILE_SCORES="${OUTPUT_DIR}/${MODEL_NAME}_scores.tsv"
OUT_FILE_SIGNIFICANCE="${OUTPUT_DIR}/${MODEL_NAME}_significance.txt"

mkdir "$OUTPUT_DIR"

cd "$BASE_DIR/scripts" 

python log_probability_bias_scores.py \
    --model ${MODEL_NAME} \
    --demographic 'GEND' \
    --data_dir "${BASE_DIR}/tests/${LANGUAGE}" \
    --tests ${TESTS} \
    --out_file "${OUT_FILE_SCORES}"
 
python statistical_significance.py ${OUT_FILE_SCORES} > ${OUT_FILE_SIGNIFICANCE}


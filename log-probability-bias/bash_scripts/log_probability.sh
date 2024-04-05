set -e 
source activate hurtfulwords

BASE_DIR="/home/viktorija/bakalaurinis/log-probability-bias"
OUTPUT_DIR="${BASE_DIR}/results"
#MODEL_NAME="baseline_clinical_BERT_1_epoch_512"
#MODEL_NAME="adv_clinical_BERT_1_epoch_512" 
MODEL_NAME="CamemBERT"

cd "$BASE_DIR/scripts"

python log_probability_bias_scores.py \
    --model "${OUTPUT_DIR}/models/${MODEL_NAME}/" \
    --demographic 'GEND' \
    --template_file "${BASE_DIR}/fill_in_blanks_examples/templates.txt" \
    --attributes_file "${BASE_DIR}/fill_in_blanks_examples/attributes.csv" \
    --out_file "${OUTPUT_DIR}/${MODEL_NAME}_log_scores.tsv" 
 
python statistical_significance.py "${OUTPUT_DIR}/${MODEL_NAME}_log_scores.tsv" > "${OUTPUT_DIR}/${MODEL_NAME}_log_score_significance.txt"


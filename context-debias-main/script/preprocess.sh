model_type=$1
data=$2
seed=42
block_size=128
OUTPUT_DIR=../preprocess/$seed/$model_type
LANGUAGE="french"

rm -r $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

python -u ../src/preprocess.py --input ../data/$data \
                        --stereotypes ../data/${LANGUAGE}/stereotypes.txt \
                        --attributes ../data/${LANGUAGE}/female.txt,../data/${LANGUAGE}/male.txt \
                        --output $OUTPUT_DIR \
                        --seed $seed \
                        --block_size $block_size \
                        --model_type $model_type


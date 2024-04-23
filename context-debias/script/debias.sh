mamba activate sent-bias

model_type=camembert
gpu=1
debias_layer=all # first last all
loss_target=token # token sentence
dev_data_size=1000
seed=42
alpha=0.2
beta=0.8

if [ $model_type = 'bert' ]; then
    model_name_or_path=bert-base-uncased
elif [ $model_type = 'camembert' ]; then
    model_name_or_path=camembert-base
fi

BASE_DIR="/home/viktorija/bakalaurinis/context-debias"

TRAIN_DATA="${BASE_DIR}/preprocess/$seed/$model_type/data.bin"
OUTPUT_DIR="${BASE_DIR}/../models/${model_type}-debiased"

rm -r $OUTPUT_DIR

echo $model_type $seed

python ../src/run_debias_mlm.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$model_type \
    --model_name_or_path=$model_name_or_path \
    --do_train \
    --data_file=$TRAIN_DATA \
    --do_eval \
    --learning_rate 5e-5 \
    --per_gpu_train_batch_size 32 \
    --per_gpu_eval_batch_size 32 \
    --num_train_epochs 3 \
    --block_size 128 \
    --loss_target $loss_target \
    --debias_layer $debias_layer \
    --seed $seed \
    --evaluate_during_training \
    --weighted_loss $alpha $beta \
    --dev_data_size $dev_data_size \
    --square_loss \
    --line_by_line \
    --mlm

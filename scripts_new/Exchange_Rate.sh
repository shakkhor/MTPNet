# Random Seeds
seeds=(2023)
lr=5e-5
model=mcformer
patch_sizes='58, 96'
# Dozer attention parameters
local_windows='3, 3'
strides='0, 0'
vary_len=1
devices=1
gpu=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_sizes "$patch_sizes" --loss L1 \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len --devices "$devices" --gpu "$gpu"

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 48 --pred_len 192 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_sizes "$patch_sizes" --loss L1 \
    --local_windows "$local_windows" --strides "$strides" --devices "$devices" --gpu "$gpu"

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_sizes "$patch_sizes" --loss L1 \
    --local_windows "$local_windows" --strides "$strides" --devices "$devices" --gpu "$gpu"

    #----------------------------------predict length 720---------------------------------------
    python run.py --seed $seed --data Exchange --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 720 --embed_dim 32 --dropout 0.2 \
    --learning_rate $lr --patch_sizes "$patch_sizes" --loss L1 \
    --local_windows "$local_windows" --strides "$strides" --devices "$devices" --gpu "$gpu"
done









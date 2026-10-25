# 10 mins
# Random Seeds
seeds=(2023)
lr=1e-3
model=mcformer
patch_sizes='12, 96'
# Dozer attention parameters
local_windows='3, 3'
strides='3, 3'
vary_len=1
gpu=0
devices=0
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data Weather --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 1e-4  \
    --local_windows "$local_windows" --strides "$strides" --patch_sizes "$patch_sizes" --gpu "$gpu" --devices "$devices"

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data Weather  --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 192 --embed_dim 8 \
    --learning_rate 1e-4  \
    --local_windows "$local_windows" --strides "$strides" --patch_sizes "$patch_sizes" --gpu "$gpu" --devices "$devices"

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data Weather --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 8 \
    --learning_rate 1e-4  \
    --local_windows "$local_windows" --strides "$strides" --patch_sizes "$patch_sizes" --gpu "$gpu" --devices "$devices"

    #----------------------------------predict length 96---------------------------------------------------------------------------
    python run.py --seed $seed --data Weather --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 720 --embed_dim 8 \
    --learning_rate 1e-4  \
    --local_windows "$local_windows" --strides "$strides" --patch_sizes "$patch_sizes" --gpu "$gpu" --devices "$devices"
done
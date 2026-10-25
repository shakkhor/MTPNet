# Random Seeds
# 15 minutes
seeds=(2023)
lr=8e-5
model=mcformer
patch_sizes='12, 16'
# Dozer attention parameters
local_windows='2, 1'
strides='4, 0'
vary_len=1
devices=0
gpu=0
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    echo $seed
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data ETTm1 --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --devices "$devices" --gpu $gpu

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data ETTm1  --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 192 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --devices "$devices" --gpu $gpu

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data ETTm1 --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --devices "$devices" --gpu $gpu

    #----------------------------------predict length 720---------------------------------------
    python run.py --seed $seed --data ETTm1 --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 720 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --devices "$devices" --gpu $gpu
done

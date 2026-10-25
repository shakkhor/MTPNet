# Random Seeds
seeds=(2023)
lr=5e-4
model=mcformer
patch_sizes='24, 48, 96'
# Dozer attention parameters
local_windows='3, 3, 3'
strides='0, 0, 0'
vary_len=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 24---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 24 --pred_len 24 --embed_dim 64 --dropout 0.4 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

    #----------------------------------predict length 36---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 24 --pred_len 36 --embed_dim 64 --dropout 0.4 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

    #----------------------------------predict length 48---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 24 --pred_len 48 --embed_dim 64 --dropout 0.4 \
    --learning_rate 5e-4 --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

    #----------------------------------predict length 60---------------------------------------
    python run.py --seed $seed --data ILI --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 24 --pred_len 60 --embed_dim 64 --dropout 0.2 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len
done

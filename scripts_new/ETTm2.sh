# Random Seeds
# 15 minutes
seeds=(2023)
lr=5e-5
model=mcformer
#Dozer attention parameters
patch_sizes='12, 16'
# Dozer attention parameters
local_windows='2, 1'
strides='4, 0'
vary_len=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    echo $seed
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data ETTm2 --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate $lr  \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data ETTm2  --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 192 --embed_dim 8 \
    --learning_rate $lr  \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data ETTm2 --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 8 \
    --learning_rate $lr  \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

    #----------------------------------predict length 720---------------------------------------
    python run.py --seed $seed --data ETTm2 --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 720 --embed_dim 8 \
    --learning_rate $lr  \
    --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len
done

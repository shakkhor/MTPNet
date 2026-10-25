# 10 mins
# Random Seeds
seeds=(2023)
lr=9e-2
model=mcformer
patch_sizes='58, 96'
# Dozer attention parameters
local_windows='1, 1'
strides='2, 1'
devices=1
gpu=1
d_model=256
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data electricity --model $model --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 \
    --learning_rate "$lr" --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides" --devices $devices --gpu $gpu --d_model "$d_model"

#    #----------------------------------predict length 192---------------------------------------
#    python run.py --seed $seed --data electricity --model $model --moving_avg '13, 17' \
#    --seq_len 96 --label_len 96 --pred_len 192 \
#    --learning_rate "$lr" --patch_sizes "$patch_sizes" \
#    --local_windows "$local_windows" --strides "$strides" --devices $devices --gpu $gpu
#
#    #----------------------------------predict length 336---------------------------------------
#    python run.py --seed $seed --data electricity --model $model --moving_avg '13, 17' \
#    --seq_len 96 --label_len 96 --pred_len 336 \
#    --learning_rate "$lr" --patch_sizes "$patch_sizes" \
#    --local_windows "$local_windows" --strides "$strides" --devices $devices --gpu $gpu
#
#    #----------------------------------predict length 720---------------------------------------
#    python run.py --seed $seed --data electricity --model $model --batch_size 16 --moving_avg '13, 17' \
#    --seq_len 96 --label_len 96 --pred_len 720 \
#    --learning_rate "$lr" --patch_sizes "$patch_sizes" \
#    --local_windows "$local_windows" --strides "$strides" --devices $devices --gpu $gpu
done
printf '\a'
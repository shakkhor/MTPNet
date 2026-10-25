# Random Seeds
seeds=(2023)
lr=1e-3
model=mcformer
patch_sizes='24, 48, 96'
# Dozer attention parameters
local_windows='3, 3, 2, 1'
strides='7, 7, 3, 0'
vary_len=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data ETTh2 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8  --patch_sizes "$patch_sizes" \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

      #----------------------------------predict length 192---------------------------------------
      python run.py --seed $seed --data ETTh2 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 192 --embed_dim 8  --patch_sizes "$patch_sizes" \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

      #----------------------------------predict length 336---------------------------------------
      python run.py --seed $seed --data ETTh2 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 8  --patch_sizes "$patch_sizes" \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len

      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data ETTh2 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 720 --embed_dim 8  --patch_sizes "$patch_sizes" \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len
done




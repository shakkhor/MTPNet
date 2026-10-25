# Random Seeds
seeds=(2023)
lr=5e-6
model=mcformer
# Dozer attention
patch_sizes='24, 48, 96'
#local_windows='3, 3, 3, 3'
#strides='7, 7, 7, 7'
local_windows='3, 3, 3'
strides='7, 7, 7'
vary_len=1
devices=1
gpu=1
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
      echo $seed
      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len --patch_sizes "$patch_sizes" --devices "$devices" --gpu $gpu

      #----------------------------------predict length 192---------------------------------------
      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 192 --embed_dim 8 \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len --patch_sizes "$patch_sizes" --devices "$devices" --gpu $gpu

      #----------------------------------predict length 336---------------------------------------
      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 8 \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len --patch_sizes "$patch_sizes" --devices "$devices" --gpu $gpu

      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
      --seq_len 96 --label_len 96 --pred_len 720 --embed_dim 8 \
      --learning_rate $lr  \
      --local_windows "$local_windows" --strides "$strides" --vary_len $vary_len --patch_sizes "$patch_sizes" --devices "$devices" --gpu $gpu
done
printf '\a'

#
## Random Seeds
#seeds=(2023)
#lr=5e-5
#model=dozerformer_Linear
#patch_size=24
## Dozer attention parameters
#local_window=3
#stride=7
#vary_len=1
## shellcheck disable=SC2068
#for seed in ${seeds[@]}
#do
#      echo $seed
#      #----------------------------------predict length 96---------------------------------------
#      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
#      --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
#
#      #----------------------------------predict length 192---------------------------------------
#      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
#      --seq_len 720 --label_len 96 --pred_len 192 --embed_dim 8 \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
#
#      #----------------------------------predict length 336---------------------------------------
#      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
#      --seq_len 720 --label_len 96 --pred_len 336 --embed_dim 8 \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
#
#      #----------------------------------predict length 720---------------------------------------
#      python run.py --seed $seed --data ETTh1 --model $model --moving_avg '13, 17' \
#      --seq_len 720 --label_len 96 --pred_len 720 --embed_dim 8 \
#      --learning_rate $lr --patch_size $patch_size \
#      --local_window $local_window --stride $stride --vary_len $vary_len
#done








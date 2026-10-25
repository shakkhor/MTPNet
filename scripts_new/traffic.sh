# Hourly data
# '8, 12, 24, 48'
seeds=(2023)
lr=5e-4
model=mcformer
# Dozer attention parameters
local_windows='3, 3, 3'
strides='2, 2, 2'
vary_len=1
patch_sizes='8, 24, 48'
# shellcheck disable=SC2068
for seed in ${seeds[@]}
do
    echo $seed
    #----------------------------------predict length 96---------------------------------------
    python run.py --seed $seed --data Traffic --model $model --batch_size 32  --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides"

    #----------------------------------predict length 192---------------------------------------
    python run.py --seed $seed --data Traffic --model $model --batch_size 32 --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 192 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides"

    #----------------------------------predict length 336---------------------------------------
    python run.py --seed $seed --data Traffic --model $model --batch_size 32 --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 336 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides"

    #----------------------------------predict length 720---------------------------------------
    python run.py --seed $seed --data Traffic --model $model --batch_size 16 --moving_avg '13, 17' \
    --seq_len 96 --label_len 96 --pred_len 720 --embed_dim 8 \
    --learning_rate $lr --patch_sizes "$patch_sizes" \
    --local_windows "$local_windows" --strides "$strides"
done





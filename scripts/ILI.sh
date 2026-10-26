# Random Seeds
seeds=(2023)
data=ILI
model=SVPNet
seq_len=96
label_len=24
embed_dim=64
moving_avg='13, 17'
vary_len=1
lr=5e-4
batch_size=32
patch_size='24, 48, 96'
# Dozer attention parameters
local_window=3
stride=0
# shellcheck disable=SC2068

for seed in ${seeds[@]}
do
      echo $seed
      #----------------------------------predict length 24---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 24 --embed_dim $embed_dim --dropout 0.4 \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len

      #----------------------------------predict length 36---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 36 --embed_dim $embed_dim --dropout 0.4 \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len

      #----------------------------------predict length 48---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 48 --embed_dim $embed_dim --dropout 0.4 \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len

      #----------------------------------predict length 60---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 60 --embed_dim $embed_dim --dropout 0.2 \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len
done
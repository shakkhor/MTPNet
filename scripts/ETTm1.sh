# Random Seeds
seeds=(2023)
data=ETTm1
model=SVPNet
seq_len=720
label_len=96
embed_dim=8
moving_avg='13, 17'
vary_len=1
lr=8e-5
batch_size=32
patch_size='12, 16'
# Dozer attention parameters
local_window=2
stride=4
# shellcheck disable=SC2068

for seed in ${seeds[@]}
do
      echo $seed
      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 96 --embed_dim $embed_dim \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len

      #----------------------------------predict length 192---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 192 --embed_dim $embed_dim \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len

      #----------------------------------predict length 336---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 336 --embed_dim $embed_dim \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len

      #----------------------------------predict length 720---------------------------------------
      python run.py --seed $seed --data $data --model $model --batch_size $batch_size --moving_avg "$moving_avg" \
      --seq_len $seq_len --label_len $label_len --pred_len 720 --embed_dim $embed_dim \
      --learning_rate $lr --patch_size "$patch_size" --local_window $local_window --stride $stride --vary_len $vary_len
done

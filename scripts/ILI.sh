# Random Seeds
seeds=(2023)
# shellcheck disable=SC2068

for seed in ${seeds[@]}
do
      echo $seed
      #----------------------------------predict length 24---------------------------------------
      python run.py --seed $seed --data ILI --model SVPNet --moving_avg '13, 17' \
      --seq_len 96 --label_len 24 --pred_len 24 --embed_dim 64 --dropout 0.4 \
      --learning_rate 5e-4 --patch_size '24, 48, 96' --local_window 3 --stride 0 --vary_len 1

      #----------------------------------predict length 36---------------------------------------
      python run.py --seed $seed --data ILI --model SVPNet --moving_avg '13, 17' \
      --seq_len 96 --label_len 24 --pred_len 36 --embed_dim 64 --dropout 0.4 \
      --learning_rate 5e-4 --patch_size '24, 48, 96' --local_window 3 --stride 0 --vary_len 1

      #----------------------------------predict length 48---------------------------------------
      python run.py --seed $seed --data ILI --model SVPNet --moving_avg '13, 17' \
      --seq_len 96 --label_len 24 --pred_len 48 --embed_dim 64 --dropout 0.4 \
      --learning_rate 5e-4 --patch_size '24, 48, 96' --local_window 3 --stride 0 --vary_len 1

      #----------------------------------predict length 60---------------------------------------
      python run.py --seed $seed --data ILI --model SVPNet --moving_avg '13, 17' \
      --seq_len 96 --label_len 24 --pred_len 60 --embed_dim 64 --dropout 0.2 \
      --learning_rate 5e-4 --patch_size '24, 48, 96' --local_window 3 --stride 0 --vary_len 1
done
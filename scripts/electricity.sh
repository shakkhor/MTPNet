# Random Seeds
seeds=(2023)
# shellcheck disable=SC2068

for seed in ${seeds[@]}
do
      echo $seed
      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data electricity --model SVPNet --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 96 --embed_dim 8 \
      --learning_rate 9e-2 --patch_size '58, 96' --local_window 1 --stride 2 --vary_len 1

      #----------------------------------predict length 192---------------------------------------
      python run.py --seed $seed --data electricity --model SVPNet --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 192 --embed_dim 8 \
      --learning_rate 9e-2 --patch_size '58, 96' --local_window 1 --stride 2 --vary_len 1

      #----------------------------------predict length 336---------------------------------------
      python run.py --seed $seed --data electricity --model SVPNet --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 336 --embed_dim 8 \
      --learning_rate 9e-2 --patch_size '58, 96' --local_window 1 --stride 2 --vary_len 1

      #----------------------------------predict length 720---------------------------------------
      python run.py --seed $seed --data electricity --model SVPNet --batch_size 16 --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 720 --embed_dim 8 \
      --learning_rate 9e-2 --patch_size '58, 96' --local_window 1 --stride 2 --vary_len 1
done
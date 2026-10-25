# Random Seeds
seeds=(2023)
# shellcheck disable=SC2068

for seed in ${seeds[@]}
do
      echo $seed
      #----------------------------------predict length 96---------------------------------------
      python run.py --seed $seed --data ETTh1 --model SVPNet --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 96 --embed_dim 8 \
      --learning_rate 5e-6 --patch_size '24, 48, 96' --local_window 3 --stride 7 --vary_len 1

      #----------------------------------predict length 192---------------------------------------
      python run.py --seed $seed --data ETTh1 --model SVPNet --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 192 --embed_dim 8 \
      --learning_rate 5e-6 --patch_size '24, 48, 96' --local_window 3 --stride 7 --vary_len 1

      #----------------------------------predict length 336---------------------------------------
      python run.py --seed $seed --data ETTh1 --model SVPNet --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 336 --embed_dim 8 \
      --learning_rate 5e-6 --patch_size '24, 48, 96' --local_window 3 --stride 7 --vary_len 1

      #----------------------------------predict length 720---------------------------------------
      python run.py --seed $seed --data ETTh1 --model SVPNet --moving_avg '13, 17' \
      --seq_len 720 --label_len 96 --pred_len 720 --embed_dim 8 \
      --learning_rate 5e-6 --patch_size '24, 48, 96' --local_window 3 --stride 7 --vary_len 1
done

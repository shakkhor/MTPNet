#!/usr/bin/env bash
SEED=2023

# Separate working folder with links to the same code and data
REPO=$PWD
WORK=runs/mtpnet_newparams_seq96
mkdir -p "$WORK"
for f in run.py exp models utils visualizer data; do ln -sfn "$REPO/$f" "$WORK/$f"; done
cd "$WORK"
echo "Working folder: $PWD"

LOG_DIR=logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

GPU_LIST=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null))
[ ${#GPU_LIST[@]} -eq 0 ] && GPU_LIST=(0)
GPU_PID=()   # GPU_PID[i] = run going on GPU_LIST[i], empty if free
echo "GPUs: ${GPU_LIST[*]} | one run per GPU at a time"

# Wait until some GPU has no running job; sets FREE to its position in GPU_LIST
wait_for_free_gpu() {
  while true; do
    local running=" $(jobs -rp | tr '\n' ' ') "
    for i in "${!GPU_LIST[@]}"; do
      local pid=${GPU_PID[$i]}
      if [ -z "$pid" ]; then FREE=$i; return; fi
      case "$running" in *" $pid "*) ;; *) FREE=$i; return ;; esac
    done
    sleep 5
  done
}

# run DATA [run.py args...]
run() {
  local data=$1
  shift
  wait_for_free_gpu
  local gpu=${GPU_LIST[$FREE]}
  local log="$LOG_DIR/${data}_seed${SEED}.log"
  echo ">>> start  $data  (GPU $gpu)"
  (
    if CUDA_VISIBLE_DEVICES=$gpu python run.py --model MTPNet --seed "$SEED" --data "$data" "$@" > "$log" 2>&1; then
      echo "<<< done   $data  $(grep '^mse:' "$log")"
    else
      echo "!!! FAILED $data  $(grep -m1 'Error' "$log")  (see $log)"
    fi
  ) &
  GPU_PID[$FREE]=$!
}

run ETTh1 --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 5e-6 --patch_size '24, 48, 96' --local_window 3 --stride 7 --vary_len 1

run ETTh2 --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 1e-3 --patch_size '24, 48, 96' --local_window 3 --stride 7 --vary_len 1

run ETTm1 --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 8e-5 --patch_size '12, 16' --local_window 2 --stride 4

run ETTm2 --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 5e-5 --local_window 2 --stride 4 --vary_len 1

run Weather --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 1e-4 --patch_size '12, 96' --local_window 3 --stride 3

run Exchange --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 32 --dropout 0.2 \
    --learning_rate 5e-5 --patch_size '58, 96' --loss L1 --local_window 3 --stride 0 --vary_len 1

run ILI --moving_avg '13, 17' --seq_len 96 --label_len 24 --pred_len 24 --embed_dim 64 --dropout 0.4 \
    --learning_rate 5e-4 --patch_size '24, 48, 96' --local_window 3 --stride 0 --vary_len 1

run electricity --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 \
    --learning_rate 9e-2 --patch_size '58, 96' --local_window 1 --stride 2

run Traffic --batch_size 32 --moving_avg '13, 17' --seq_len 96 --label_len 96 --pred_len 96 --embed_dim 8 \
    --learning_rate 5e-4 --patch_size '8, 24, 48' --local_window 3 --stride 2

wait
echo "All runs finished. Results in $PWD/results.csv, logs in $PWD/$LOG_DIR"

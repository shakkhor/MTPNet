#!/usr/bin/env bash
# MTPNet with two patch sizes (12, 24): one run per dataset, input length 96.
# All 9 runs start at once, alternating between the GPUs.
# Run from the MTPNet repo root:  bash scripts/mtpnet_2patch_seq96.sh
#
# Everything this writes (results.csv, results/, checkpoints/, logs/) goes into its own
# folder, runs/mtpnet_2patch_seq96/, so it doesn't touch any other run's files.
SEQ_LEN=96
SEED=${SEED:-2023}
PATCH='12, 24'
W=${W:-3}   # local_window
S=${S:-1}   # stride

# Separate working folder with links to the same code and data
REPO=$PWD
WORK=runs/mtpnet_2patch_seq${SEQ_LEN}
mkdir -p "$WORK"
for f in run.py exp models utils visualizer data; do ln -sfn "$REPO/$f" "$WORK/$f"; done
cd "$WORK"
echo "Working folder: $PWD"

LOG_DIR=logs
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1

GPU_LIST=($(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null))
[ ${#GPU_LIST[@]} -eq 0 ] && GPU_LIST=(0)
N=0
echo "GPUs: ${GPU_LIST[*]} | patch_size '$PATCH' | local_window $W | stride $S"

# run DATA PRED_LEN LABEL_LEN [extra run.py args...]
run() {
  local data=$1 pl=$2 label=$3
  shift 3
  local gpu=${GPU_LIST[$((N % ${#GPU_LIST[@]}))]}
  N=$((N + 1))
  local name="${data}_sl${SEQ_LEN}_pl${pl}_w${W}_s${S}_seed${SEED}"
  local log="$LOG_DIR/$name.log"
  echo ">>> start  $name  (GPU $gpu)"
  (
    if CUDA_VISIBLE_DEVICES=$gpu python run.py --model MTPNet --seed "$SEED" --data "$data" \
         --seq_len "$SEQ_LEN" --label_len "$label" --pred_len "$pl" \
         --patch_size "$PATCH" --local_window "$W" --stride "$S" "$@" > "$log" 2>&1; then
      echo "<<< done   $name  $(grep '^mse:' "$log")"
    else
      echo "!!! FAILED $name  $(grep -m1 'Error' "$log")  (see $log)"
    fi
  ) &
}

run ETTh1       96 96 --embed_dim 8  --learning_rate 5e-5
run ETTh2       96 96 --embed_dim 8  --learning_rate 5e-5
run ETTm1       96 96 --embed_dim 8  --learning_rate 1e-5
run ETTm2       96 96 --embed_dim 8  --learning_rate 1e-5
run Weather     96 96 --embed_dim 8  --learning_rate 1e-4
run Exchange    96 96 --embed_dim 8  --learning_rate 1e-3 --trend_patch_size '24, 48' --dropout 0.2
run ILI         24 24 --embed_dim 64 --learning_rate 5e-4 --trend_patch_size '12, 24' --dropout 0.4
run electricity 96 96 --embed_dim 8  --learning_rate 5e-5 --batch_size 4
run Traffic     96 96 --embed_dim 8  --learning_rate 5e-4 --batch_size 4

wait
echo "All runs finished. Results in $PWD/results.csv, logs in $PWD/$LOG_DIR"
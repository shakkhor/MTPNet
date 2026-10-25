#!/usr/bin/env bash
SEQ_LEN=96
SEED=${SEED:-2023}
PRED_LEN=${PRED_LEN:-96}
ILI_PRED_LEN=${ILI_PRED_LEN:-24}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange ILI electricity Traffic"}
WINDOWS="1 3 5 7"
STRIDES="0 1 2 3 5"
LOG_DIR=logs/mtpnet_dozer_seq${SEQ_LEN}
mkdir -p "$LOG_DIR"
# write run.py output to the log files right away instead of buffering it
export PYTHONUNBUFFERED=1

GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ')}
GPUS=${GPUS:-0}
PER_GPU=${PER_GPU:-4}   # runs sharing one GPU at the same time

# One slot per run that can go at once, alternating GPUs: 0 1 0 1 ...
SLOT_GPU=()
for ((k = 0; k < PER_GPU; k++)); do for g in $GPUS; do SLOT_GPU+=("$g"); done; done
SLOT_PID=()   # SLOT_PID[i] = process running in slot i, empty if free
echo "GPUs: $GPUS | $PER_GPU runs per GPU, ${#SLOT_GPU[@]} at a time"

# Wait until some slot has no running job; sets FREE to that slot
wait_for_free_slot() {
  while true; do
    local running=" $(jobs -rp | tr '\n' ' ') "
    for i in "${!SLOT_GPU[@]}"; do
      local pid=${SLOT_PID[$i]}
      if [ -z "$pid" ]; then FREE=$i; return; fi
      case "$running" in *" $pid "*) ;; *) FREE=$i; return ;; esac
    done
    sleep 5
  done
}

# run_grid DATA PRED_LEN LABEL_LEN [extra run.py args...]
run_grid() {
  local data=$1 pl=$2 label=$3
  shift 3
  case " $DATASETS " in *" $data "*) ;; *) return ;; esac
  for w in $WINDOWS; do
    for s in $STRIDES; do
      name="${data}_sl${SEQ_LEN}_pl${pl}_w${w}_s${s}_seed${SEED}"
      log="$LOG_DIR/$name.log"
      # finished runs are skipped, so re-running the script only redoes failed/missing ones
      if grep -q '^mse:' "$log" 2>/dev/null; then echo "--- skip   $name  (already done)"; continue; fi
      # start in whichever slot is free, on that slot's GPU; the run sees it as cuda:0
      wait_for_free_slot
      gpu=${SLOT_GPU[$FREE]}
      echo ">>> start  $name  (GPU $gpu)"
      (
        if CUDA_VISIBLE_DEVICES=$gpu python run.py --model MTPNet --seed "$SEED" --data "$data" \
             --seq_len "$SEQ_LEN" --label_len "$label" --pred_len "$pl" \
             --local_window "$w" --stride "$s" "$@" > "$log" 2>&1; then
          echo "<<< done   $name  $(grep '^mse:' "$log")"
        else
          echo "!!! FAILED $name  $(grep -m1 'Error' "$log")  (see $log)"
        fi
      ) &
      SLOT_PID[$FREE]=$!
    done
  done
}

# Per-dataset settings from scripts/<dataset>.sh, with input length 96.
# Window: only odd values are distinct (w=2 behaves like w=3). Stride 0 = off;
# stride s attends every s+1 patches. One window/stride is shared by all patch scales.
run_grid ETTh1       $PRED_LEN     96 --embed_dim 8  --learning_rate 5e-5 --patch_size '4, 8, 12, 24, 48'
run_grid ETTh2       $PRED_LEN     96 --embed_dim 8  --learning_rate 5e-5 --patch_size '4, 8, 12, 24, 48'
run_grid ETTm1       $PRED_LEN     96 --embed_dim 8  --learning_rate 1e-5 --patch_size '4, 8, 12, 24, 48'
run_grid ETTm2       $PRED_LEN     96 --embed_dim 8  --learning_rate 1e-5 --patch_size '6, 12, 24, 32, 48, 96'
run_grid Weather     $PRED_LEN     96 --embed_dim 8  --learning_rate 1e-4 --patch_size '6, 12, 24, 48, 96'
run_grid Exchange    $PRED_LEN     96 --embed_dim 8  --learning_rate 1e-3 --patch_size '6, 12, 24' --trend_patch_size '24, 48' --dropout 0.2
run_grid ILI         $ILI_PRED_LEN 24 --embed_dim 64 --learning_rate 5e-4 --patch_size '4, 6, 8, 12' --trend_patch_size '12, 24' --dropout 0.4
run_grid electricity $PRED_LEN     96 --embed_dim 8  --learning_rate 5e-5 --patch_size '4, 8, 12, 24, 48' --batch_size 4
run_grid Traffic     $PRED_LEN     96 --embed_dim 8  --learning_rate 5e-4 --patch_size '8, 12, 24, 48' --batch_size 4

wait
echo "All runs finished. Results in results.csv, logs in $LOG_DIR"
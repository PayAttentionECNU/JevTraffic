#!/usr/bin/env bash
# v6b: DAgger round-2 students with and without layout augmentation, after v6 stages A-B showed augmentation
# alone hurts on held-out junctions (57.2 s vs 49.4 s). Data: v3 + DAgger r1 + r2 (scripts/v6_pipeline.sh).
#   v6-dagger2-0.5B         augmentation 0.5 (GPUs 0-3)
#   v6-dagger2-noaug-0.5B   no augmentation   (GPUs 4-7)
# Both get the full evaluation (all events, both variants, 3 seeds); both are reported.
#   bash scripts/screen_run.sh v6b bash scripts/v6b_pipeline.sh
set -uo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
P=.venv/bin/python
B=${BACKBONE:-$HOME/zzy/hf_models/Qwen2.5-0.5B-Instruct}
TEACHER=ckpt/unitsa/universal/model.zip
DATA=("data/sumo_unitsa/train/*.jsonl" "data/sumo_dagger/r1/*.jsonl" "data/sumo_dagger/r2/*.jsonl")
mkdir -p logs/v6
stage() { echo "$(date '+%F %T') == $*"; }

train() {  # train <out> <gpus comma-separated> <augment>, then calibrate
  local out=$1 gpus=$2 aug=$3
  local n=$(echo $gpus | tr , '\n' | wc -l)
  if [ ! -f $out/adapter_model.safetensors ]; then
    CUDA_VISIBLE_DEVICES=$gpus $P -m torch.distributed.run --standalone --nproc_per_node $n scripts/train_lora.py \
      --model $B --train "${DATA[@]}" "data/sumo_finetune/train/*.jsonl" --kinds switch situation --loss soft \
      --state_field state --lr 2e-4 --batch_size $((16 / n)) --save_every 0 --augment $aug --out $out \
      > logs/v6/train_$(basename $out).log 2>&1 || { echo "FAILED train $out"; return 1; }
  fi
  [ -f $out/calib.json ] || CUDA_VISIBLE_DEVICES=${gpus%%,*} $P scripts/sumo_calibrate.py --model $B --adapters $out \
    --fit "data/sumo_unitsa/val/*.jsonl" "data/sumo_finetune/val/*.jsonl" --state_field state \
    --eval "data/sumo_unitsa/test_junction/*.jsonl" "data/sumo_unitsa/test_pattern/*.jsonl" \
    > logs/v6/calib_$(basename $out).log 2>&1 || echo "FAILED calibrate $out"
}

evaluate() {  # evaluate <adapter>: all events, both variants, 3 seeds, GPUs 0-7
  for s in 0 1 2; do
    MODEL=$B ADAPTERS=$1 SLOTS=3 GPUS="0 1 2 3 4 5 6 7" SEED=$s \
      EXTRA="--teacher $TEACHER --action_type next_or_not --state_format full" bash scripts/run_sumo_jev.sh v3 \
      | grep -v "^done" || true
  done
}

stage "train both round-2 students"
train ckpt/v6-dagger2-0.5B 0,1,2,3 0.5 &
train ckpt/v6-dagger2-noaug-0.5B 4,5,6,7 0 &
wait
stage "evaluate v6-dagger2-noaug"
evaluate ckpt/v6-dagger2-noaug-0.5B
stage "evaluate v6-dagger2"
evaluate ckpt/v6-dagger2-0.5B
stage V6B_DONE

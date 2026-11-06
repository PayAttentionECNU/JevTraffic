#!/usr/bin/env bash
# v6: DAgger + layout augmentation for the keep/switch Jev student (this round may use GPUs 0-7).
#   bash scripts/screen_run.sh v6 bash scripts/v6_pipeline.sh
# Students (Qwen2.5-0.5B, full state, soft UniTSA targets, same recipe and effective batch 16 as v3):
#   v6-aug-0.5B       v3 data, layout augmentation                          (ablation: augmentation alone)
#   v6-dagger1-0.5B   v3 data + DAgger round 1 (states of the v3 student), augmentation
#   v6-dagger2-0.5B   + DAgger round 2 (states of dagger1)                 (the v6 model, fixed in advance)
# DAgger episodes use the training junctions and patterns only, seeds 11/12 (v3: 0/1, validation 100, test 0-2).
# Stages skip finished work, so the script can be restarted. Do not edit it while it runs.
set -uo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
P=.venv/bin/python
B=${BACKBONE:-$HOME/zzy/hf_models/Qwen2.5-0.5B-Instruct}
TEACHER=ckpt/unitsa/universal/model.zip
S0=ckpt/unitsa-jev-0.5B-full
AUG=${AUG:-0.5}
TRAIN_J="Beijing_Beihuan Beijing_Beishahe Beijing_Changjianglu Beijing_Gaojiaoyuan Beijing_Pinganli Beijing_Yongrunlu Chengdu_Chenghannanlu Chengdu_Guanghua Tianjin_zhijingdao"
PATTERNS="low_density high_density fluctuating_commuter increasing_demand"
V3_DATA="data/sumo_unitsa/train/*.jsonl"
mkdir -p logs/v6
stage() { echo "$(date '+%F %T') == $*"; }

train() {  # train <out> <gpus comma-separated> <data globs...>, then calibrate
  local out=$1 gpus=$2; shift 2
  local n=$(echo $gpus | tr , '\n' | wc -l)
  if [ ! -f $out/adapter_model.safetensors ]; then
    CUDA_VISIBLE_DEVICES=$gpus $P -m torch.distributed.run --standalone --nproc_per_node $n scripts/train_lora.py --model $B \
      --train "$@" "data/sumo_finetune/train/*.jsonl" --kinds switch situation --loss soft --state_field state \
      --lr 2e-4 --batch_size $((16 / n)) --save_every 0 --augment $AUG --out $out > logs/v6/train_$(basename $out).log 2>&1 \
      || { echo "FAILED train $out"; return 1; }
  fi
  [ -f $out/calib.json ] || CUDA_VISIBLE_DEVICES=${gpus%%,*} $P scripts/sumo_calibrate.py --model $B --adapters $out \
    --fit "data/sumo_unitsa/val/*.jsonl" "data/sumo_finetune/val/*.jsonl" --state_field state \
    --eval "data/sumo_unitsa/test_junction/*.jsonl" "data/sumo_unitsa/test_pattern/*.jsonl" \
    > logs/v6/calib_$(basename $out).log 2>&1 || echo "FAILED calibrate $out"
}

evaluate() {  # evaluate <mode e2|v3> <adapter> <gpus> <slots>: 3 seeds, all 12 junctions
  local mode=$1 a=$2 g=$3 sl=$4
  for s in 0 1 2; do
    MODEL=$B ADAPTERS=$a SLOTS=$sl GPUS="$g" SEED=$s \
      EXTRA="--teacher $TEACHER --action_type next_or_not --state_format full" bash scripts/run_sumo_jev.sh $mode \
      | grep -v "^done" || true
  done
}

collect() {  # collect <round> <student adapter> <gpus...>: student drives, teacher labels, 3 processes per GPU
  local r=$1 student=$2; shift 2
  local gpus=($@) slots=3 eps=()
  for j in $TRAIN_J; do for net in easy normal; do for pat in $PATTERNS; do for e in 0 0.1; do
    eps+=("$j:${net}_$pat:$e:$((10 + r))")
  done; done; done; done
  local n=$(( ${#gpus[@]} * slots ))
  for s in $(seq 0 $((n - 1))); do
    local mine=()
    for k in "${!eps[@]}"; do [ $((k % n)) -eq $s ] && mine+=("${eps[$k]}"); done
    CUDA_VISIBLE_DEVICES=${gpus[$((s % ${#gpus[@]}))]} $P scripts/dagger_collect.py --model $B --adapters $student \
      --teacher $TEACHER --episodes "${mine[@]}" --tag r$r --out_dir data/sumo_dagger/r$r \
      > logs/v6/collect_r${r}_$s.log 2>&1 || echo "FAILED collect r$r slot $s" &
  done
  wait
  echo "collected r$r: $(ls data/sumo_dagger/r$r/*.jsonl | wc -l) episodes, $(cat data/sumo_dagger/r$r/*.jsonl | wc -l) records"
}

stage "A: augmentation-only student (GPUs 4-7) | DAgger round 1 with the v3 student, then dagger1 (GPUs 0-3)"
train ckpt/v6-aug-0.5B 4,5,6,7 "$V3_DATA" &
( collect 1 $S0 0 1 2 3 && train ckpt/v6-dagger1-0.5B 0,1,2,3 "$V3_DATA" "data/sumo_dagger/r1/*.jsonl" ) &
wait
stage "B: ablation evaluations in the background (no events, 3 seeds) | DAgger round 2 with dagger1"
( evaluate e2 ckpt/v6-aug-0.5B "0 1 2 3 4 5 6 7" 1; evaluate e2 ckpt/v6-dagger1-0.5B "0 1 2 3 4 5 6 7" 1 ) \
  > logs/v6/ablation_eval.log 2>&1 &
abl=$!
collect 2 ckpt/v6-dagger1-0.5B 0 1 2 3 4 5 6 7
stage "C: dagger2 (GPUs 0-7)"
train ckpt/v6-dagger2-0.5B 0,1,2,3,4,5,6,7 "$V3_DATA" "data/sumo_dagger/r1/*.jsonl" "data/sumo_dagger/r2/*.jsonl"
stage "D: dagger2, all events and both variants, 3 seeds"
evaluate v3 ckpt/v6-dagger2-0.5B "0 1 2 3 4 5 6 7" 3
wait $abl
stage V6_DONE

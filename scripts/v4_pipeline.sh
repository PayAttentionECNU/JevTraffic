#!/usr/bin/env bash
# v4: baselines for the zero-shot / few-shot main tables, and PPO on top of the distilled Jev student.
# Run each line under screen (scripts/screen_run.sh) so it survives ssh / the Claude app closing:
#   bash scripts/screen_run.sh v4_baselines bash scripts/v4_pipeline.sh baselines
#   bash scripts/screen_run.sh v4_jev       bash scripts/v4_pipeline.sh jev
# Stages (each skips work whose output exists, so a pipeline can be restarted):
#   baselines: universal PressLight / AttendLight / IntelliLight (GPU 6 for the DQN updates, SUMO on CPU)
#              -> few-shot adaptation of all four RL baselines (UniTSA included) on the held-out junctions
#              -> evaluation of universal + few-shot models (CPU)
#   jev:       PPO of the 0.5B student on the training junctions (GPU $JEV_GPU, default 6) -> few-shot PPO on the held-out
#              junctions -> evaluation (distilled, RL universal, RL few-shot; plain and with the event plan)
set -uo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
P=.venv/bin/python
HELD="France_Massy SouthKorea_Songdo Hongkong_YMT"
BUDGETS=${BUDGETS:-"1 10 50"}
STEPS=${STEPS:-2000000}
STUDENT=${STUDENT:-unitsa-jev-0.5B-full}
BACKBONE=${BACKBONE:-$HOME/zzy/hf_models/Qwen2.5-0.5B-Instruct}
FMT=${FMT:-full}
RL=rl-jev-${STUDENT#unitsa-jev-}
JEV_GPU=${JEV_GPU:-6}  # GPU 7 is shared with another user's job at 100% utilisation
mkdir -p logs/v4

stage() { echo "$(date '+%F %T') == $*"; }

baselines() {
  stage "universal DQN baselines"
  for a in presslight attendlight intellilight; do
    [ -f ckpt/$a/universal/model.zip ] && continue
    CUDA_VISIBLE_DEVICES=6 $P scripts/train_rl_baseline.py --algo $a --junctions train --copies 1 \
      --total_timesteps $STEPS --device cuda --out ckpt/$a/universal > logs/v4/train_${a}_universal.log 2>&1 &
  done
  wait
  stage "few-shot adaptation"
  for b in $BUDGETS; do
    copies=$(( (b + 1) / 2 )); [ $copies -gt 5 ] && copies=5  # workers = min(b, 2 nets x copies)
    for j in $HELD; do
      for a in presslight attendlight intellilight unitsa; do
        out=ckpt/$a/fs${b}_$j
        [ -f $out/model.zip ] && continue
        if [ $a = unitsa ]; then
          CUDA_VISIBLE_DEVICES=6 $P scripts/train_unitsa.py --junctions $j --episodes $b --copies $copies \
            --init ckpt/unitsa/universal/model.zip --device cuda --out $out > logs/v4/fs_${a}_${b}_$j.log 2>&1 &
        else
          CUDA_VISIBLE_DEVICES=6 $P scripts/train_rl_baseline.py --algo $a --junctions $j --episodes $b \
            --copies $copies --init ckpt/$a/universal/model.zip --device cuda --out $out \
            > logs/v4/fs_${a}_${b}_$j.log 2>&1 &
        fi
      done
    done
    wait
  done
  stage "evaluation"
  for a in presslight attendlight intellilight; do
    ALGO=$a MODELS=universal JOBS=40 bash scripts/run_rl_eval.sh
  done
  fs=""; for b in $BUDGETS; do fs="$fs fs$b"; done
  for a in presslight attendlight intellilight unitsa; do
    ALGO=$a MODELS="$fs" JUNCTIONS="$HELD" JOBS=40 bash scripts/run_rl_eval.sh
  done
  stage BASELINES_DONE
}

jev() {
  stage "PPO on the training junctions"
  if [ ! -f ckpt/$RL/adapter_model.safetensors ]; then
    CUDA_VISIBLE_DEVICES=$JEV_GPU $P scripts/rl_jev.py --model $BACKBONE --adapters ckpt/$STUDENT --state_format $FMT \
      --junctions train --workers 32 --total_steps ${RL_STEPS:-500000} --out ckpt/$RL > logs/v4/rl_$RL.log 2>&1 \
      || { stage "PPO FAILED (logs/v4/rl_$RL.log)"; return 1; }
  fi
  stage "few-shot PPO"
  for b in $BUDGETS; do
    for j in $HELD; do
      out=ckpt/$RL-fs${b}_$j
      [ -f $out/adapter_model.safetensors ] && continue
      # few-shot starts from the RL'd student: base + distilled + universal RL merged, a fresh LoRA on top
      CUDA_VISIBLE_DEVICES=$JEV_GPU $P scripts/rl_jev.py --model $BACKBONE --adapters ckpt/$STUDENT ckpt/$RL \
        --state_format $FMT --junctions $j --episodes $b --workers 10 --n_steps 32 --critic_warmup 0 \
        --critic_state ckpt/$RL/critic.pt --out $out > logs/v4/rl_fs_${b}_$j.log 2>&1
    done
  done
  stage "evaluation"
  T="--teacher ckpt/unitsa/universal/model.zip --action_type next_or_not --state_format $FMT"
  for s in 0 1 2; do
    MODEL=$BACKBONE ADAPTERS=ckpt/$STUDENT SLOTS=4 GPUS="6 7" SEED=$s EXTRA="$T" bash scripts/run_sumo_jev.sh v3
    MODEL=$BACKBONE ADAPTERS="ckpt/$STUDENT ckpt/$RL" SLOTS=4 GPUS="6 7" SEED=$s EXTRA="$T" bash scripts/run_sumo_jev.sh v3
    for b in $BUDGETS; do for j in $HELD; do
      MODEL=$BACKBONE ADAPTERS="ckpt/$STUDENT ckpt/$RL ckpt/$RL-fs${b}_$j" SLOTS=4 GPUS="6 7" SEED=$s EXTRA="$T" \
        JUNCTIONS=$j bash scripts/run_sumo_jev.sh v3
    done; done
  done
  stage JEV_DONE
}

case ${1:-} in
  baselines) baselines ;;
  jev) jev ;;
  *) echo "usage: $0 baselines|jev"; exit 1 ;;
esac

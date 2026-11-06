#!/usr/bin/env bash
# E2-E4 on SingleTSCBaselines with the fine-tuned Jev (GPU). One process per (junction, job): the model is loaded
# once and runs all 10 envs of the junction. SLOTS processes share each GPU (7B bf16 ~ 16 GB each).
#   JOBS file lines: "<adapters (comma-separated)> <events (comma-separated)> <variants (comma-separated)>"
#   ADAPTERS="ckpt/sumo-distill" GPUS="6 7" bash scripts/run_sumo_jev.sh e2      (no events, plain Jev)
#   ADAPTERS="ckpt/sumo-distill" GPUS="..." bash scripts/run_sumo_jev.sh e34                  (context + events)
#   v3 keep/switch students (EXTRA goes to run_sumo.py as is):
#   MODEL=~/zzy/hf_models/Qwen2.5-0.5B-Instruct ADAPTERS=ckpt/unitsa-jev-0.5B-compact \
#     EXTRA="--action_type next_or_not --state_format compact --teacher ckpt/unitsa/universal/model.zip" \
#     bash scripts/run_sumo_jev.sh v3
set -uo pipefail
ulimit -c 0  # no core dumps (a crashed 7B process writes several GB)
cd "$(dirname "$0")/.."
MODEL=${MODEL:-$HOME/zzy/hf_models/Qwen2.5-7B-Instruct}
ADAPTERS=${ADAPTERS-ckpt/sumo-distill}  # empty = zero-shot base model
GPUS=(${GPUS:-6 7})  # our allocation on the shared server: GPUs 6 and 7
SLOTS=${SLOTS:-2}
SEED=${SEED:-0}
EXTRA=${EXTRA:-}
MODE=${1:-e2}
J=${JUNCTIONS:-"Beijing_Beihuan Beijing_Beishahe Beijing_Changjianglu Beijing_Gaojiaoyuan Beijing_Pinganli Beijing_Yongrunlu Chengdu_Chenghannanlu Chengdu_Guanghua Tianjin_zhijingdao France_Massy SouthKorea_Songdo Hongkong_YMT"}
ENVS=""; for net in easy normal; do for pat in low_density high_density fluctuating_commuter increasing_demand random_perturbation; do ENVS="$ENVS ${net}_$pat"; done; done
mkdir -p logs/sumo_jev
case $MODE in
  e2)  EVENTS="none";                        VARIANTS="jev:none:none" ;;
  e34) EVENTS="none event_1 random:1 random:2"; VARIANTS="jev:none:none jev:triggered:rule jev:triggered:jev jev:always:jev" ;;
  v3)  EVENTS="none event_1 random:1 random:2"; VARIANTS="jev:none:none jev:triggered:jev" ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
jobs=()  # one job per (junction, event set) so the slots stay busy
for j in $J; do for e in $EVENTS; do jobs+=("$j $e"); done; done
nslots=$(( ${#GPUS[@]} * SLOTS ))
for s in $(seq 0 $((nslots - 1))); do
  gpu=${GPUS[$(( s % ${#GPUS[@]} ))]}
  (
    for k in "${!jobs[@]}"; do
      [ $(( k % nslots )) -eq "$s" ] || continue
      set -- ${jobs[$k]}
      j=$1; ev=$2
      tag=$(for a in $ADAPTERS; do basename "$a"; done | paste -sd+)
      log=logs/sumo_jev/${MODE}_${tag:-base}_${j}_${ev/:/}_s$SEED.log
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python run_sumo.py --junction "$j" --env_name $ENVS --events "$ev" \
        --variants $VARIANTS --model "$MODEL" ${ADAPTERS:+--adapters $ADAPTERS} --seed "$SEED" $EXTRA --skip_existing \
        > "$log" 2>&1 && echo "done $MODE $tag $j $ev" || echo "FAILED $MODE $tag $j $ev ($log)"
    done
  ) &
done
wait
echo ALLDONE

#!/usr/bin/env bash
# Evaluate RL baselines in our harness (CPU, one episode per process), same scenarios / seeds / events as E1-E4.
#   ALGO=unitsa MODELS="universal" bash scripts/run_rl_eval.sh                          # all 12 junctions
#   ALGO=presslight MODELS="fs1 fs10 fs50" JUNCTIONS="France_Massy ..." bash scripts/run_rl_eval.sh
# ALGO: unitsa | presslight | attendlight | intellilight. A model name other than "universal" resolves to
# ckpt/<algo>/<name>_<junction>/model.zip (scratch_, ft_, fs<N>_ ...).
set -uo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=
ALGO=${ALGO:-unitsa}
JOBS=${JOBS:-40}
SEEDS=${SEEDS:-"0 1 2"}
MODELS=${MODELS:-universal}
EVENTS=${EVENTS:-"none event_1 random:1 random:2"}
J=${JUNCTIONS:-"Beijing_Beihuan Beijing_Beishahe Beijing_Changjianglu Beijing_Gaojiaoyuan Beijing_Pinganli Beijing_Yongrunlu Chengdu_Chenghannanlu Chengdu_Guanghua Tianjin_zhijingdao France_Massy SouthKorea_Songdo Hongkong_YMT"}
ENVS=""; for net in easy normal; do for pat in low_density high_density fluctuating_commuter increasing_demand random_perturbation; do ENVS="$ENVS ${net}_$pat"; done; done
flag=--rl_model; [ "$ALGO" = unitsa ] && flag=--unitsa_model
mkdir -p logs/sumo_rl
{
  for m in $MODELS; do for j in $J; do
    path=ckpt/$ALGO/$m/model.zip; [ "$m" = universal ] || path=ckpt/$ALGO/${m}_$j/model.zip
    [ -f "$path" ] || { echo "missing $path" >&2; continue; }
    for e in $ENVS; do for s in $SEEDS; do for ev in $EVENTS; do
      echo "$j $e $s $ev $path"
    done; done; done
  done; done
} | xargs -P "$JOBS" -L 1 bash -c \
  'log=logs/sumo_rl/'"$ALGO"'_$(echo "$*" | tr " :/" "___").log; .venv/bin/python run_sumo.py --junction $0 --env_name $1 --seed $2 --events $3 --agent '"$ALGO $flag"' $4 --skip_existing > $log 2>&1 || echo "FAILED $*"'
echo done

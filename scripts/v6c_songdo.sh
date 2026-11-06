#!/usr/bin/env bash
# v6c: redo every Jev result on SouthKorea_Songdo after the approach-naming fix (sumo_env.compass_names).
# Songdo is rotated ~45 degrees and plain compass binning named its approaches north / north-2 / south / south-2;
# the other 11 junctions keep their names (their bins were already distinct), so training data and students are
# unchanged. Old Songdo Jev results go to results/sumo/runs_badnames/; the Jev few-shot RL models of Songdo were
# trained on the old names, so they are retrained (old ones kept as ckpt/badnames_*). UniTSA and the DQN / rule
# baselines do not read the text state and are not affected.
#   bash scripts/screen_run.sh v6c bash scripts/v6c_songdo.sh
set -uo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
P=.venv/bin/python
B=${BACKBONE:-$HOME/zzy/hf_models/Qwen2.5-0.5B-Instruct}
J=SouthKorea_Songdo
S0=ckpt/unitsa-jev-0.5B-full
RL=ckpt/rl5-0.5B-full
T="--teacher ckpt/unitsa/universal/model.zip --action_type next_or_not --state_format full"
stage() { echo "$(date '+%F %T') == $*"; }

stage "archive the Songdo Jev results and few-shot models made with the old names (once)"
if [ ! -f results/sumo/runs_badnames/.done ]; then
  mkdir -p results/sumo/runs_badnames
  find results/sumo/runs -maxdepth 1 -name "${J}__*__jev-ns*" -exec mv -t results/sumo/runs_badnames/ {} +
  find results/sumo/val_fs -path "*rl5-0.5B-full*" -name "${J}__*" -delete
  for b in 1 10 50; do [ -d $RL-fs${b}_$J ] && mv $RL-fs${b}_$J ckpt/badnames_$(basename $RL)-fs${b}_$J; done
  touch results/sumo/runs_badnames/.done
fi
echo "archived $(ls results/sumo/runs_badnames | wc -l) runs"

stage "retrain the Jev few-shot RL models of Songdo"
g=0
for b in 1 10 50; do
  out=$RL-fs${b}_$J
  [ -f $out/adapter_model.safetensors ] && continue
  CUDA_VISIBLE_DEVICES=$g $P scripts/rl_jev.py --model $B --adapters $S0 $RL --state_format full --junctions $J \
    --episodes $b --workers 10 --n_steps 32 --epochs 1 --lr 5e-6 --kl_coef 0.5 --reward little --wait_coef 0.1 \
    --critic_warmup 0 --critic_state $RL/critic.pt --normalizer $RL/normalizer.json --save_every 0 --out $out \
    > logs/v5/rl_fs_${b}_$J.log 2>&1 || echo "FAILED few-shot RL fs$b" &
  g=$((g + 1))
done
wait

stage "evaluate every Jev variant on Songdo, 3 seeds"
k=0
run() {  # run <mode> <adapters>
  for s in 0 1 2; do
    MODEL=$B ADAPTERS="$2" SLOTS=1 GPUS="$((k % 8))" SEED=$s JUNCTIONS=$J EXTRA="$T" bash scripts/run_sumo_jev.sh $1 \
      | grep -v "^done" &
    k=$((k + 1))
  done
}
run v3 "$S0"
run v3 "$S0 $RL"
for b in 1 10 50; do run v3 "$S0 $RL $RL-fs${b}_$J"; done
run v3 ckpt/v6-dagger2-noaug-0.5B
run v3 ckpt/v6-dagger2-0.5B
run e2 ckpt/v6-aug-0.5B
run e2 ckpt/v6-dagger1-0.5B
wait

stage "few-shot validation gate (only the Songdo Jev runs are recomputed)"
$P scripts/fewshot_gate.py --gpus 0 1 2 3 4 5 6 7 --slots 1 --cpu_jobs 16 > logs/v6/gate_v6c.log 2>&1 || echo "FAILED gate"
tail -14 logs/v6/gate_v6c.log
stage V6C_DONE

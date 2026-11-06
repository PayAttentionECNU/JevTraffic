#!/usr/bin/env bash
# Kept for the v3 commands: UniTSA evaluation is now scripts/run_rl_eval.sh with ALGO=unitsa.
exec env ALGO=unitsa bash "$(dirname "$0")/run_rl_eval.sh" "$@"

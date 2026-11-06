#!/usr/bin/env bash
# Run a long experiment in a detached screen session on the server, so it survives ssh / the Claude app closing.
#   bash scripts/screen_run.sh <name> <command...>        e.g. bash scripts/screen_run.sh tr_presslight bash scripts/v4_train.sh presslight
# Output: logs/screen/<name>.log (screen -L) ending with "EXIT <code>"; attach with `screen -r <name>`, list with `screen -ls`.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
mkdir -p logs/screen
if screen -ls | grep -q "[.]$name[[:space:]]"; then
  echo "screen session $name is already running" >&2; exit 1
fi
cmd=$(printf '%q ' "$@")
screen -dmS "$name" -L -Logfile "logs/screen/$name.log" bash -c "ulimit -c 0; cd $(pwd); $cmd; echo EXIT \$?"
echo "started screen $name -> logs/screen/$name.log"

#!/bin/bash
# Start a pipeline run on the 8 GB model host, detached, with a log the run monitor can read.
#   scripts/run_remote.sh configs/dev.yaml          # first real run (small dev slice)
#   scripts/run_remote.sh configs/full.yaml         # Saturday full run
# REFUSES to start before 5 PM ET on a weekday (checked here AND again on the host by the pipeline itself).
# Never run this from a scheduler that bypasses the guard. Do not start it before 5 PM today.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
CONFIG="${1:-configs/dev.yaml}"
HOST="${APART_REMOTE_SSH:-${APART_REMOTE_HOST:-model-host}}"
DIR="${APART_REMOTE_DIR:-apart-epistemics}"; DIR="${DIR#\~/}"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
"$PY" "$ROOT/scripts/run_pipeline.py" --check --config "$CONFIG" || { echo "refusing to start (run window closed)"; exit 3; }
"$ROOT/scripts/sync_to_remote.sh"
TAG="$(basename "$CONFIG" .yaml)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${TAG}--apart"
echo "run id: $RUN_ID"
ssh -o BatchMode=yes "$HOST" bash -s "$DIR" "$CONFIG" "$RUN_ID" <<'REMOTE'
set -euo pipefail
DIR="$1"; CONFIG="$2"; RUN_ID="$3"
cd "$DIR"
if pgrep -f "python -m apart_epistemics" >/dev/null; then echo "another apart run is already running on this host; not starting"; exit 4; fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader
nohup bash -c "
  . venv/bin/activate; export PYTHONPATH=src PYTHONUNBUFFERED=1
  echo \"[\$(date +%Y-%m-%dT%H:%M:%S)] started run_id=$RUN_ID config=$CONFIG\"
  python -m apart_epistemics run --config $CONFIG --run-id $RUN_ID; rc=\$?
  echo \"[\$(date +%Y-%m-%dT%H:%M:%S)] finished exit=\$rc\"
" > "logs/$RUN_ID.log" 2>&1 < /dev/null &
echo "started on $(hostname); log: $DIR/logs/$RUN_ID.log"
REMOTE
echo "pull results with: scripts/sync_from_remote.sh $RUN_ID"

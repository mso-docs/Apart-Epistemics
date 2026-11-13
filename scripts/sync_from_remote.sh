#!/bin/bash
# Pull a finished (or running) run's outputs and log from the model host. Read-only on the host.
#   scripts/sync_from_remote.sh <run_id>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
HOST="${APART_REMOTE_SSH:-${APART_REMOTE_HOST:-model-host}}"
DIR="${APART_REMOTE_DIR:-apart-epistemics}"; DIR="${DIR#\~/}"
RUN_ID="${1:?usage: sync_from_remote.sh <run_id>}"
mkdir -p "$ROOT/outputs/$RUN_ID" "$ROOT/logs"
rsync -az "$HOST:$DIR/outputs/$RUN_ID/" "$ROOT/outputs/$RUN_ID/"
rsync -az "$HOST:$DIR/logs/$RUN_ID.log" "$ROOT/logs/" || true
echo "pulled outputs/$RUN_ID"

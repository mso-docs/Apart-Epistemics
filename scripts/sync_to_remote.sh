#!/bin/bash
# Push code, configs and raw datasets to the 8 GB model host. No model is started. Safe at any time.
# Host/dir come from .env or the defaults below (an SSH alias from ~/.ssh/config).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
HOST="${APART_REMOTE_SSH:-${APART_REMOTE_HOST:-model-host}}"
DIR="${APART_REMOTE_DIR:-apart-epistemics}"; DIR="${DIR#\~/}"
cd "$ROOT"
ssh -o BatchMode=yes "$HOST" "mkdir -p $DIR/logs $DIR/outputs $DIR/data/raw"
rsync -az --delete --exclude='__pycache__' src/ "$HOST:$DIR/src/"
rsync -az --delete configs/ "$HOST:$DIR/configs/"
rsync -az scripts/ "$HOST:$DIR/scripts/"
rsync -az data/raw/ "$HOST:$DIR/data/raw/"
# read-only copy of the spend ledger for the run monitor's spend panel (judge calls happen on this Mac)
L="${APART_OPENROUTER_LEDGER:-$HOME/.local/share/openrouter-budget/apart_spend.jsonl}"; L="${L/#\~/$HOME}"
[ -f "$L" ] && rsync -az "$L" "$HOST:$DIR/logs/openrouter_spend.jsonl" || true
echo "synced to $HOST:$DIR"

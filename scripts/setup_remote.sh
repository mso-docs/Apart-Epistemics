#!/bin/bash
# One-time model-host setup (already done on 2026-11-13): venv + torch(cu128) + transformers + bitsandbytes + Phi-3.5-mini weights.
# Runs no inference. Re-runnable.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/.env" ] && set -a && . "$ROOT/.env" && set +a
HOST="${APART_REMOTE_SSH:-${APART_REMOTE_HOST:-model-host}}"
DIR="${APART_REMOTE_DIR:-apart-epistemics}"; DIR="${DIR#\~/}"
ssh -o BatchMode=yes "$HOST" bash -s "$DIR" <<'REMOTE'
set -euo pipefail
cd "$1"; mkdir -p models logs
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate bitsandbytes numpy scikit-learn pyyaml huggingface_hub
hf download microsoft/Phi-3.5-mini-instruct --local-dir models/Phi-3.5-mini-instruct
REMOTE

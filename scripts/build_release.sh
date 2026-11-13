#!/bin/bash
# Stage a publishable copy into release/ (git-ignored): code, configs, docs, tests, README. No data, weights, outputs, logs,
# .env, or service scripts that name hosts. Then run the scans on it. Does NOT commit, push, or enable Pages.
#   scripts/build_release.sh [--with-pages]   # --with-pages also exports the Evalboard static site into release/pages/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
rm -rf release && mkdir -p release
rsync -a --exclude='__pycache__' --exclude='*.egg-info' src configs docs tests pyproject.toml requirements.txt requirements-model.txt README.md .gitignore .gitleaks.toml .env.example release/
mkdir -p release/scripts
rsync -a scripts/download_data.py scripts/run_pipeline.py scripts/ingest_evalboard.py scripts/setup_evalboard.py scripts/pii_scan.py scripts/prepublish_scan.sh release/scripts/
if [ "${1:-}" = "--with-pages" ]; then
  DB="$HOME/Library/Application Support/evalboard/data/projects/apart-epistemics/workspace.sqlite3"
  EB="${EVALBOARD_ROOT:-$HOME/Documents/Code/modular_dashboard}"
  PYTHONPATH="$EB" python3 -m evalboard.export_static --db "$DB" --out release/pages/ --trim-bytes 250000
  echo "static pages exported to release/pages (review: question text is from public datasets; no PII expected)"
fi
echo "staged release/; scanning..."
"$ROOT/scripts/prepublish_scan.sh" --staged

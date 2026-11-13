#!/bin/bash
# Pre-release gate: gitleaks (custom rules) + PII/private-link scan. Never commits, pushes, or uploads anything.
#   scripts/prepublish_scan.sh            # scan the working tree
#   scripts/prepublish_scan.sh --staged   # scan release/ (built by scripts/build_release.sh)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
TARGET="."; [ "${1:-}" = "--staged" ] && TARGET="release"
rc=0
echo "== gitleaks ($TARGET) =="
gitleaks dir "$TARGET" --config .gitleaks.toml --no-banner --redact --exit-code 1 || rc=1
echo "== PII / private links =="
if [ "$TARGET" = "release" ]; then "$PY" scripts/pii_scan.py release || rc=1; else "$PY" scripts/pii_scan.py || rc=1; fi
[ $rc -eq 0 ] && echo "PREPUBLISH SCAN: CLEAN" || echo "PREPUBLISH SCAN: FINDINGS (do not publish)"
exit $rc

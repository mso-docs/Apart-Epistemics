#!/usr/bin/env python3
"""Start the pipeline, but REFUSE before 5 PM ET on weekdays (weekends unrestricted). No override flag by design.

    python3 scripts/run_pipeline.py --check                     # report whether the window is open; start nothing
    python3 scripts/run_pipeline.py --config configs/dev.yaml   # local run (needs torch + the model; normally use the remote wrapper)
    scripts/run_remote.sh configs/dev.yaml                      # the real path: runs on the 8 GB model host (also guarded)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apart_epistemics import guard  # noqa: E402
from apart_epistemics.config import load_config  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dev.yaml")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ok, why = guard.allowed(None, cfg["guard"]["timezone"], int(cfg["guard"]["weekday_start_hour"]))
    print(("OPEN: " if ok else "CLOSED: ") + why)
    if args.check:
        return 0 if ok else 3
    if not ok:
        return 3
    cmd = [sys.executable, "-m", "apart_epistemics", "run", "--config", args.config] + (["--run-id", args.run_id] if args.run_id else [])
    return subprocess.call(cmd, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})


if __name__ == "__main__":
    sys.exit(main())

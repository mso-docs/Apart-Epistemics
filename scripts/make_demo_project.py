#!/usr/bin/env python3
"""Build the clearly-labeled SYNTHETIC demo Evalboard project (separate DB; never the real project).

    python3 scripts/make_demo_project.py            # writes synthetic outputs and ingests into the DEMO db
Prints the db path; view with: python3 -m evalboard.server --db <that path> --port 8768
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apart_epistemics.analyze import run_analyze  # noqa: E402
from apart_epistemics.config import load_config  # noqa: E402
from apart_epistemics.synthetic import make_synthetic_run  # noqa: E402

DB = Path.home() / "Library/Application Support/evalboard/data/projects/apart-epistemics-DEMO-synthetic/workspace.sqlite3"
RUN = "SYNTH-demo--00000000"


def main():
    cfg = load_config(ROOT / "configs/synthetic.yaml")
    make_synthetic_run(cfg, RUN, n_per_class=80)
    run_analyze(cfg, RUN)
    DB.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/ingest_evalboard.py"), "--run", RUN, "--config", "configs/synthetic.yaml", "--db", str(DB)], cwd=ROOT)
    print("demo db:", DB)
    return rc


if __name__ == "__main__":
    sys.exit(main())

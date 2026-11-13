#!/usr/bin/env python3
"""Map a pipeline run into a LOCAL Evalboard project (traces + evalboard.interp/v1 artifacts).

    python3 scripts/ingest_evalboard.py --run <run_id> --dry-run            # build + validate + count; write nothing
    python3 scripts/ingest_evalboard.py --run <run_id>                      # POST to $APART_EVALBOARD_URL (default http://127.0.0.1:8767)
    python3 scripts/ingest_evalboard.py --run <run_id> --db /path/to/workspace.sqlite3

Reads outputs/<run_id>/{items.jsonl, analysis/*}; run `apart analyze` first. Re-running is safe (identical records are no-ops).
Safety: loopback URLs only; synthetic runs (run id starts with SYNTH-) are refused unless --db points at a path containing
'DEMO' (the separate, clearly labeled demo project). Standard library + the repo package only.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from apart_epistemics.config import load_config, resolve_path  # noqa: E402
from apart_epistemics.evalboard_export import attach_interp_refs, build_artifacts, build_traces  # noqa: E402
from apart_epistemics.runio import read_jsonl  # noqa: E402


def post(url, path, body):
    req = urllib.request.Request(url + path, json.dumps(body, allow_nan=False).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{path} rejected ({e.code}): {e.read().decode()[:600]}")


def get(url, path):
    with urllib.request.urlopen(url + path, timeout=60) as r:
        return json.load(r)


def load_store(db):
    for c in (os.environ.get("EVALBOARD_ROOT"), Path.home() / "Documents/Code/modular_dashboard", Path.home() / "Documents/code/modular_dashboard"):
        if c and (Path(c) / "evalboard" / "core.py").is_file():
            sys.path.insert(0, str(c))
            from evalboard.core import Store
            return Store(Path(db))
    raise SystemExit("Evalboard toolkit not found; set EVALBOARD_ROOT")


def merge_metrics(layout, layout_file):
    wanted = json.loads(Path(layout_file).read_text()).get("metrics") or []
    layout = dict(layout or {})
    metrics = list(layout.get("metrics") or [])
    have = {m.get("id"): i for i, m in enumerate(metrics)}
    changed = False
    for m in wanted:
        if m["id"] not in have:
            metrics.append(m); changed = True
        elif bool(m.get("primary")) != bool(metrics[have[m["id"]]].get("primary")):
            metrics[have[m["id"]]] = {**metrics[have[m["id"]]], "primary": bool(m.get("primary"))}; changed = True
    return {**layout, "metrics": metrics}, changed


def batches(items, max_bytes=6_000_000):
    b, size = [], 0
    for it in items:
        n = len(json.dumps(it))
        if b and size + n > max_bytes:
            yield b; b, size = [], 0
        b.append(it); size += n
    if b:
        yield b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--url", default=os.environ.get("APART_EVALBOARD_URL"))
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-layout", action="store_true")
    ap.add_argument("--layout-file", default=str(ROOT / "configs" / "evalboard_layout.json"))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    url = args.url or cfg["evalboard"]["url"]
    rdir = resolve_path(cfg["outputs"]["dir"]) / args.run
    for need in ("items.jsonl", "analysis/metrics.json", "analysis/curves.json", "analysis/probe_scores.jsonl"):
        if not (rdir / need).exists():
            raise SystemExit(f"missing {rdir / need}; run generate + analyze first")
    metrics = json.loads((rdir / "analysis/metrics.json").read_text())
    curves = json.loads((rdir / "analysis/curves.json").read_text())
    ps = {r["id"]: r["probe_score"] for r in read_jsonl(rdir / "analysis/probe_scores.jsonl")}
    rows = read_jsonl(rdir / "items.jsonl")
    synthetic = args.run.startswith("SYNTH-") or any(r.get("synthetic") for r in rows)
    if synthetic and not (args.db and "DEMO" in args.db):
        raise SystemExit("Refusing to ingest synthetic data into a real project (only into a --db path containing 'DEMO').")
    if not synthetic and args.db and "DEMO" in args.db:
        raise SystemExit("Refusing to put real data into the DEMO project.")
    model = cfg["model"]["name"]
    traces = build_traces(rdir, args.run, model, metrics, ps)
    arts = build_artifacts(args.run, model, metrics, curves, rows, ps, synthetic=synthetic)
    attach_interp_refs(traces, arts, args.run)
    print(f"{len(traces)} traces, {len(arts)} interpretability artifacts (run {args.run}{', SYNTHETIC' if synthetic else ''})")
    if args.dry_run:
        # validate with Evalboard's own validators when the toolkit is present
        try:
            sys.path.insert(0, str(Path.home() / "Documents/Code/modular_dashboard"))
            from evalboard.interp import validate_artifact
            for a in arts:
                validate_artifact(a)
            print("artifacts validate against evalboard.interp/v1")
        except ImportError:
            print("(Evalboard toolkit not importable here; skipped local validation)")
        return 0
    if args.db:
        store = load_store(args.db)
        for b in batches(traces):
            print("traces:", store.ingest(b))
        for b in batches(arts):
            print("interp:", store.ingest_interp(b))
        if not args.no_layout:
            lay, ch = merge_metrics(store.snapshot().get("layout") or {}, args.layout_file)
            if ch:
                store.save_layout(lay); print("layout metrics merged")
        return 0
    host = urllib.parse.urlsplit(url).hostname
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise SystemExit("Refusing non-loopback URL: traces stay on this machine.")
    for b in batches(traces):
        print("traces:", post(url, "/api/traces", {"traces": b}))
    for b in batches(arts):
        print("interp:", post(url, "/api/interp", {"artifacts": b}))
    if not args.no_layout:
        lay, ch = merge_metrics(get(url, "/api/workspace").get("layout") or {}, args.layout_file)
        if ch:
            post(url, "/api/layout", lay); print("layout metrics merged")
    return 0


if __name__ == "__main__":
    sys.exit(main())

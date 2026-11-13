"""apart CLI.

    apart check   --config configs/default.yaml        # validate config, data, budget wiring, run window (no model, no GPU)
    apart generate --config ... [--dry-run] [--limit N] [--run-id ID]
    apart analyze  --config ... --run-id ID
    apart run      --config ... [--dry-run]            # generate + analyze
    apart export   --config ... --run-id ID            # write Evalboard traces/artifacts as files (no network)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import guard
from .config import ROOT, load_config, resolve_path, validate_config
from .scrub import scrub_obj


def cmd_check(cfg, args) -> int:
    errs = validate_config(cfg)
    print(f"config: {cfg.get('_path')}")
    if errs:
        print("INVALID:\n  " + "\n  ".join(errs))
        return 2
    print("config: valid")
    rc = 0
    from .data import load_all_items
    try:
        items = load_all_items(cfg)
        c = Counter((i.dataset, "answerable" if i.answerable else "unanswerable", i.split) for i in items)
        print(f"data: {len(items)} items")
        for k in sorted(c):
            print(f"  {k[0]:9s} {k[1]:12s} {k[2]:5s} {c[k]}")
    except FileNotFoundError as e:
        print(f"data: MISSING ({e}); run scripts/download_data.py")
        rc = 1
    mp = resolve_path(cfg["model"]["path"]) if cfg["model"].get("path") else None
    print(f"model: {cfg['model']['name']}  quantization={cfg['model'].get('quantization')}  "
          f"local weights {'present' if mp and mp.exists() else 'NOT on this machine (expected on the model host)'}")
    try:
        from . import budget
        st = budget.status()
        print(f"budget: ${st['spent_usd']:.2f} of ${st['cap_usd']:.2f} spent; paid calls allowed={st['paid_calls_allowed']}; ledger={st['ledger']}")
    except Exception as e:  # noqa: BLE001
        print(f"budget: guard not loadable here ({e})" + ("" if cfg["judge"].get("backend") == "openrouter" else "; fine, judge.backend=none"))
        rc = rc or (1 if cfg["judge"].get("backend") == "openrouter" else 0)
    ok, why = guard.allowed(None, cfg["guard"]["timezone"], int(cfg["guard"]["weekday_start_hour"]))
    print(f"run window: {'OPEN' if ok else 'CLOSED'}: {why}")
    print(f"judge backend: {cfg['judge'].get('backend')}")
    return rc


def cmd_generate(cfg, args) -> int:
    from .generate import plan, run_generate
    from .runio import new_run_id
    if args.dry_run:
        rc = cmd_check(cfg, args)
        n = len(plan(cfg, args.limit))
        print(f"dry-run: would generate for {n} items (answer + logprobs + hidden states + verbalized confidence"
              f"{' + abstain-option pass' if cfg.get('behavior', {}).get('abstain_option_pass') else ''}); no model loaded.")
        return rc
    run_id = args.run_id or new_run_id(cfg, "generate")
    print(f"run id: {run_id}")
    run_generate(cfg, run_id, limit=args.limit)
    print(f"done: outputs/{run_id}")
    return 0


def cmd_analyze(cfg, args) -> int:
    from .analyze import run_analyze
    run_analyze(cfg, args.run_id)
    print(f"analysis written to outputs/{args.run_id}/analysis/")
    return 0


def cmd_run(cfg, args) -> int:
    if args.dry_run:
        return cmd_generate(cfg, args)
    from .runio import new_run_id
    args.run_id = args.run_id or new_run_id(cfg, "run")
    cmd_generate(cfg, args)
    rc = cmd_analyze(cfg, args)
    print("[apart] stage=export start", flush=True)
    return rc or cmd_export(cfg, args)


def cmd_export(cfg, args) -> int:
    from .evalboard_export import build_artifacts, build_traces, attach_interp_refs
    from .runio import read_jsonl, run_dir, write_json
    rdir = run_dir(cfg, args.run_id)
    metrics = json.loads((rdir / "analysis" / "metrics.json").read_text())
    curves = json.loads((rdir / "analysis" / "curves.json").read_text())
    ps = {r["id"]: r["probe_score"] for r in read_jsonl(rdir / "analysis" / "probe_scores.jsonl")}
    rows = read_jsonl(rdir / "items.jsonl")
    syn = bool(rows and rows[0].get("synthetic"))
    traces = build_traces(rdir, args.run_id, cfg["model"]["name"], metrics, ps)
    arts = build_artifacts(args.run_id, cfg["model"]["name"], metrics, curves, rows, ps, synthetic=syn)
    attach_interp_refs(traces, arts, args.run_id)
    out = rdir / "evalboard"
    out.mkdir(exist_ok=True)
    (out / "traces.jsonl").write_text("".join(json.dumps(scrub_obj(t), allow_nan=False) + "\n" for t in traces))
    for a in arts:
        write_json(out / "interp" / (a["id"].split(":", 1)[1] + ".json"), a)
    print(f"{len(traces)} traces, {len(arts)} artifacts -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="apart", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("generate", cmd_generate), ("analyze", cmd_analyze), ("run", cmd_run), ("export", cmd_export)):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/default.yaml")
        sp.add_argument("--run-id", default=None)
        sp.add_argument("--limit", type=int, default=None)
        sp.add_argument("--dry-run", action="store_true", help="validate everything, load no model")
        sp.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd in ("analyze", "export") and not args.run_id:
        ap.error("--run-id is required")
    try:
        return args.fn(cfg, args)
    except guard.OutsideRunWindow as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

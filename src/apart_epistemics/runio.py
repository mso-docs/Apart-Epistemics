"""Run directory layout and incremental readers/writers.

outputs/<run_id>/
  meta.json                 config snapshot, versions, host, timings
  items.jsonl               one row per item: prompt, answer, scores, confidence, logprobs (append-only, resumable)
  hidden/<dataset>-NNNN.npz ids (n,), H (n, n_layers, d) float16 last-prompt-token residual stream per layer
  analysis/metrics.json     probes, baselines, ECE, gating, gap numbers
  analysis/curves.json      reliability bins and coverage/gating curves
  analysis/probe_scores.jsonl  per-item best-layer probe score (+ split)
  evalboard/                (written by ingest --dry-run --export) traces.jsonl + interp/*.json
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .config import ROOT, resolve_path, config_hash
from .scrub import scrub_obj  # secrets/host/PII scrub at write time


def new_run_id(cfg: dict, tag: str = "run") -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{ts}-{tag}--{config_hash(cfg)}"


def run_dir(cfg: dict, run_id: str) -> Path:
    base = resolve_path(cfg["outputs"]["dir"])
    d = base / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(scrub_obj(r), allow_nan=False, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def clean_nan(obj):
    """Replace NaN/inf with None so JSON stays strict (missing stays missing)."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_nan(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(scrub_obj(clean_nan(obj)), indent=2, allow_nan=False) + "\n")


def save_hidden_shard(rdir: Path, dataset: str, ids: list[str], H: np.ndarray) -> Path:
    hd = rdir / "hidden"
    hd.mkdir(exist_ok=True)
    n = len(list(hd.glob(f"{dataset}-*.npz")))
    p = hd / f"{dataset}-{n:04d}.npz"
    np.savez(p, ids=np.array(ids), H=H.astype(np.float16))
    return p


def load_hidden(rdir: Path, dataset: str | None = None) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    mats = []
    pat = f"{dataset}-*.npz" if dataset else "*.npz"
    for p in sorted((Path(rdir) / "hidden").glob(pat)):
        z = np.load(p, allow_pickle=False)
        ids += [str(i) for i in z["ids"]]
        mats.append(z["H"])
    if not mats:
        return [], np.zeros((0, 0, 0), np.float16)
    return ids, np.concatenate(mats, 0)

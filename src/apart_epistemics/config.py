"""YAML config loading, `extends` merging and validation (no model or GPU needed)."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

REQUIRED = {
    "project": str,
    "seed": int,
    "model": dict,
    "datasets": dict,
    "splits": dict,
    "probe": dict,
    "metrics": dict,
    "judge": dict,
    "guard": dict,
    "evalboard": dict,
    "outputs": dict,
}
QUANT = ("none", "8bit", "4bit")
JUDGE_BACKENDS = ("none", "openrouter")
KNOWN_DATASETS = ("squad_v2", "selfaware")


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = (Path.cwd() / path) if (Path.cwd() / path).exists() else ROOT / path
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        data = deep_merge(base, data)
    data["_path"] = str(path)
    return data


def resolve_path(p: str | Path) -> Path:
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else ROOT / p


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty = valid). Never touches a model or the network."""
    errs: list[str] = []
    for key, typ in REQUIRED.items():
        if key not in cfg:
            errs.append(f"missing key: {key}")
        elif not isinstance(cfg[key], typ):
            errs.append(f"{key} must be {typ.__name__}")
    if errs:
        return errs
    m = cfg["model"]
    if not m.get("name"):
        errs.append("model.name is required")
    if m.get("quantization", "none") not in QUANT:
        errs.append(f"model.quantization must be one of {QUANT}")
    if int(m.get("max_new_tokens", 0)) < 1:
        errs.append("model.max_new_tokens must be >= 1")
    if m.get("batch_size", 1) != 1:
        errs.append("model.batch_size must be 1 (last-prompt-token hooks; see PROPOSAL fallback)")
    ds = cfg["datasets"]
    if not ds:
        errs.append("datasets must not be empty")
    for name, d in ds.items():
        if name not in KNOWN_DATASETS:
            errs.append(f"unknown dataset {name!r} (known: {KNOWN_DATASETS})")
            continue
        if not d.get("path"):
            errs.append(f"datasets.{name}.path is required")
        if int(d.get("n_per_class", 0)) < 1:
            errs.append(f"datasets.{name}.n_per_class must be >= 1")
    sp = cfg["splits"]
    total = sum(float(sp.get(k, 0)) for k in ("train", "val", "test"))
    if abs(total - 1.0) > 1e-6:
        errs.append(f"splits train+val+test must sum to 1 (got {total})")
    if any(float(sp.get(k, 0)) <= 0 for k in ("train", "val", "test")):
        errs.append("splits train, val and test must each be > 0")
    if int(cfg["metrics"].get("ece_bins", 0)) < 2:
        errs.append("metrics.ece_bins must be >= 2")
    j = cfg["judge"]
    if j.get("backend", "none") not in JUDGE_BACKENDS:
        errs.append(f"judge.backend must be one of {JUDGE_BACKENDS}")
    if j.get("backend") == "openrouter":
        if not j.get("model"):
            errs.append("judge.model is required for the openrouter backend")
        if not 0 < float(j.get("sample_frac", 0)) <= 1:
            errs.append("judge.sample_frac must be in (0, 1]")
    g = cfg["guard"]
    if not 0 <= int(g.get("weekday_start_hour", 17)) <= 23:
        errs.append("guard.weekday_start_hour must be 0-23")
    if not g.get("timezone"):
        errs.append("guard.timezone is required")
    if cfg["evalboard"].get("url", "").split("://")[-1].split(":")[0] not in ("127.0.0.1", "localhost", "::1", "[::1]"):
        errs.append("evalboard.url must be a loopback URL (traces stay local)")
    return errs


def config_hash(cfg: dict[str, Any]) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:8]

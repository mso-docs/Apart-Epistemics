"""Stage 1: for each item, greedy answer + logprobs + last-token hidden states + verbalized confidence.

Resumable: items already in items.jsonl are skipped. Refuses to start in the weekday window (guard) unless dry-run."""
from __future__ import annotations

import time

from . import guard
from .config import validate_config
from .data import load_all_items
from .data.loaders import _h
from .prompts import answer_messages, confidence_messages
from .runio import append_jsonl, read_jsonl, run_dir, save_hidden_shard, write_json
from .scoring import parse_confidence, score_answer


def plan(cfg: dict, limit: int | None = None):
    """Items to run (no model needed). Used by --dry-run and by the real run."""
    items = load_all_items(cfg)
    return items[:limit] if limit else items


def run_generate(cfg: dict, run_id: str, limit: int | None = None, shard_size: int = 100, model=None, now=None) -> dict:
    errs = validate_config(cfg)
    if errs:
        raise SystemExit("config invalid:\n  " + "\n  ".join(errs))
    why = guard.require_window(cfg, now)
    print(f"[guard] {why}", flush=True)
    rdir = run_dir(cfg, run_id)
    items = plan(cfg, limit)
    done = {r["id"] for r in read_jsonl(rdir / "items.jsonl")}
    todo = [i for i in items if i.id not in done]
    print(f"{len(items)} items planned, {len(done)} already done, {len(todo)} to run", flush=True)
    print("[apart] stage=load start", flush=True)
    if model is None:
        from .model import Phi
        model = Phi(cfg)
    print("[apart] stage=generate start", flush=True)
    bcfg = cfg.get("behavior", {})
    t0 = time.time()
    buf_rows, buf_ids, buf_H, cur_ds = [], [], [], None

    def flush():
        nonlocal buf_rows, buf_ids, buf_H
        if buf_rows:
            save_hidden_shard(rdir, cur_ds, buf_ids, __import__("numpy").stack(buf_H))
            append_jsonl(rdir / "items.jsonl", buf_rows)   # rows written after their hidden shard: resume-safe
            buf_rows, buf_ids, buf_H = [], [], []

    for n, it in enumerate(todo, 1):
        if cur_ds is not None and it.dataset != cur_ds:
            flush()
        cur_ds = it.dataset
        t1 = time.time()
        g = model.generate(answer_messages(it), want_hidden=True)
        sc = score_answer(it, g.text)
        cg = model.generate(confidence_messages(it, g.text), want_hidden=False, max_new=8)
        row = {**it.to_dict(), "response": g.text, **sc, "verbal_confidence": parse_confidence(cg.text),
               "confidence_raw": cg.text, "mean_logprob": g.mean_logprob, "max_logprob": g.max_logprob,
               "min_logprob": g.min_logprob, "mean_entropy": g.mean_entropy, "first_token_entropy": g.first_token_entropy,
               "n_answer_tokens": g.n_tokens, "duration_ms": int((time.time() - t1) * 1000)}
        if bcfg.get("abstain_option_pass", False):
            ga = model.generate(answer_messages(it, abstain_option=True), want_hidden=False)
            row["abstain_option_response"] = ga.text
            row["abstained_when_allowed"] = score_answer(it, ga.text)["abstained"]
        cons = cfg.get("consistency", {})
        if cons.get("enabled") and (_h(it.id, 'cons') % 1000) / 1000 < float(cons.get("subset_frac", 0.3)):
            samples = [model.generate(answer_messages(it), want_hidden=False, do_sample=True,
                                      temperature=float(cons.get("temperature", 0.7))).text
                       for _ in range(int(cons.get("n_samples", 5)))]
            from .scoring import normalize
            agree = sum(normalize(s) == normalize(g.text) for s in samples) / len(samples)
            row["consistency"], row["consistency_samples"] = agree, samples
        buf_rows.append(row), buf_ids.append(it.id), buf_H.append(g.hidden_last)
        if len(buf_rows) >= shard_size:
            flush()
        if n % 20 == 0:
            print(f"  {n}/{len(todo)}  {n / (time.time() - t0):.2f} it/s", flush=True)
    flush()
    versions = {}
    for mod in ("torch", "transformers", "bitsandbytes", "numpy", "sklearn"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = None
    meta = {"run_id": run_id, "versions": versions, "config": {k: v for k, v in cfg.items() if not k.startswith("_")}, "n_items": len(items),
            "elapsed_s": time.time() - t0}
    write_json(rdir / "meta.json", meta)
    return meta

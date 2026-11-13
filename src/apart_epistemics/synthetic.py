"""SYNTHETIC run generator for tests and the clearly-labeled demo Evalboard project. Never mix into real runs.

Every row carries synthetic=True and the run id starts with 'SYNTH-'. No model is involved."""
from __future__ import annotations

import numpy as np

from .config import load_config
from .data.loaders import Item, assign_splits
from .runio import append_jsonl, run_dir, save_hidden_shard, write_json
from .scoring import score_answer


def make_synthetic_run(cfg: dict, run_id: str = "SYNTH-demo--00000000", n_per_class: int = 60, n_layers: int = 9, d: int = 24,
                       seed: int = 0):
    rng = np.random.default_rng(seed)
    rdir = run_dir(cfg, run_id)
    (rdir / "items.jsonl").write_text("")
    for ds, qtype in (("squad_v2", "contextual"), ("selfaware", "inherent")):
        items = []
        for k in range(2 * n_per_class):
            ans = k < n_per_class
            items.append(Item(id=f"{ds}:synth-{k}", dataset=ds, question=f"[synthetic] question {k}?", answerable=ans,
                              gold=[f"gold{k}"] if ans else [], context=f"[synthetic passage {k}]" if ds == "squad_v2" else None,
                              unanswerable_type=None if ans else qtype, group=f"{ds}:g{k % 60}", source="synthetic"))
        assign_splits(items, int(cfg["seed"]), cfg["splits"])
        rows, H = [], np.zeros((len(items), n_layers, d), np.float32)
        direction = rng.normal(size=d)
        for i, it in enumerate(items):
            for L in range(n_layers):
                strength = 0.0 if L < 2 else min(1.0, (L - 1) / (n_layers - 3)) * (1.3 if ds == "squad_v2" else 0.8)
                H[i, L] = rng.normal(size=d) + (1 if it.answerable else -1) * strength * direction
            right = rng.random() < (0.7 if it.answerable else 0.0)
            committed = rng.random() < (0.97 if not it.answerable else 0.99)
            resp = (it.gold[0] if (it.answerable and right) else "wrong answer") if committed else "I don't know"
            sc = score_answer(it, resp)
            conf = float(np.clip(rng.normal(0.85 if committed else 0.2, 0.1), 0, 1))   # overconfident by construction
            rows.append({**it.to_dict(), "response": resp, **sc, "verbal_confidence": round(conf, 2), "confidence_raw": str(int(conf * 100)),
                         "mean_logprob": float(-rng.gamma(2.0, 0.3 if it.answerable else 0.5)), "max_logprob": -0.01, "min_logprob": -2.0,
                         "mean_entropy": float(rng.gamma(2.0, 0.4 if it.answerable else 0.7)), "first_token_entropy": 1.0,
                         "n_answer_tokens": 5, "duration_ms": 100, "synthetic": True})
        save_hidden_shard(rdir, ds, [r["id"] for r in rows], H)
        append_jsonl(rdir / "items.jsonl", rows)
    write_json(rdir / "meta.json", {"run_id": run_id, "synthetic": True, "note": "synthetic fixture; no model was run"})
    return rdir

"""Pipeline sanity checks that run with every analysis.

These catch the ways a high probe AUROC can be an artifact: a dataset source
that aliases the label, a length cue, a train/test protocol that can overfit a
small test set (shuffled-label control), and a probe that does not transfer
across unanswerability types.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ..metrics import auroc
from .layerwise import fit_layerwise

SELFAWARE_SOURCE_CONFOUND_NOTE = (
    "In the official SelfAware dataset, unanswerable questions are almost all author-written "
    "items with source 'SelfAware', while answerable questions are harvested from SQuAD, "
    "TriviaQA, and HotpotQA. Source therefore aliases the label. Treat SQuAD 2.0 as the "
    "primary probe result; do not read a high SelfAware or pooled probe AUROC as evidence "
    "that hidden states encode answerability."
)

PRIMARY_DATASET = "squad_v2"


def _src(row: dict) -> str:
    return str(row.get("source") or "")


def source_label_crosstab(rows: list[dict]) -> dict:
    """Per-dataset counts of (source, answerable), plus whether source aliases the label."""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    out = {}
    for ds, items in by_ds.items():
        out[ds] = _crosstab_one(items)
    if len(by_ds) > 1:
        out["pooled"] = _crosstab_one(rows)
    return out


def _crosstab_one(rows: list[dict]) -> dict:
    counts: dict[str, dict[str, int]] = {}
    c = Counter((_src(r), bool(r["answerable"])) for r in rows)
    for (src, ans), n in sorted(c.items()):
        slot = counts.setdefault(src, {"answerable": 0, "unanswerable": 0})
        slot["answerable" if ans else "unanswerable"] += n
    ans_sources = sorted({src for src, slot in counts.items() if slot["answerable"]})
    unans_sources = sorted({src for src, slot in counts.items() if slot["unanswerable"]})
    overlap = sorted(set(ans_sources) & set(unans_sources))
    n_sources = len(counts)
    # A source aliases the label only when several sources are each class-pure.
    # A single shared source (SQuAD's official split name) does not distinguish classes.
    perfect = n_sources >= 2 and not overlap
    te_auroc = _source_feature_auroc(rows, "test")
    return {
        "n": len(rows),
        "counts": counts,
        "sources_answerable": ans_sources,
        "sources_unanswerable": unans_sources,
        "source_label_overlap": overlap,
        "n_sources": n_sources,
        "perfect_source_alias": perfect,
        "source_feature_test_auroc": te_auroc,
        "note": SELFAWARE_SOURCE_CONFOUND_NOTE if perfect else None,
    }


def _onehot(rows: list[dict], sources: list[str]) -> np.ndarray:
    idx = {s: i for i, s in enumerate(sources)}
    X = np.zeros((len(rows), len(sources)), dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, idx[_src(r)]] = 1.0
    return X


def _source_feature_auroc(rows: list[dict], split: str) -> float | None:
    sources = sorted({_src(r) for r in rows})
    if len(sources) < 2 or len(rows) < 8:
        return None
    y = np.array([int(bool(r["answerable"])) for r in rows])
    sp = np.array([r["split"] for r in rows])
    tr, te = sp == "train", sp == split
    if tr.sum() < 4 or te.sum() < 2 or len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        return None
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(_onehot(rows, sources)[tr], y[tr])
    return auroc(y[te], clf.decision_function(_onehot(rows, sources)[te]))


def length_baselines(rows: list[dict]) -> dict:
    """Question / prompt length as a trivial label predictor, fit on train, scored on val/test."""
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    out = {ds: _length_one(items) for ds, items in by_ds.items()}
    if len(by_ds) > 1:
        out["pooled"] = _length_one(rows)
    return out


def _length_one(rows: list[dict]) -> dict:
    y = np.array([int(bool(r["answerable"])) for r in rows])
    sp = np.array([r["split"] for r in rows])
    q_chars = np.array([len(str(r.get("question") or "")) for r in rows], float)
    q_words = np.array([len(str(r.get("question") or "").split()) for r in rows], float)
    prompt = np.array([len(str(r.get("context") or "")) + len(str(r.get("question") or "")) for r in rows], float)
    feats = {"question_chars": q_chars, "question_words": q_words, "prompt_chars": prompt}
    rec: dict = {"univariate_test_auroc": {}, "logistic": {}}
    for name, x in feats.items():
        rec["univariate_test_auroc"][name] = {
            "raw": auroc(y[sp == "test"], x[sp == "test"]),
            "negated": auroc(y[sp == "test"], -x[sp == "test"]),
        }
        rec["logistic"][name] = _fit_length_logistic(x.reshape(-1, 1), y, sp)
    X = np.column_stack([q_chars, q_words, prompt])
    rec["logistic"]["chars_words_prompt"] = _fit_length_logistic(X, y, sp)
    rec["by_class"] = {}
    for flag, lab in ((True, "answerable"), (False, "unanswerable")):
        m = y == int(flag)
        rec["by_class"][lab] = {
            "n": int(m.sum()),
            "mean_question_words": float(q_words[m].mean()) if m.any() else None,
            "mean_question_chars": float(q_chars[m].mean()) if m.any() else None,
            "mean_prompt_chars": float(prompt[m].mean()) if m.any() else None,
        }
    return rec


def _fit_length_logistic(X: np.ndarray, y: np.ndarray, sp: np.ndarray) -> dict:
    tr, va, te = sp == "train", sp == "val", sp == "test"
    empty = {"val_auroc": None, "test_auroc": None}
    if tr.sum() < 4 or len(np.unique(y[tr])) < 2:
        return empty
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0)
    clf.fit(X[tr], y[tr])
    s = clf.decision_function(X)
    return {
        "val_auroc": auroc(y[va], s[va]) if va.any() and len(np.unique(y[va])) == 2 else None,
        "test_auroc": auroc(y[te], s[te]) if te.any() and len(np.unique(y[te])) == 2 else None,
    }


def shuffled_label_control(H, y, split, *, n_shuffles: int = 20, C: float = 1.0,
                           max_iter: int = 1000, seed: int = 0) -> dict:
    """Refit the official layer-wise protocol on permuted labels.

    A real test AUROC of 1.0 that is far above this null is a genuine
    held-out separation (which may still be a source confound). A real
    number inside the null range is not trustworthy on a small test set.
    """
    H = np.asarray(H)
    y = np.asarray(y).astype(int)
    split = np.asarray(split)
    tr = split == "train"
    rng = np.random.RandomState(seed)
    vals, tests, layers = [], [], []
    skipped = 0
    for _ in range(int(n_shuffles)):
        ysh = rng.permutation(y)
        if tr.sum() < 4 or len(np.unique(ysh[tr])) < 2:
            skipped += 1
            continue
        res = fit_layerwise(H, ysh, split, kind="logistic", C=C, max_iter=max_iter, seed=seed)
        vals.append(res.val_auroc[res.best_layer])
        tests.append(res.test_auroc[res.best_layer])
        layers.append(res.best_layer)
    arr = np.asarray(tests, float)
    arr = arr[np.isfinite(arr)]
    summary = None
    if len(arr):
        summary = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n_ge_0_8": int((arr >= 0.8).sum()),
            "n_eq_1": int((np.abs(arr - 1.0) < 1e-12).sum()),
        }
    return {
        "n_shuffles_requested": int(n_shuffles),
        "n_shuffles_used": int(len(arr)),
        "n_skipped": int(skipped),
        "protocol": "layer-wise logistic on shuffled labels; best layer chosen on val; AUROC on test",
        "val_auroc_at_val_best": [float(x) for x in vals],
        "test_auroc_at_val_best": [float(x) for x in tests],
        "best_layers": [int(x) for x in layers],
        "summary": summary,
    }


def transfer_summary(results: dict) -> dict:
    """Flatten per-dataset transfer_in into a single table."""
    out = {}
    for ds, o in results.items():
        for src, info in (o.get("transfer_in") or {}).items():
            aucs = info.get("test_auroc_by_layer") or []
            finite = [(i, a) for i, a in enumerate(aucs) if a is not None and np.isfinite(a)]
            best_i, best_a = (max(finite, key=lambda t: t[1]) if finite else (None, None))
            out[f"{src}_to_{ds}"] = {
                "source": src,
                "target": ds,
                "source_best_layer": None,  # filled by caller if known
                "at_source_best_layer": info.get("at_source_best_layer"),
                "max_over_layers": best_a,
                "argmax_layer": best_i,
            }
    return out


def confound_flags(crosstab: dict) -> dict[str, str]:
    """Map dataset name -> short reason, including pooled when a subset is aliased."""
    flags = {}
    aliased = [ds for ds, rec in crosstab.items() if ds != "pooled" and rec.get("perfect_source_alias")]
    for ds in aliased:
        flags[ds] = "source_label_alias"
    if aliased and "pooled" in crosstab:
        flags["pooled"] = "includes_source_confounded_subset"
    return flags


def reporting_label(name: str, flags: dict[str, str]) -> str:
    if name == PRIMARY_DATASET and name not in flags:
        return f"{name} (primary)"
    if name in flags:
        return f"{name} ({flags[name]}; not a primary probe result)"
    return name

"""Calibration, discrimination and abstention metrics (numpy only; all tested on synthetic data).

Conventions: `score` arrays are "higher = more likely answerable / the model should answer";
`conf` arrays are probabilities in [0, 1]; NaN means missing and is dropped pairwise, never treated as zero.
"""
from __future__ import annotations

import numpy as np

__all__ = ["auroc", "ece", "reliability_bins", "brier", "coverage_accuracy", "aurc", "selective_accuracy_at",
           "abstention_stats", "abstention_decision_metrics", "pr_f1", "gating_curve", "gap_rate",
           "recognition_action_gap", "youden_threshold", "bootstrap_ci", "valid_pairs"]


def valid_pairs(*arrays):
    arrs = [np.asarray(a, dtype=float) for a in arrays]
    mask = np.ones(len(arrs[0]), dtype=bool)
    for a in arrs:
        mask &= np.isfinite(a)
    return [a[mask] for a in arrs]


def auroc(y_true, score) -> float:
    """Rank-based AUROC with average ranks for ties. NaN if only one class present."""
    y, s = valid_pairs(y_true, score)
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reliability_bins(conf, correct, n_bins: int = 10) -> list[dict]:
    c, y = valid_pairs(conf, correct)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
                    "mean_conf": float(c[m].mean()) if n else None, "accuracy": float(y[m].mean()) if n else None})
    return out


def ece(conf, correct, n_bins: int = 10) -> float:
    c, y = valid_pairs(conf, correct)
    if len(c) == 0:
        return float("nan")
    total = 0.0
    for b in reliability_bins(c, y, n_bins):
        if b["n"]:
            total += b["n"] / len(c) * abs(b["accuracy"] - b["mean_conf"])
    return float(total)


def brier(conf, correct) -> float:
    c, y = valid_pairs(conf, correct)
    return float(np.mean((c - y) ** 2)) if len(c) else float("nan")


def coverage_accuracy(score, correct):
    """Answer the most-confident items first. Returns (coverage, accuracy_among_answered) arrays, one point per k."""
    s, y = valid_pairs(score, correct)
    if len(s) == 0:
        return np.array([]), np.array([])
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    k = np.arange(1, len(y) + 1)
    return k / len(y), np.cumsum(y) / k


def aurc(score, correct) -> float:
    """Area under the risk-coverage curve (lower is better); risk = 1 - accuracy among answered."""
    cov, acc = coverage_accuracy(score, correct)
    if len(cov) == 0:
        return float("nan")
    return float(np.mean(1 - acc))


def selective_accuracy_at(score, correct, coverage: float) -> float:
    cov, acc = coverage_accuracy(score, correct)
    if len(cov) == 0:
        return float("nan")
    k = max(1, int(round(coverage * len(cov))))
    return float(acc[k - 1])


def abstention_stats(abstain, answerable, correct_if_answered) -> dict:
    """abstain: bool per item (gate says refuse). answerable: bool label. correct_if_answered: would the model's own
    answer have been correct (only meaningful for answerable items)."""
    a = np.asarray(abstain, bool)
    ans = np.asarray(answerable, bool)
    corr = np.asarray(correct_if_answered, bool) & ans
    n = len(a)
    tp = int((a & ~ans).sum())
    prec = tp / a.sum() if a.sum() else float("nan")
    rec = tp / (~ans).sum() if (~ans).sum() else float("nan")
    fa_den = corr.sum()
    false_abst = float((a & corr).sum() / fa_den) if fa_den else float("nan")
    answered = ~a
    acc = float((corr & answered).sum() / answered.sum()) if answered.sum() else float("nan")
    return {"coverage": float(answered.mean()) if n else float("nan"), "accuracy_answered": acc,
            "abstention_precision": float(prec), "abstention_recall": float(rec), "false_abstention_rate": false_abst,
            "n": int(n)}


def pr_f1(precision, recall) -> float:
    """Harmonic mean of precision and recall. NaN if either is missing or both are zero."""
    p, r = float(precision), float(recall)
    if not (np.isfinite(p) and np.isfinite(r)) or p + r == 0:
        return float("nan")
    return float(2 * p * r / (p + r))


def abstention_decision_metrics(abstain, answerable, correct_if_answered=None) -> dict:
    """Hard abstain/answer decision vs should-abstain (unanswerable).

    When `correct_if_answered` is given, also returns the coverage / accuracy_answered /
    false_abstention_rate fields from `abstention_stats`. AUROC uses the binary score
    (~abstain); for a hard decision this equals balanced accuracy.
    """
    a = np.asarray(abstain, bool)
    ans = np.asarray(answerable, bool)
    n = len(a)
    if correct_if_answered is not None:
        out = abstention_stats(a, ans, correct_if_answered)
    else:
        tp = int((a & ~ans).sum())
        prec = tp / a.sum() if a.sum() else float("nan")
        rec = tp / (~ans).sum() if (~ans).sum() else float("nan")
        out = {"n": int(n), "abstention_precision": float(prec), "abstention_recall": float(rec)}
    out["abstention_f1"] = pr_f1(out["abstention_precision"], out["abstention_recall"])
    out["abstain_rate"] = float(a.mean()) if n else float("nan")
    out["abstain_rate_on_unanswerable"] = float(a[~ans].mean()) if (~ans).any() else float("nan")
    out["abstain_rate_on_answerable"] = float(a[ans].mean()) if ans.any() else float("nan")
    out["decision_accuracy"] = float((a == ~ans).mean()) if n else float("nan")
    out["auroc_answerability"] = auroc(ans.astype(float), (~a).astype(float))
    return out


def youden_threshold(y_true, score) -> float:
    """Threshold maximizing TPR - FPR for predicting y_true (positive = answerable). Pick on val, never test."""
    y, s = valid_pairs(y_true, score)
    y = y.astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    best, thr = -2.0, float(np.median(s))
    for t in np.unique(s):
        pred = s >= t
        j = pred[y].mean() - pred[~y].mean()
        if j > best:
            best, thr = j, float(t)
    return thr


def gating_curve(score, answerable, correct_if_answered, n_points: int = 41) -> list[dict]:
    """Sweep a threshold over `score` (answer iff score >= t). Returns one stats row per threshold quantile."""
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    s, ans, corr = s[ok], np.asarray(answerable, bool)[ok], np.asarray(correct_if_answered, bool)[ok]
    if len(s) == 0:
        return []
    qs = np.unique(np.quantile(s, np.linspace(0, 1, n_points)))
    rows = []
    for t in qs:
        r = abstention_stats(s < t, ans, corr)
        r["threshold"] = float(t)
        rows.append(r)
    # threshold above the max = abstain on everything
    r = abstention_stats(np.ones(len(s), bool), ans, corr)
    r["threshold"] = float(s.max()) + 1e-9
    rows.append(r)
    return rows


def gap_rate(probe_says_unanswerable, conf, committed, correct, conf_high: float = 0.8) -> dict:
    """Internal-vs-expressed gap: probe flags 'insufficient evidence' while the model commits with high confidence.
    Returns the share of ALL items in the gap and the error rate among gap items."""
    p = np.asarray(probe_says_unanswerable, bool)
    c = np.asarray(conf, float)
    com = np.asarray(committed, bool)
    y = np.asarray(correct, bool)
    gap = p & com & np.isfinite(c) & (c >= conf_high)
    n = len(p)
    return {"gap_rate": float(gap.mean()) if n else float("nan"), "n_gap": int(gap.sum()),
            "error_rate_in_gap": float((~y[gap]).mean()) if gap.any() else float("nan"), "conf_high": conf_high}


def recognition_action_gap(probe_score, answerable, abstained, thr) -> dict:
    """Recognition = balanced accuracy of the probe at threshold `thr` (chosen on val) for telling answerable from
    unanswerable. Action = balanced accuracy of the model's own behavior (abstain iff unanswerable). Gap = recognition - action."""
    ans = np.asarray(answerable, bool)
    s = np.asarray(probe_score, float)
    ab = np.asarray(abstained, bool)
    ok = np.isfinite(s)
    s, ans, ab = s[ok], ans[ok], ab[ok]
    if ans.all() or (~ans).all():
        return {"recognition": float("nan"), "action": float("nan"), "gap": float("nan")}
    pred_ans = s >= thr
    rec = 0.5 * (pred_ans[ans].mean() + (~pred_ans[~ans]).mean())
    act = 0.5 * ((~ab[ans]).mean() + ab[~ans].mean())
    return {"recognition": float(rec), "action": float(act), "gap": float(rec - act)}


def bootstrap_ci(fn, arrays, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap over items for fn(*arrays) -> float. Returns (lo, hi); NaN-safe."""
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = fn(*[a[idx] for a in arrs])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))

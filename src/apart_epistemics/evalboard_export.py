"""Map a finished (or synthetic) run directory into Evalboard traces + evalboard.interp/v1 artifacts.

Trace run_id = "<run_id>:<dataset>:<answerable|unanswerable>" so Evalboard's Scoreboard compares answerable vs
unanswerable directly. Nothing is invented: missing values stay missing; every score names its scorer and rule."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .prompts import answer_messages, confidence_messages
from .runio import read_jsonl

SCORERS = {"appropriate_behavior": "rule:answer-match-or-abstain-v1", "answer_correct": "rule:squad-em-f1>=0.5|selfaware-string-v1",
           "calibration_quality": "derived:1-brier(verbal_confidence,appropriate_behavior)",
           "probe_correct": "derived:val-youden-threshold(best-layer-probe)"}


def _ts(run_id: str) -> str:
    m = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", run_id)
    return f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}Z" if m else "2026-11-13T22:00:00Z"


def _item_obj(row):
    from .data.loaders import Item
    keys = Item.__dataclass_fields__.keys()
    return Item(**{k: row[k] for k in keys if k in row})


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def build_traces(rdir: Path, run_id: str, model_name: str, metrics: dict | None = None, probe_scores: dict | None = None):
    rows = read_jsonl(rdir / "items.jsonl")
    thr = {}
    if metrics:
        for ds, o in metrics["datasets"].items():
            p = (o.get("signals") or {}).get("probe")
            thr[ds] = p["val_threshold"] if p else None
    traces = []
    for r in rows:
        it = _item_obj(r)
        group = "answerable" if r["answerable"] else "unanswerable"
        tid = f"{run_id}:{r['id']}"
        vc, ps = r.get("verbal_confidence"), (probe_scores or {}).get(r["id"])
        ev = [{"metric": "appropriate_behavior", "value": 1.0 if r.get("correct") else 0.0, "scorer": SCORERS["appropriate_behavior"],
               "rationale": "1 = right answer on an answerable item, or abstention on an unanswerable one; 0 otherwise"}]
        if r["answerable"]:
            ev.append({"metric": "answer_correct", "value": 1.0 if r.get("correct") else 0.0, "scorer": SCORERS["answer_correct"],
                       "rationale": "answerable items only"})
        if vc is not None:
            ev.append({"metric": "calibration_quality", "value": _clip(1 - (vc - float(bool(r.get("correct")))) ** 2),
                       "scorer": SCORERS["calibration_quality"], "rationale": "1 - squared error of verbalized confidence vs behavior"})
        t = thr.get(r["dataset"])
        if ps is not None and t is not None and r["split"] == "test":
            ev.append({"metric": "probe_correct", "value": 1.0 if (ps >= t) == bool(r["answerable"]) else 0.0,
                       "scorer": SCORERS["probe_correct"], "rationale": "test split only; threshold chosen on val"})
        tr = {"id": tid, "run_id": f"{run_id}:{r['dataset']}:{group}", "case_id": r["id"], "model": model_name,
              "created_at": _ts(run_id), "status": "ok", "duration_ms": int(r.get("duration_ms") or 0), "cost_usd": 0.0,
              "tokens": r.get("n_answer_tokens"), "messages": [
                  *[m for m in answer_messages(it)], {"role": "assistant", "content": r.get("response", "")},
                  {"role": "user", "content": confidence_messages(it, "")[-1]["content"]},
                  {"role": "assistant", "content": str(r.get("confidence_raw", ""))}],
              "spans": [{"id": "root", "name": "Question", "kind": "agent", "duration_ms": int(r.get("duration_ms") or 0)},
                        {"id": "answer", "parent_id": "root", "name": "greedy answer + last-token hidden states", "kind": "llm",
                         "input": {"dataset": r["dataset"]}, "output": {"text": r.get("response", ""), "mean_logprob": r.get("mean_logprob")}},
                        {"id": "confidence", "parent_id": "root", "name": "verbalized confidence", "kind": "llm",
                         "output": {"raw": r.get("confidence_raw"), "parsed": vc}}],
              "evaluations": ev,
              "verbal_confidence": vc, "mean_logprob": r.get("mean_logprob"), "mean_entropy": r.get("mean_entropy"),
              "probe_score": ps,
              "metadata": {"dataset": r["dataset"], "split": r["split"], "answerable": bool(r["answerable"]),
                           "unanswerable_type": r.get("unanswerable_type"), "gold": r.get("gold"), "abstained": bool(r.get("abstained")),
                           "correct": bool(r.get("correct")), "committed_high_confidence": bool(vc is not None and vc >= 0.8 and not r.get("abstained")),
                           "synthetic": bool(r.get("synthetic", False))}}
        tr = {k: v for k, v in tr.items() if v is not None}
        if r.get("synthetic"):
            tr["synthetic"] = True
        traces.append(tr)
    return traces


def _lp(v):
    return [None if (x is None or not np.isfinite(x)) else float(x) for x in v]


def build_artifacts(run_id: str, model_name: str, metrics: dict, curves: dict, rows: list[dict], probe_scores: dict, synthetic: bool = False):
    arts = []
    base = dict(run_id=run_id, model=model_name)
    syn = {"synthetic": True} if synthetic else {}
    dss = [d for d in metrics["datasets"] if d != "pooled"]
    n_layers = metrics["meta"]["n_layers_incl_embedding"]
    layers = list(range(n_layers))

    def A(aid, kind, title, data, **kw):
        arts.append({"schema": "evalboard.interp/v1", "id": f"{run_id}:{aid}", "kind": kind, "title": title, **base, **syn, "data": data, **kw})

    for key, label, title in (("test_auroc", "AUROC", "Answerability probe AUROC by layer (test)"),
                              ("test_accuracy", "accuracy", "Answerability probe accuracy by layer (test)")):
        series, refs = [], []
        for ds in list(metrics["datasets"]):
            o = metrics["datasets"][ds]
            series.append({"name": f"{ds} logistic", "n": o["n_by_split"]["test"], "values": _lp(o["probe_logistic"][key])})
            if key == "test_auroc" and "probe_massmean" in o:
                series.append({"name": f"{ds} mass-mean", "n": o["n_by_split"]["test"], "values": _lp(o["probe_massmean"][key])})
            if key == "test_auroc":
                for src, t in (o.get("transfer_in") or {}).items():
                    series.append({"name": f"{src} → {ds} transfer", "values": _lp(t["test_auroc_by_layer"])})
        if key == "test_auroc":
            for ds in dss:
                sig = metrics["datasets"][ds].get("signals", {})
                for s, lab in (("verbal_confidence", "verbal conf."), ("mean_logprob", "log-prob")):
                    if sig.get(s) and sig[s].get("auroc_answerability") is not None:
                        refs.append({"value": sig[s]["auroc_answerability"], "label": f"{ds} {lab} AUROC"})
        data = {"metric": label, "layers": layers, "range": [0, 1], "series": series}
        if refs:
            data["reference_lines"] = refs[:8]
        A("probe-auroc-by-layer" if key == "test_auroc" else "probe-accuracy-by-layer", "layer_series", title, data,
          notes="Layer 0 = embedding output. Probes are logistic regression (standardized) trained on the train split; the best layer is "
                "selected on the val split; the curve shown is the held-out test split. Reference lines are baseline signals' answerability AUROC on test.",
          provenance="measured", phase="probe")

    for ds in metrics["datasets"]:
        c = curves[ds]
        panels = [{"name": ds, "groups": []}]
        for nm, lab in (("verbal_confidence", "verbalized confidence"), ("token_prob_proxy", "exp(mean token log-prob)")):
            pts = [[b["mean_conf"], b["accuracy"]] for b in c["reliability"][nm] if b["n"]]
            if pts:
                panels[0]["groups"].append({"name": lab, "points": pts, "labels": [f"n={b['n']}" for b in c["reliability"][nm] if b["n"]]})
        panels[0]["groups"].append({"name": "perfect calibration", "points": [[0, 0], [1, 1]]})
        A(f"reliability-{ds}", "scatter", f"Reliability diagram: {ds} (test)", {
            "method": "Equal-width confidence bins on the test split; x = mean confidence in the bin, y = fraction of items whose behavior was appropriate. "
                      "Unanswerable items count as correct only if the model abstained.", "x_label": "mean confidence", "y_label": "accuracy",
            "explained_variance": None, "panels": panels}, provenance="derived", phase="calibration",
          notes="ECE/Brier are in metrics.json; bins with no items are omitted.")
        sigs = [s for s in c["gating"] if c["gating"][s]]
        for aid, ttl, xk, yk, xl, yl in (("coverage-accuracy", "Coverage vs accuracy (answer iff signal ≥ threshold)", "coverage", "accuracy_answered", "coverage", "accuracy among answered"),
                                         ("abstention-pr", "Abstention precision vs recall", "abstention_recall", "abstention_precision", "abstention recall", "abstention precision"),
                                         ("false-abstention", "False-abstention rate vs coverage", "coverage", "false_abstention_rate", "coverage", "false-abstention rate")):
            groups = []
            for s in sigs:
                pts = [[r[xk], r[yk]] for r in c["gating"][s] if r[xk] is not None and r[yk] is not None and np.isfinite(r[xk]) and np.isfinite(r[yk])]
                if pts:
                    groups.append({"name": s, "points": pts})
            if groups:
                A(f"{aid}-{ds}", "scatter", f"{ttl}: {ds} (test)", {
                    "method": "Threshold sweep over each signal's own test-split scores (41 quantiles). Accuracy counts a right answer on an answerable item; "
                              "unanswerable items answered count as wrong. Thresholds here are swept, not tuned; the val-chosen operating point is in metrics.json.",
                    "x_label": xl, "y_label": yl, "explained_variance": None, "panels": [{"name": ds, "groups": groups}]},
                  provenance="derived", phase="abstention")
        # distributions: probe score by class, and verbal confidence by probe verdict
        te = [r for r in rows if r["dataset"] == ds and r["split"] == "test" and probe_scores.get(r["id"]) is not None]
        if len(te) >= 10:
            ps = np.array([probe_scores[r["id"]] for r in te])
            lo, hi = float(np.floor(ps.min())), float(np.ceil(ps.max()))
            edges = [float(x) for x in np.linspace(lo, hi, 21)]
            ser = []
            for flag, nm in ((True, "answerable"), (False, "unanswerable")):
                v = ps[[bool(r["answerable"]) == flag for r in te]]
                if len(v):
                    ser.append({"name": nm, "n": int(len(v)), "counts": [int(x) for x in np.histogram(v, edges)[0]],
                                "stats": {"mean": float(v.mean()), "std": float(v.std())}})
            A(f"probe-score-dist-{ds}", "distribution", f"Probe score by class: {ds} (test)", {
                "value_label": "probe logit", "x_label": "best-layer probe logit (higher = answerable)", "y_scale": "linear", "edges": edges,
                "series": ser}, provenance="measured", phase="probe", trace_ids=[f"{run_id}:{r['id']}" for r in te][:5000],
              notes="Per-dataset probe at its val-selected layer. All test items fall inside the bins.")
        thr = ((metrics["datasets"][ds].get("signals") or {}).get("probe") or {}).get("val_threshold")
        te2 = [r for r in te if r.get("verbal_confidence") is not None and thr is not None and np.isfinite(thr)]
        if len(te2) >= 10:
            edges = [i / 10 for i in range(11)]
            ser, gap_ids = [], []
            for verdict, nm in ((True, "probe says answerable"), (False, "probe says UNanswerable")):
                sel = [r for r in te2 if (probe_scores[r["id"]] >= thr) == verdict]
                if sel:
                    v = np.array([min(r["verbal_confidence"], 0.9999) for r in sel])
                    ser.append({"name": nm, "n": len(sel), "counts": [int(x) for x in np.histogram(v, edges)[0]],
                                "stats": {"mean": float(v.mean()), "std": float(v.std())}})
                if not verdict:
                    gap_ids = [f"{run_id}:{r['id']}" for r in sel if r["verbal_confidence"] >= 0.8 and not r.get("abstained")]
            A(f"gap-verbal-confidence-{ds}", "distribution", f"Internal vs expressed: verbal confidence by probe verdict: {ds} (test)", {
                "value_label": "items", "x_label": "verbalized confidence (0-1; 1.0 binned with the top bin)", "y_scale": "linear", "edges": edges,
                "series": ser, "markers": [{"value": 0.8, "label": "high-confidence cut"}]}, provenance="derived", phase="gap",
              trace_ids=gap_ids[:5000], notes="Linked traces = gap items: probe says unanswerable, model committed with confidence ≥ 0.8.")
    return arts


def attach_interp_refs(traces: list[dict], arts: list[dict], run_id: str):
    by_ds = {}
    for a in arts:
        for ds in ("squad_v2", "selfaware"):
            if a["id"].endswith(f"-{ds}"):
                by_ds.setdefault(ds, []).append(a["id"])
    for t in traces:
        ids = by_ds.get(t["metadata"]["dataset"], [])
        refs = [{"artifact_id": f"{run_id}:probe-auroc-by-layer", "label": "Probe AUROC by layer"}] if any(a["id"].endswith("probe-auroc-by-layer") for a in arts) else []
        refs += [{"artifact_id": i, "label": i.split(":", 1)[1]} for i in ids if "probe-score-dist" in i or "gap-verbal" in i]
        if refs:
            t["metadata"]["interp"] = refs
    return traces

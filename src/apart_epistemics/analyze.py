"""Stage 2 (CPU only): probes + baselines + calibration + gating + gap metrics from a finished generate run."""
from __future__ import annotations

import numpy as np

from .metrics import (abstention_stats, auroc, aurc, bootstrap_ci, ece, gap_rate, gating_curve, recognition_action_gap,
                      reliability_bins, selective_accuracy_at, youden_threshold, brier)
from .probing import fit_layerwise, project_scores
from .probing.sanity import (PRIMARY_DATASET, SELFAWARE_SOURCE_CONFOUND_NOTE, confound_flags,
                             length_baselines, reporting_label, shuffled_label_control,
                             source_label_crosstab, transfer_summary)
from .runio import load_hidden, read_jsonl, run_dir, write_json

SIGNALS = ("probe", "verbal_confidence", "mean_logprob", "neg_entropy", "consistency")


def _f(rows, key):
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows], float)


def align(rows, ids, H):
    pos = {i: k for k, i in enumerate(ids)}
    keep = [r for r in rows if r["id"] in pos]
    return keep, H[[pos[r["id"]] for r in keep]]


def build_arrays(rows):
    ans = np.array([bool(r["answerable"]) for r in rows])
    abst = np.array([bool(r.get("abstained")) for r in rows])
    correct = np.array([bool(r.get("correct")) for r in rows])        # right answer, or abstained on unanswerable
    right_answer = correct & ans                                      # model's own answer was right (answerable only)
    return {"ans": ans, "abst": abst, "correct": correct, "right": right_answer,
            "split": np.array([r["split"] for r in rows]), "ds": np.array([r["dataset"] for r in rows]),
            "signals": {"verbal_confidence": _f(rows, "verbal_confidence"), "mean_logprob": _f(rows, "mean_logprob"),
                        "neg_entropy": -_f(rows, "mean_entropy"), "consistency": _f(rows, "consistency")}}


def _logprob_to_conf(lp):
    return np.exp(lp)  # mean token prob proxy in (0, 1]


def analyze_dataset(name, A, H, cfg, extra_models=None):
    m = A["ds"] == name if name != "pooled" else np.ones(len(A["ds"]), bool)
    pc = cfg["probe"]
    out: dict = {"n": int(m.sum())}
    sp = A["split"][m]
    y = A["ans"][m]
    out["n_by_split"] = {s: int((sp == s).sum()) for s in ("train", "val", "test")}
    res = {}
    kinds = ["logistic"] + (["massmean"] if pc.get("mass_mean", True) else [])
    for kind in kinds:
        res[kind] = fit_layerwise(H[m], y, sp, kind=kind, C=float(pc.get("C", 1.0)), max_iter=int(pc.get("max_iter", 1000)),
                                  seed=int(cfg["seed"]))
        out[f"probe_{kind}"] = res[kind].summary()
    main = res["logistic"]
    ps = np.full(len(A["ans"]), np.nan)
    ps[m] = project_scores(main, H[m])
    sig = dict(A["signals"])
    sig["probe"] = ps
    te, va = m & (A["split"] == "test"), m & (A["split"] == "val")
    ans, right, abst = A["ans"], A["right"], A["abst"]

    out["behavior"] = {
        "answer_accuracy_on_answerable": float(right[te & ans].mean()) if (te & ans).any() else None,
        "abstention_rate_on_unanswerable": float(abst[te & ~ans].mean()) if (te & ~ans).any() else None,
        "abstention_rate_on_answerable": float(abst[te & ans].mean()) if (te & ans).any() else None,
    }
    nb = int(cfg["metrics"].get("ece_bins", 10))
    base: dict = {}
    for s in SIGNALS:
        v = sig[s]
        if np.isfinite(v[te]).sum() < 5:
            base[s] = None
            continue
        thr = youden_threshold(ans[va], v[va]) if np.isfinite(v[va]).sum() > 4 else float("nan")
        rec = {"auroc_answerability": auroc(ans[te], v[te]), "auroc_correctness": auroc(right[te & ans], v[te & ans]),
               "aurc": aurc(v[te], A["correct"][te] & ans[te]), "val_threshold": thr}
        ci = bootstrap_ci(lambda a, b: auroc(a, b), [ans[te], v[te]], n_boot=int(cfg["metrics"].get("bootstrap", 300)),
                          seed=int(cfg["seed"]))
        rec["auroc_answerability_ci95"] = list(ci)
        if np.isfinite(thr):
            rec["gating_at_val_threshold"] = abstention_stats(v[te] < thr, ans[te], right[te])
        for c in (0.5, 0.8):
            rec[f"selective_accuracy_at_{int(c * 100)}pct_coverage"] = selective_accuracy_at(v[te], right[te] & ans[te], c)
        base[s] = rec
    out["signals"] = base
    # calibration of expressed confidence (verbalized + exp(mean logprob) proxy)
    vc = sig["verbal_confidence"]
    cal = {}
    for nm, cv in (("verbal_confidence", vc), ("token_prob_proxy", _logprob_to_conf(sig["mean_logprob"]))):
        cal[nm] = {"ece": ece(cv[te], A["correct"][te], nb), "brier": brier(cv[te], A["correct"][te]),
                   "mean_conf": float(np.nanmean(cv[te])) if np.isfinite(cv[te]).any() else None,
                   "accuracy": float(A["correct"][te].mean()) if te.any() else None,
                   "ece_answered_only": ece(cv[te & ~abst], A["correct"][te & ~abst], nb)}
    out["calibration"] = cal
    # internal-vs-expressed gap
    thr = base["probe"]["val_threshold"] if base.get("probe") else float("nan")
    if np.isfinite(thr):
        says_un = ps < thr
        out["gap"] = {**gap_rate(says_un[te], vc[te], ~abst[te], A["correct"][te], conf_high=float(cfg["metrics"].get("conf_high", 0.8))),
                      "among_unanswerable": gap_rate(says_un[te & ~ans], vc[te & ~ans], ~abst[te & ~ans], A["correct"][te & ~ans],
                                                     conf_high=float(cfg["metrics"].get("conf_high", 0.8))),
                      "recognition_action": recognition_action_gap(ps[te], ans[te], abst[te], thr)}
    curves = {"reliability": {nm: reliability_bins(cv[te], A["correct"][te], nb)
                              for nm, cv in (("verbal_confidence", vc), ("token_prob_proxy", _logprob_to_conf(sig["mean_logprob"])))},
              "gating": {s: gating_curve(sig[s][te], ans[te], right[te]) for s in SIGNALS if base.get(s)}}
    # transfer: models fit on another dataset, scored on this dataset's test split
    if extra_models:
        out["transfer_in"] = {}
        for src, r in extra_models.items():
            if src == name:
                continue
            aucs = []
            for L in range(H.shape[1]):
                if L in r.models:
                    mu, sd, w, b = r.models[L]
                    aucs.append(auroc(ans[te], (((H[te, L, :].astype(float) - mu) / sd) @ w + b)))
                else:
                    aucs.append(float("nan"))
            out["transfer_in"][src] = {"test_auroc_by_layer": aucs, "at_source_best_layer": aucs[r.best_layer]}
    return out, curves, ps, res["logistic"]


def run_analyze(cfg: dict, run_id: str) -> dict:
    print("[apart] stage=analyze start", flush=True)
    rdir = run_dir(cfg, run_id)
    rows = read_jsonl(rdir / "items.jsonl")
    ids, H = load_hidden(rdir)
    if not rows or H.size == 0:
        raise SystemExit(f"no generated items/hidden states in {rdir}")
    rows, H = align(rows, ids, H)
    A = build_arrays(rows)
    names = [d for d in cfg["datasets"] if (A["ds"] == d).any()]
    results, curves, ps_all, mains = {}, {}, np.full(len(rows), np.nan), {}
    # first pass fits per-dataset probes (needed for transfer), second computes everything
    for nm in names:
        m = A["ds"] == nm
        mains[nm] = fit_layerwise(H[m], A["ans"][m], A["split"][m], kind="logistic", C=float(cfg["probe"].get("C", 1.0)),
                                  max_iter=int(cfg["probe"].get("max_iter", 1000)), seed=int(cfg["seed"]))
    for nm in names + (["pooled"] if len(names) > 1 else []):
        o, c, ps, _ = analyze_dataset(nm, A, H, cfg, extra_models=mains)
        results[nm], curves[nm] = o, c
        if nm != "pooled":
            m = A["ds"] == nm
            ps_all[m] = ps[m]
    sanity = _run_sanity(rows, A, H, cfg, results, mains)
    flags = sanity["confound_flags"]
    n_layers = int(H.shape[1])
    primary = PRIMARY_DATASET if PRIMARY_DATASET in results else (names[0] if names else None)
    meta = {
        "run_id": run_id, "n_items": len(rows), "n_layers_incl_embedding": n_layers, "hidden_dim": int(H.shape[2]),
        "primary_probe_dataset": primary,
        "primary_probe_scores": (
            "SQuAD 2.0 logistic probe at its val-selected best layer is the primary result. "
            "SelfAware and pooled are reported but are not primary when source aliases the label."
        ),
        "confounded_datasets": sorted(flags),
        "confound_note": SELFAWARE_SOURCE_CONFOUND_NOTE if flags else None,
    }
    fmt = lambda v: "na" if v is None or (isinstance(v, float) and v != v) else f"{v:.3f}"  # noqa: E731
    order = ([primary] if primary else []) + [n for n in results if n != primary]
    for nm in order:
        o = results[nm]
        pl = o["probe_logistic"]
        pa = pl["test_auroc"][pl["best_layer"]]
        vc = (o["signals"].get("verbal_confidence") or {}).get("auroc_answerability")
        rag = (o.get("gap") or {}).get("recognition_action", {}).get("gap")
        print(f"headline {reporting_label(nm, flags)}: probe_auroc={fmt(pa)} best_layer={pl['best_layer']} "
              f"verbal_auroc={fmt(vc)} ece_verbal={fmt(o['calibration']['verbal_confidence']['ece'])} "
              f"recognition_action_gap={fmt(rag)}", flush=True)
    _print_sanity(sanity, results, flags, fmt)
    payload = {"meta": meta, "datasets": results, "sanity": sanity}
    write_json(rdir / "analysis" / "metrics.json", payload)
    write_json(rdir / "analysis" / "sanity.json", sanity)
    write_json(rdir / "analysis" / "curves.json", curves)
    from .runio import append_jsonl
    pf = rdir / "analysis" / "probe_scores.jsonl"
    pf.write_text("")
    append_jsonl(pf, [{"id": r["id"], "probe_score": None if not np.isfinite(s) else float(s)} for r, s in zip(rows, ps_all)])
    return payload


def _run_sanity(rows, A, H, cfg, results, mains) -> dict:
    pc = cfg["probe"]
    n_shuf = int(cfg.get("metrics", {}).get("n_shuffles", 20))
    C, max_iter, seed = float(pc.get("C", 1.0)), int(pc.get("max_iter", 1000)), int(cfg["seed"])
    crosstab = source_label_crosstab(rows)
    flags = confound_flags(crosstab)
    length = length_baselines(rows)
    shuffled = {}
    scopes = list(results.keys())
    for nm in scopes:
        m = np.ones(len(A["ds"]), bool) if nm == "pooled" else (A["ds"] == nm)
        rec = shuffled_label_control(H[m], A["ans"][m].astype(int), A["split"][m],
                                     n_shuffles=n_shuf, C=C, max_iter=max_iter, seed=seed)
        pl = results[nm]["probe_logistic"]
        rec["real_test_auroc_at_val_best"] = pl["test_auroc"][pl["best_layer"]]
        rec["real_best_layer"] = pl["best_layer"]
        shuffled[nm] = rec
    transfer = transfer_summary(results)
    for key, rec in transfer.items():
        src = rec["source"]
        if src in mains:
            rec["source_best_layer"] = int(mains[src].best_layer)
    return {
        "source_label_crosstab": crosstab,
        "shuffled_label": shuffled,
        "length_baseline": length,
        "transfer": transfer,
        "confound_flags": flags,
    }


def _print_sanity(sanity, results, flags, fmt):
    xt = sanity["source_label_crosstab"]
    for ds, rec in xt.items():
        alias = "perfect source/label alias" if rec.get("perfect_source_alias") else "no source/label alias"
        print(f"sanity source-crosstab {reporting_label(ds, flags)}: sources={rec.get('n_sources')} "
              f"overlap={rec.get('source_label_overlap')} {alias} "
              f"source_feature_test_auroc={fmt(rec.get('source_feature_test_auroc'))}", flush=True)
    for ds, rec in sanity["shuffled_label"].items():
        s = rec.get("summary") or {}
        print(f"sanity shuffled-label {reporting_label(ds, flags)}: n={s.get('n', 0)} "
              f"null_test_auroc mean={fmt(s.get('mean'))} max={fmt(s.get('max'))} "
              f"real={fmt(rec.get('real_test_auroc_at_val_best'))}", flush=True)
    for ds, rec in sanity["length_baseline"].items():
        logistic = rec.get("logistic") or {}
        words = (logistic.get("question_words") or {}).get("test_auroc")
        combo = (logistic.get("chars_words_prompt") or {}).get("test_auroc")
        print(f"sanity length-baseline {reporting_label(ds, flags)}: "
              f"words_logistic_test_auroc={fmt(words)} combo_test_auroc={fmt(combo)}", flush=True)
    for key, rec in sanity["transfer"].items():
        print(f"sanity transfer {rec['source']} -> {reporting_label(rec['target'], flags)}: "
              f"at_source_best_layer={fmt(rec.get('at_source_best_layer'))} "
              f"max_over_layers={fmt(rec.get('max_over_layers'))} at L={rec.get('argmax_layer')}", flush=True)

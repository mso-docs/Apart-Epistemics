"""Post hoc percentile-bootstrap intervals for quantities whose intervals are not stored in metrics.json.
Resamples the 240 SQuAD test items with replacement (300 resamples, seed 13, same scheme as the run's bootstrap);
the validation-chosen probe threshold is held fixed. Prints JSON."""
import json, sys, numpy as np
from pathlib import Path
R = Path(sys.argv[1])
M = json.load(open(R/"analysis/metrics.json"))
rows = [json.loads(l) for l in open(R/"items.jsonl")]
ps = {json.loads(l)["id"]: json.loads(l)["probe_score"] for l in open(R/"analysis/probe_scores.jsonl")}
sq = [r for r in rows if r["dataset"] == "squad_v2" and r["split"] == "test"]
y = np.array([r["answerable"] for r in sq]); p = np.array([ps[r["id"]] for r in sq])
vc = np.array([r["verbal_confidence"] for r in sq], float); ab = np.array([r["abstained"] for r in sq]); co = np.array([r["correct"] for r in sq])
ao = np.array([r["abstained_when_allowed"] for r in sq])
thr = M["datasets"]["squad_v2"]["signals"]["probe"]["val_threshold"]
def stats(i):
    yy, pp, vv, aa, cc, oo = y[i], p[i], vc[i], ab[i], co[i], ao[i]
    says = pp < thr; gap = says & ~aa & (vv >= 0.8)
    pred_ans = pp >= thr
    rec = 0.5*(pred_ans[yy].mean()+(~pred_ans[~yy]).mean()); act = 0.5*((~aa[yy]).mean()+aa[~yy].mean())
    act_o = 0.5*((~oo[yy]).mean()+oo[~yy].mean())
    return {"gap_rate": gap.mean(), "error_in_gap": (~cc[gap]).mean() if gap.any() else np.nan,
            "recognition": rec, "action_plain": act, "ra_gap_plain": rec-act, "ra_gap_with_option": rec-act_o,
            "option_recall": (oo & ~yy).sum()/(~yy).sum(), "option_precision": (oo & ~yy).sum()/oo.sum() if oo.any() else np.nan,
            "probe_share_flagged_in_gap": gap.sum()/max(1, says.sum())}
rng = np.random.default_rng(13); draws = []
for _ in range(300):
    draws.append(stats(rng.integers(0, len(y), len(y))))
pt = stats(np.arange(len(y))); out = {}
for k in pt:
    v = np.array([d[k] for d in draws], float); v = v[np.isfinite(v)]
    out[k] = {"point": float(pt[k]), "lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975)), "n_valid": int(len(v))}
print(json.dumps(out, indent=1))

"""Re-derive the headline numbers from the run's raw files (items.jsonl, probe_scores.jsonl) and compare with
analysis/metrics.json. Prints PASS/FAIL per check. Uses only numpy; no model or hidden states needed."""
import json, sys, numpy as np
from pathlib import Path
R = Path(sys.argv[1])
M = json.load(open(R/"analysis/metrics.json")); S = json.load(open(R/"analysis/sanity.json"))
rows = [json.loads(l) for l in open(R/"items.jsonl")]
ps = {json.loads(l)["id"]: json.loads(l)["probe_score"] for l in open(R/"analysis/probe_scores.jsonl")}

def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    ok = np.isfinite(s); y, s = y[ok], s[ok]
    n1, n0 = y.sum(), (~y).sum()
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ss = s[order]; i = 0
    while i < len(ss):
        j = i
        while j+1 < len(ss) and ss[j+1] == ss[i]: j += 1
        ranks[order[i:j+1]] = (i+j)/2+1; i = j+1
    return (ranks[y].sum()-n1*(n1+1)/2)/(n1*n0)

def chk(name, got, want, tol=1e-6):
    ok = abs(got-want) <= tol
    print(("PASS " if ok else "FAIL ")+f"{name}: recomputed={got:.6f} json={want:.6f}")

sq = [r for r in rows if r["dataset"] == "squad_v2" and r["split"] == "test"]
y = np.array([r["answerable"] for r in sq]); p = np.array([ps[r["id"]] for r in sq])
vc = np.array([np.nan if r["verbal_confidence"] is None else r["verbal_confidence"] for r in sq])
ab = np.array([r["abstained"] for r in sq]); co = np.array([r["correct"] for r in sq])
cons = np.array([np.nan if r.get("consistency") is None else r["consistency"] for r in sq])
sg = M["datasets"]["squad_v2"]["signals"]; gp = M["datasets"]["squad_v2"]["gap"]
print("squad test n =", len(sq), "answerable", int(y.sum()), "unanswerable", int((~y).sum()))
chk("probe AUROC", auroc(y, p), sg["probe"]["auroc_answerability"])
chk("verbal AUROC", auroc(y, vc), sg["verbal_confidence"]["auroc_answerability"])
chk("consistency AUROC", auroc(y, cons), sg["consistency"]["auroc_answerability"])
print("consistency n finite:", int(np.isfinite(cons).sum()), "answerable among them", int(y[np.isfinite(cons)].sum()))
print("verbal confidence missing:", int(np.isnan(vc).sum()))
thr = sg["probe"]["val_threshold"]
says_un = p < thr
com = ~ab
gap = says_un & com & np.isfinite(vc) & (vc >= 0.8)
chk("gap rate", gap.mean(), gp["gap_rate"]); chk("error in gap", (~co[gap]).mean(), gp["error_rate_in_gap"])
print("probe flags unanswerable:", int(says_un.sum()), "of", len(sq), "; of those committed & conf>=0.8:", int(gap.sum()),
      f"({gap.sum()/says_un.sum():.3f})")
print("flagged: truly unanswerable", int((says_un & ~y).sum()), "answerable", int((says_un & y).sum()))
print("gap items by true class: unanswerable", int((gap & ~y).sum()), "answerable", int((gap & y).sum()))
print("overall error rate (1-correct) on test:", 1-co.mean(), " error among committed:", (~co[com]).mean())
print("error rate among NON-gap items:", (~co[~gap]).mean(), "n", int((~gap).sum()))
print("answerable error among gap items:", (~co[gap & y]).mean() if (gap & y).any() else None)
print("items with committed & conf>=0.8 (any probe):", int((com & (vc >= 0.8)).sum()), " error:", (~co[com & (vc >= 0.8)]).mean())
print("share of items conf>=0.8:", float((vc >= 0.8).mean()), "; unanswerable conf>=0.8:", float((vc[~y] >= 0.8).mean()), "; answerable:", float((vc[y] >= 0.8).mean()))
print("mean verbal conf unanswerable / answerable:", np.nanmean(vc[~y]), np.nanmean(vc[y]))
print("abstained in plain prompt: unanswerable", int(ab[~y].sum()), "answerable", int(ab[y].sum()))
print("answerable accuracy:", co[y].mean(), " (n answerable", int(y.sum()), ")")
# recognition-action
pred_ans = p >= thr
rec = 0.5*(pred_ans[y].mean()+(~pred_ans[~y]).mean()); act = 0.5*((~ab[y]).mean()+ab[~y].mean())
chk("recognition", rec, gp["recognition_action"]["recognition"]); chk("action", act, gp["recognition_action"]["action"])
chk("gap", rec-act, gp["recognition_action"]["gap"])
# abstain option
ao = np.array([r["abstained_when_allowed"] for r in sq])
a = M["datasets"]["squad_v2"]["abstain_option"]["test"]
prec = (ao & ~y).sum()/ao.sum(); recl = (ao & ~y).sum()/(~y).sum()
chk("opt precision", prec, a["abstention_precision"]); chk("opt recall", recl, a["abstention_recall"])
chk("opt F1", 2*prec*recl/(prec+recl), a["abstention_f1"])
print("abstain option abstentions:", int(ao.sum()), "TP", int((ao & ~y).sum()), "of unanswerable", int((~y).sum()))
print("derived: balanced accuracy of abstain option =", 0.5*((~ao[y]).mean()+ao[~y].mean()),
      " -> recognition minus option-action =", rec-0.5*((~ao[y]).mean()+ao[~y].mean()))
# length baselines and other json fields
print("length baselines json:", {k: v["test_auroc"] for k, v in S["length_baseline"]["squad_v2"]["logistic"].items()})
print("shuffled mean/max:", S["shuffled_label"]["squad_v2"]["summary"]["mean"], S["shuffled_label"]["squad_v2"]["summary"]["max"])
print("source-only AUROC selfaware / pooled:", S["source_label_crosstab"]["selfaware"]["source_feature_test_auroc"], S["source_label_crosstab"]["pooled"]["source_feature_test_auroc"])
for k, v in S["transfer"].items(): print("transfer", k, v)
# bootstrap CI replicate (percentile, 300 resamples, seed 13)
rng = np.random.default_rng(13); vals = []
for _ in range(300):
    idx = rng.integers(0, len(y), len(y)); v = auroc(y[idx], p[idx])
    if np.isfinite(v): vals.append(v)
print("bootstrap CI replicate:", np.quantile(vals, .025), np.quantile(vals, .975), " json:", sg["probe"]["auroc_answerability_ci95"])

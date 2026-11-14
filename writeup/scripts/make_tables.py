"""Generate LaTeX tables from the run's JSON (no hand-typed numbers). Usage: make_tables.py <run_dir> <out.tex>"""
import json, sys
from pathlib import Path
R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
M = json.load(open(R/"analysis/metrics.json")); S = json.load(open(R/"analysis/sanity.json"))
PB = json.load(open(OUT.parent/"posthoc_bootstrap.json"))
f3 = lambda v: "--" if v is None or v != v else f"{v:.3f}"
f2 = lambda v: "--" if v is None or v != v else f"{v:.2f}"
def pr_f1(p, r): return float("nan") if (p is None or r is None or p+r == 0) else 2*p*r/(p+r)
L = []
sq = M["datasets"]["squad_v2"]; sg = sq["signals"]
# ---- Table 1: signals on SQuAD test
L.append(r"""\begin{table}[t]
\centering\small
\caption{Answerability signals on the SQuAD~2.0 test split (n=240; 120 answerable, 120 unanswerable). AUROC columns use the raw signal; the 95\% interval is a percentile bootstrap over test items (300 resamples). The right-hand block thresholds each signal at the cut that maximises Youden's J on the validation split, then scores flagging an item as unanswerable (precision, recall and F1 are for the unanswerable class; coverage is the share of items still answered). The explicit abstain option is a single hard decision, so its AUROC equals its balanced accuracy and it has no threshold. Sampling consistency exists for only 51 test items (23 answerable), so its AUROC is not comparable with the others and its thresholded columns are omitted (items without a score would be silently counted as answered).}
\label{tab:signals}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lccccccc@{}}
\toprule
Signal & AUROC (answerability) & 95\% CI & AUROC (correctness) & Coverage & Prec. & Recall & F1 \\
\midrule""")
names = [("probe", "Linear probe, L18"), ("verbal_confidence", "Verbal confidence"), ("mean_logprob", "Mean token log-prob"), ("neg_entropy", "Negative entropy")]
for k, lab in names:
    r = sg[k]; g = r["gating_at_val_threshold"]
    L.append(f"{lab} & {f3(r['auroc_answerability'])} & [{f3(r['auroc_answerability_ci95'][0])}, {f3(r['auroc_answerability_ci95'][1])}] & {f3(r['auroc_correctness'])} & {f3(g['coverage'])} & {f3(g['abstention_precision'])} & {f3(g['abstention_recall'])} & {f3(pr_f1(g['abstention_precision'], g['abstention_recall']))} \\\\")
r = sg["consistency"]
L.append(f"Sampling consistency (n=51) & {f3(r['auroc_answerability'])} & [{f3(r['auroc_answerability_ci95'][0])}, {f3(r['auroc_answerability_ci95'][1])}] & {f3(r['auroc_correctness'])}$^{{\\dagger}}$ & -- & -- & -- & -- \\\\")
a = sq["abstain_option"]["test"]
L.append(f"Explicit abstain option & {f3(a['auroc_answerability'])} & -- & -- & {f3(a['coverage'])} & {f3(a['abstention_precision'])} & {f3(a['abstention_recall'])} & {f3(a['abstention_f1'])} \\\\")
ls = S["length_baseline"]["squad_v2"]["logistic"]
L.append(f"Question word count (logistic) & {f3(ls['question_words']['test_auroc'])} & -- & -- & -- & -- & -- & -- \\\\")
sh = S["shuffled_label"]["squad_v2"]["summary"]
L.append(f"Shuffled-label probe (mean of 20) & {f3(sh['mean'])} & -- & -- & -- & -- & -- & -- \\\\")
L.append(r"""\bottomrule
\end{tabular}}
\par\vspace{2pt}{\footnotesize $^\dagger$ Computed on the 23 answerable test items that have a consistency score, only one of which was answered wrongly, so this value rests on 22 pairs and is not reliable. The correctness AUROC of the other signals uses the 120 answerable test items, 12 of which were answered wrongly.}
\end{table}
""")
# ---- Table: per-layer AUROC (appendix)
pl, pm = sq["probe_logistic"], sq["probe_massmean"]
L.append(r"""\begin{table}[p]
\centering\scriptsize
\caption{Layer-wise probe AUROC on SQuAD~2.0 (answerable = positive). Layer 0 is the embedding output, where the last-token input is identical for every item, so 0.500 is expected. Validation n=240, test n=240. The layer chosen on validation is marked with an asterisk for each probe type; the layer with the highest test AUROC is marked with a dagger.}
\label{tab:layers}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}rcccc@{\hspace{2em}}rcccc@{}}
\toprule
Layer & Log. val & Log. test & MM val & MM test & Layer & Log. val & Log. test & MM val & MM test \\
\midrule""")
import numpy as np
bl, bm = pl["best_layer"], pm["best_layer"]; tl, tm = int(np.argmax(pl["test_auroc"])), int(np.argmax(pm["test_auroc"]))
def cell(arr, i, mark_sel, mark_best):
    s = f3(arr[i]) + ("$^*$" if mark_sel else "") + ("$^\\dagger$" if mark_best else "")
    return s
for i in range(17):
    j = i + 17
    parts = []
    for ii in (i, j):
        if ii > 32: parts.append("&&&&"); continue
        parts.append(f"{ii} & {cell(pl['val_auroc'], ii, ii==bl, False)} & {cell(pl['test_auroc'], ii, False, ii==tl)} & {cell(pm['val_auroc'], ii, ii==bm, False)} & {cell(pm['test_auroc'], ii, False, ii==tm)}")
    L.append(" & ".join(parts) + r" \\")
L.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")
# ---- Table: all datasets summary (appendix)
def ds_row(nm, lab):
    o = M["datasets"][nm]; p = o["signals"]["probe"]; v = o["signals"]["verbal_confidence"]; c = o["signals"]["consistency"]
    pl_ = o["probe_logistic"]; shn = S["shuffled_label"][nm]["summary"]; xs = S["source_label_crosstab"][nm]["source_feature_test_auroc"]
    ln = S["length_baseline"][nm]["logistic"]["question_words"]["test_auroc"]
    return (f"{lab} & {o['n_by_split']['test']} & L{pl_['best_layer']} & {f3(pl_['val_auroc'][pl_['best_layer']])} & {f3(p['auroc_answerability'])} [{f3(p['auroc_answerability_ci95'][0])}, {f3(p['auroc_answerability_ci95'][1])}] "
            f"& {f3(v['auroc_answerability'])} & {f3(c['auroc_answerability'])} & {f3(ln)} & {f3(shn['mean'])} / {f3(shn['max'])} & {f3(xs)} \\\\")
L.append(r"""\begin{table}[t]
\centering\scriptsize
\caption{Answerability AUROC by dataset on each test split. Only SQuAD~2.0 is a primary result. In SelfAware the unanswerable items all have source \texttt{SelfAware} and the answerable items all come from other sources, so a source-only feature reaches 1.000; the pooled set inherits part of this alias (source-only 0.930, not 1.000, because the SQuAD half shares one source across both classes). ``Shuffled'' gives the mean / maximum test AUROC over 20 shuffled-label runs. Question word count is a one-feature logistic baseline. The source-only value for SQuAD is not defined (one source).}
\label{tab:datasets}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lcccccccccc@{}}
\toprule
Dataset & Test n & Layer & Val AUROC & Probe test AUROC [95\% CI] & Verbal & Consist. & Words & Shuffled & Source-only \\
\midrule""")
L.append(ds_row("squad_v2", "SQuAD 2.0 (primary)")); L.append(ds_row("selfaware", "SelfAware (confounded)")); L.append(ds_row("pooled", "Pooled (confounded)"))
L.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")
# ---- Table: gap + behaviour by dataset (appendix)
def beh(nm, lab):
    o = M["datasets"][nm]; g = o["gap"]; ra = g["recognition_action"]; b = o["behavior"]; ao_ = o["abstain_option"]["test"]; cal = o["calibration"]["verbal_confidence"]
    return (f"{lab} & {f3(b['answer_accuracy_on_answerable'])} & {f3(b['abstention_rate_on_unanswerable'])} & {f3(ao_['abstain_rate_on_unanswerable'])} & {f3(ao_['abstain_rate_on_answerable'])} & {f3(ao_['abstention_precision'])} & {f3(ao_['abstention_recall'])} & {f3(ao_['abstention_f1'])} "
            f"& {f3(g['gap_rate'])} ({g['n_gap']}) & {f3(g['error_rate_in_gap'])} & {f3(ra['recognition'])} & {f3(ra['action'])} & {f3(ra['gap'])} & {f3(cal['mean_conf'])} & {f3(cal['ece'])} \\\\")
L.append(r"""\begin{table}[t]
\centering\tiny
\caption{Behaviour and gap measures on each test split. ``Acc.'' is accuracy on answerable items (SQuAD: exact match or token-F1 $\geq$ 0.5; SelfAware: normalised string containment, which misses paraphrases). ``Abst.\ plain'' and ``Abst.\ option'' are abstention rates on unanswerable items without and with the explicit option; ``Opt.\ ans.'' is the option's abstention rate on answerable items. Gap rate is the share of all test items (count in parentheses) where the probe says unanswerable, the model commits, and its verbal confidence is at least 0.8; ``Err.\ in gap'' is the error rate among those items. Recognition and action are balanced accuracies defined in Section~\ref{sec:metrics}. ECE uses ten equal-width bins and counts an abstention on an unanswerable item as correct.}
\label{tab:behaviour}
\resizebox{\linewidth}{!}{\begin{tabular}{@{}lccccccccccccccc@{}}
\toprule
Dataset & Acc. & Abst. plain & Abst. option & Opt. ans. & Opt. prec. & Opt. recall & Opt. F1 & Gap rate (n) & Err. in gap & Recog. & Action & R--A gap & Mean verbal conf. & ECE \\
\midrule""")
L.append(beh("squad_v2", "SQuAD 2.0")); L.append(beh("selfaware", "SelfAware")); L.append(beh("pooled", "Pooled"))
L.append(r"""\bottomrule
\end{tabular}}
\end{table}
""")
# ---- Table: transfer
T = S["transfer"]
L.append(r"""\begin{table}[t]
\centering\small
\caption{Cross-dataset transfer of the logistic probe, scored by AUROC on the target test split. ``Reported'' uses the source dataset's own validation-selected layer. ``Maximum over layers'' picks the best of 33 layers on the target test split, so it is an optimistic upper bound and should not be quoted as performance. Pooled targets contain the confounded SelfAware half.}
\label{tab:transfer}
\begin{tabular}{@{}lcccc@{}}
\toprule
Transfer & Source layer & Reported & Max over layers & Layer of max \\
\midrule""")
for key, lab in (("squad_v2_to_selfaware", r"SQuAD $\to$ SelfAware"), ("selfaware_to_squad_v2", r"SelfAware $\to$ SQuAD"), ("squad_v2_to_pooled", r"SQuAD $\to$ pooled"), ("selfaware_to_pooled", r"SelfAware $\to$ pooled")):
    t = T[key]; L.append(f"{lab} & L{t['source_best_layer']} & {f3(t['at_source_best_layer'])} & {f3(t['max_over_layers'])} & L{t['argmax_layer']} \\\\")
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
# ---- Table: post hoc bootstrap
def pb(k, lab, pct=False):
    d = PB[k]; return f"{lab} & {d['point']:.3f} & [{d['lo']:.3f}, {d['hi']:.3f}] \\\\"
L.append(r"""\begin{table}[t]
\centering\small
\caption{Post hoc 95\% percentile-bootstrap intervals (300 resamples over the 240 SQuAD test items, seed 13) for quantities whose intervals the run did not store. The probe threshold is held fixed at the validation-chosen value. Computed by a separate script after the run; point estimates match \texttt{metrics.json} exactly. Resampling is over items and ignores clustering by passage, so intervals may be slightly too narrow.}
\label{tab:posthoc}
\begin{tabular}{@{}lcc@{}}
\toprule
Quantity & Point & 95\% CI \\
\midrule""")
for k, lab in (("recognition", "Recognition (probe balanced accuracy)"), ("action_plain", "Action, plain prompt"), ("ra_gap_plain", "Recognition--action gap, plain prompt"), ("ra_gap_with_option", "Recognition minus action with abstain option"), ("gap_rate", "Gap rate (share of all items)"), ("error_in_gap", "Error rate among gap items"), ("probe_share_flagged_in_gap", "Share of probe-flagged items that are gap items"), ("option_recall", "Abstain-option recall")):
    L.append(pb(k, lab))
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
# ---- Table: SelfAware crosstab
ct = S["source_label_crosstab"]["selfaware"]["counts"]
L.append(r"""\begin{table}[t]
\centering\small
\caption{Source by label counts for the 2{,}000 SelfAware items in this run (1{,}000 per class, all splits). The only source with unanswerable items has no answerable items, which is why a source-only classifier is perfect on the test split.}
\label{tab:crosstab}
\begin{tabular}{@{}lcc@{}}
\toprule
Source & Answerable & Unanswerable \\
\midrule""")
for s_, c in ct.items():
    L.append(f"\\texttt{{{s_.replace('_', chr(92)+'_')}}} & {c['answerable']} & {c['unanswerable']} \\\\")
L.append(r"""\bottomrule
\end{tabular}
\end{table}
""")
OUT.write_text("\n".join(L))
import re as _re
for _b in _re.findall(r"\\begin\{table\}.*?\\end\{table\}", "\n".join(L), _re.S):
    _lab = _re.search(r"\\label\{tab:(\w+)\}", _b).group(1)
    (OUT.parent/"tables").mkdir(exist_ok=True); (OUT.parent/"tables"/f"tab_{_lab}.tex").write_text(_b+"\n")
print("tables written", len(L))

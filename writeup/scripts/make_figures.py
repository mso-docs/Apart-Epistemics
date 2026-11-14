"""Render the report figures (vector PDF) from the run's Evalboard interp JSON and analysis JSON.
Usage: python make_figures.py <run_dir> <out_dir>"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
INT = R/"evalboard/interp"
M = json.load(open(R/"analysis/metrics.json")); S = json.load(open(R/"analysis/sanity.json"))
ld = lambda n: json.load(open(INT/f"{n}.json"))["data"]
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 150})
C = {"probe": "#1f4e9c", "mass": "#6a9fd8", "verbal": "#d9822b", "lp": "#8f8f8f", "len": "#7a4b9c", "null": "#b5b5b5",
     "opt": "#2e8b57", "red": "#c0392b"}
sq = M["datasets"]["squad_v2"]; pl = sq["probe_logistic"]; pm = sq["probe_massmean"]
layers = np.arange(33)

# Fig 1: layer-wise AUROC
d = ld("probe-auroc-by-layer"); ser = {s["name"]: s for s in d["series"]}
fig, ax = plt.subplots(figsize=(6.4, 3.0))
sh = S["shuffled_label"]["squad_v2"]["summary"]
ax.axhspan(sh["min"], sh["max"], color=C["null"], alpha=0.35, lw=0, label=f"shuffled-label null (20 runs; range {sh['min']:.2f}-{sh['max']:.2f})")
ax.plot(layers, ser["squad_v2 (primary) logistic"]["values"], color=C["probe"], lw=1.8, marker="o", ms=2.8, label="logistic probe, test")
ax.plot(layers, pl["val_auroc"], color=C["probe"], lw=1.0, ls="--", label="logistic probe, validation")
ax.plot(layers, ser["squad_v2 (primary) mass-mean"]["values"], color=C["mass"], lw=1.2, label="mass-mean probe, test")
for ref, col, lab in ((0.6205208333333333, C["verbal"], "verbal confidence 0.621"), (0.6674305555555555, C["lp"], "mean token log-prob 0.667"),
                      (0.569375, C["len"], "question word count 0.569")):
    ax.axhline(ref, color=col, lw=1.0, ls=":", label=lab)
ax.axvline(pl["best_layer"], color="k", lw=0.8, alpha=0.6); ax.text(pl["best_layer"]+0.3, 0.96, "L18 (chosen on validation)", fontsize=7, va="center")
ax.set_xlabel("residual-stream layer (0 = embedding output)"); ax.set_ylabel("AUROC, answerable vs unanswerable")
ax.set_ylim(0.4, 1.0); ax.set_xlim(0, 32); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
fig.set_size_inches(6.4, 3.3); fig.tight_layout(); fig.savefig(OUT/"fig_layers.pdf"); plt.close(fig)

# Fig 2: signals on SQuAD test (dot and whisker)
sg = sq["signals"]
rows = [("probe, L18 (logistic)", sg["probe"]["auroc_answerability"], sg["probe"]["auroc_answerability_ci95"], C["probe"], 240),
        ("verbal confidence", sg["verbal_confidence"]["auroc_answerability"], sg["verbal_confidence"]["auroc_answerability_ci95"], C["verbal"], 240),
        ("mean token log-prob", sg["mean_logprob"]["auroc_answerability"], sg["mean_logprob"]["auroc_answerability_ci95"], C["lp"], 240),
        ("negative entropy", sg["neg_entropy"]["auroc_answerability"], sg["neg_entropy"]["auroc_answerability_ci95"], C["lp"], 240),
        ("sampling consistency (n=51)", sg["consistency"]["auroc_answerability"], sg["consistency"]["auroc_answerability_ci95"], C["lp"], 51),
        ("explicit abstain option (hard decision)", sq["abstain_option"]["test"]["auroc_answerability"], None, C["opt"], 240),
        ("question word count (logistic)", S["length_baseline"]["squad_v2"]["logistic"]["question_words"]["test_auroc"], None, C["len"], 240),
        ("shuffled-label probe, mean of 20", sh["mean"], None, C["null"], 240)]
fig, ax = plt.subplots(figsize=(6.4, 2.6))
for i, (lab, v, ci, col, n) in enumerate(rows[::-1]):
    if ci: ax.plot(ci, [i, i], color=col, lw=2)
    ax.plot([v], [i], "o", color=col, ms=5)
    ax.text(1.005, i, f"{v:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""), va="center", fontsize=7.2)
ax.axvline(0.5, color="k", lw=0.7, ls="--"); ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
ax.set_xlim(0.4, 1.0); ax.set_xlabel("test AUROC for answerability (SQuAD 2.0, n=240; 95% bootstrap CI where computed)")
ax.grid(axis="y", visible=False); fig.tight_layout(); fig.subplots_adjust(right=0.74); fig.savefig(OUT/"fig_signals.pdf"); plt.close(fig)

# Fig 3: internal vs expressed
pd_ = ld("probe-score-dist-squad_v2"); gv = ld("gap-verbal-confidence-squad_v2")
thr = sg["probe"]["val_threshold"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axs[0]; e = np.array(pd_["edges"]); w = np.diff(e); x = e[:-1]
ax.bar(x, pd_["series"][0]["counts"], width=w, align="edge", color=C["probe"], alpha=0.65, label="answerable (n=120)")
ax.bar(x, pd_["series"][1]["counts"], width=w, align="edge", color=C["red"], alpha=0.55, label="unanswerable (n=120)")
ax.axvline(thr, color="k", lw=1.0); ax.text(thr+0.4, ax.get_ylim()[1]*0.9, f"validation\nthreshold {thr:.2f}", fontsize=7)
ax.set_xlabel("probe logit at L18 (higher = answerable)"); ax.set_ylabel("test items"); ax.legend(frameon=False, loc="upper left")
ax.set_title("(a) probe score by true class"); ax.set_xlim(-34, 17)
ax = axs[1]; e = np.array(gv["edges"]); xc = (e[:-1]+e[1:])/2
lo, hi = gv["series"][0], gv["series"][1]
ax.bar(xc-0.022, lo["counts"], width=0.044, color=C["probe"], label="probe: answerable (113)")
ax.bar(xc+0.022, hi["counts"], width=0.044, color=C["red"], label="probe: unanswerable (127)")
ax.axvline(0.8, color="k", ls=":", lw=0.9); ax.text(0.45, 80, "confidence\ncut 0.8", fontsize=7)
ax.set_xlabel("verbalized confidence"); ax.set_ylabel("test items"); ax.set_title("(b) stated confidence by probe verdict")
ax.legend(frameon=False, loc="upper center", fontsize=6.8); ax.set_ylim(0, 140)
fig.tight_layout(); fig.savefig(OUT/"fig_gap.pdf"); plt.close(fig)

# Fig 4: precision-recall operating points
ao = ld("abstain-option-squad_v2")["panels"][0]["groups"][0]
fig, ax = plt.subplots(figsize=(3.5, 3.3))
cols = [C["opt"], C["probe"], C["verbal"]]
for (rec, prec), lab, col in zip(ao["points"], ao["labels"], cols):
    ax.plot([rec], [prec], "o", ms=7, color=col, label=f"{lab} (R={rec:.2f}, P={prec:.2f})")
ax.axhline(0.5, color="k", lw=0.7, ls="--")
ax.set_xlim(0, 1.0); ax.set_ylim(0.45, 1.05); ax.set_xlabel("abstention recall"); ax.set_ylabel("abstention precision")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT/"fig_operating_points.pdf"); plt.close(fig)

# Fig A1: transfer by layer
tr_sa_to_sq = M["datasets"]["squad_v2"]["transfer_in"]["selfaware"]["test_auroc_by_layer"]
tr_sq_to_sa = M["datasets"]["selfaware"]["transfer_in"]["squad_v2"]["test_auroc_by_layer"]
fig, ax = plt.subplots(figsize=(6.2, 2.7))
ax.plot(layers, tr_sa_to_sq, color=C["red"], marker="o", ms=2.5, lw=1.3, label="SelfAware probe on SQuAD test (n=240)")
ax.plot(layers, tr_sq_to_sa, color=C["probe"], marker="s", ms=2.5, lw=1.3, label="SQuAD probe on SelfAware test (n=400)")
for arr, L, col in ((tr_sa_to_sq, 15, C["red"]), (tr_sq_to_sa, 18, C["probe"])):
    ax.plot([L], [arr[L]], "o", ms=9, mfc="none", mec=col, mew=1.5)
ax.plot([19], [tr_sa_to_sq[19]], "*", ms=10, color=C["red"]); ax.plot([10], [tr_sq_to_sa[10]], "*", ms=10, color=C["probe"])
ax.plot([], [], "o", mfc="none", mec="k", ms=8, label="source's validation-selected layer (reported)")
ax.plot([], [], "*", color="k", ms=9, label="maximum over layers on the target test set (optimistic)")
ax.axhline(0.5, color="k", ls="--", lw=0.7); ax.set_ylim(0.3, 0.85); ax.set_xlim(0, 32)
ax.set_xlabel("layer"); ax.set_ylabel("target test AUROC"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.8, ncol=1); fig.set_size_inches(6.2, 3.5)
fig.tight_layout(); fig.savefig(OUT/"fig_transfer.pdf"); plt.close(fig)

# Fig A2: all-dataset layer curves (confounded datasets dashed)
fig, ax = plt.subplots(figsize=(6.2, 2.7))
for nm, col, ls in (("squad_v2", C["probe"], "-"), ("selfaware", C["red"], "--"), ("pooled", C["lp"], "--")):
    o = M["datasets"][nm]["probe_logistic"]
    ax.plot(layers, o["test_auroc"], color=col, ls=ls, lw=1.4, label=f"{nm} (test n={M['datasets'][nm]['n_by_split']['test']}, L{o['best_layer']} chosen)" + ("" if nm == "squad_v2" else ", source-confounded"))
    ax.plot([o["best_layer"]], [o["test_auroc"][o["best_layer"]]], "o", color=col, ms=5)
ax.axhline(0.5, color="k", lw=0.7, ls="--"); ax.set_xlabel("layer"); ax.set_ylabel("test AUROC"); ax.set_ylim(0.4, 1.02); ax.set_xlim(0, 32)
ax.legend(frameon=False, loc="lower right", fontsize=7); fig.tight_layout(); fig.savefig(OUT/"fig_layers_all.pdf"); plt.close(fig)

# Fig A3: reliability + coverage-accuracy
rel = ld("reliability-squad_v2")["panels"][0]["groups"]; cov = ld("coverage-accuracy-squad_v2")["panels"][0]["groups"]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
ax = axs[0]
for g, col, mk in zip(rel[:2], (C["verbal"], C["lp"]), ("o", "s")):
    p = np.array(g["points"]); ax.plot(p[:, 0], p[:, 1], marker=mk, ms=4.5, color=col, lw=1, label=g["name"])
    for (x_, y_), lab in zip(p, g["labels"]): ax.annotate(lab, (x_, y_), fontsize=6, xytext=(3, -8), textcoords="offset points")
ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="perfect calibration"); ax.set_xlabel("mean confidence in bin"); ax.set_ylabel("fraction appropriate")
ax.set_title("(a) reliability (test, n=240)"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax = axs[1]
cmap = {"probe": C["probe"], "verbal_confidence": C["verbal"], "mean_logprob": C["lp"], "neg_entropy": "#555555", "consistency": C["opt"]}
for g in cov:
    p = np.array(g["points"]); ax.plot(p[:, 0], p[:, 1], lw=1.2, color=cmap.get(g["name"], "k"), label=g["name"].replace("_", " "))
ax.set_xlabel("coverage"); ax.set_ylabel("accuracy among answered"); ax.set_title("(b) coverage vs accuracy (threshold swept)")
ax.legend(frameon=False, fontsize=6.5, loc="lower left"); ax.set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig(OUT/"fig_reliability_coverage.pdf"); plt.close(fig)

# Fig A4: shuffled-label histogram
sd = ld("shuffled-label-squad_v2"); e = np.array(sd["edges"]); fig, ax = plt.subplots(figsize=(3.4, 2.4))
ax.bar(e[:-1], sd["series"][0]["counts"], width=np.diff(e), align="edge", color=C["null"])
ax.axvline(S["shuffled_label"]["squad_v2"]["real_test_auroc_at_val_best"], color=C["probe"], lw=1.6); ax.text(0.62, 8, "real probe\n0.906", fontsize=7, color=C["probe"])
ax.set_xlim(0.3, 1.0); ax.set_xlabel("test AUROC with shuffled training labels"); ax.set_ylabel("shuffles (of 20)")
fig.tight_layout(); fig.savefig(OUT/"fig_shuffle.pdf"); plt.close(fig)
print("ok", sorted(p.name for p in OUT.iterdir()))

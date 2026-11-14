"""Supplementary lexical baselines (added after the run; not part of the run's analysis/metrics.json).
Train on the stored train split, report val and test AUROC on the stored splits, label = answerable.
Features: (a) TF-IDF of the question text; (b) question-passage lexical overlap (SQuAD only); (c) both."""
import json, re, sys, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack, csr_matrix
R = Path(sys.argv[1])
rows = [json.loads(l) for l in open(R/"items.jsonl")]
STOP = set("the a an of in on at to for and or is are was were be been by with as from that this which who whom what when where why how did does do has have had it its their his her they he she not no".split())
def toks(s): return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]
def overlap_feats(r):
    q = set(toks(r["question"])); c = toks(r["context"] or ""); cs = set(c)
    frac = len(q & cs)/max(1, len(q))
    sents = [set(toks(s)) for s in re.split(r"(?<=[.!?])\s+", r["context"] or "")]
    best = max((len(q & s)/max(1, len(q)) for s in sents), default=0.0)
    return [frac, best, len(q)]
for ds in ("squad_v2", "selfaware"):
    sub = [r for r in rows if r["dataset"] == ds]
    sp = np.array([r["split"] for r in sub]); y = np.array([int(r["answerable"]) for r in sub])
    tr, va, te = sp == "train", sp == "val", sp == "test"
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xq = vec.fit_transform([r["question"] for r in sub][i] for i in range(len(sub))) if False else None
    texts = [r["question"] for r in sub]
    vec.fit([t for t, m in zip(texts, tr) if m]); Xq = vec.transform(texts)
    out = {}
    def fit(X, name):
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(X[tr], y[tr])
        s = clf.decision_function(X)
        out[name] = {"val_auroc": round(roc_auc_score(y[va], s[va]), 4), "test_auroc": round(roc_auc_score(y[te], s[te]), 4)}
    fit(Xq, "tfidf_question")
    if ds == "squad_v2":
        O = np.array([overlap_feats(r) for r in sub], float)
        O = (O-O[tr].mean(0))/O[tr].std(0)
        fit(csr_matrix(O), "overlap_features")
        fit(hstack([Xq, csr_matrix(O)]).tocsr(), "tfidf_plus_overlap")
    print(ds, "n_train/val/test", int(tr.sum()), int(va.sum()), int(te.sum()), json.dumps(out))

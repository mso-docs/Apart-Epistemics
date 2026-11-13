"""Layer-wise linear probes (logistic regression; optional mass-mean) on last-prompt-token hidden states.

H has shape (n_items, n_layers, d). y = 1 for answerable. The best layer is chosen on VAL AUROC only; test is untouched
until reporting (the plan: pick the best layer before looking at gating results on test)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..metrics import auroc


@dataclass
class LayerProbeResult:
    kind: str
    val_auroc: list[float]
    test_auroc: list[float]
    best_layer: int
    test_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    models: dict = field(default_factory=dict, repr=False)   # layer -> (mean, std, w, b)

    def summary(self) -> dict:
        return {"kind": self.kind, "best_layer": self.best_layer, "val_auroc": self.val_auroc,
                "test_auroc": self.test_auroc,
                "test_accuracy": self.test_acc, "val_accuracy": self.val_acc}


def _standardize(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def _fit_logistic(X, y, C, max_iter, seed):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(X, y)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def _fit_massmean(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    mid = 0.5 * (X[y == 1].mean(0) + X[y == 0].mean(0))
    return w, float(-(w @ mid))


def project_scores(res: LayerProbeResult, H, layer: int | None = None):
    layer = res.best_layer if layer is None else layer
    mu, sd, w, b = res.models[layer]
    return ((H[:, layer, :].astype(np.float64) - mu) / sd) @ w + b


def fit_layerwise(H, y, split, kind: str = "logistic", C: float = 1.0, max_iter: int = 1000, seed: int = 0,
                  layers: list[int] | None = None) -> LayerProbeResult:
    H = np.asarray(H)
    y = np.asarray(y).astype(int)
    split = np.asarray(split)
    tr, va, te = split == "train", split == "val", split == "test"
    if tr.sum() < 4 or len(np.unique(y[tr])) < 2:
        raise ValueError("need >= 4 train items with both classes to fit probes")
    n_layers = H.shape[1]
    layers = list(range(n_layers)) if layers is None else layers
    val_auc, test_auc, models = [float("nan")] * n_layers, [float("nan")] * n_layers, {}
    val_acc, test_acc = [float("nan")] * n_layers, [float("nan")] * n_layers
    for L in layers:
        X = H[:, L, :].astype(np.float64)
        mu, sd = _standardize(X[tr])
        Z = (X - mu) / sd
        if kind == "massmean":
            w, b = _fit_massmean(Z[tr], y[tr])
        else:
            w, b = _fit_logistic(Z[tr], y[tr], C, max_iter, seed)
        models[L] = (mu, sd, w, b)
        s = Z @ w + b
        if va.any():
            val_auc[L] = auroc(y[va], s[va])
            val_acc[L] = float(((s[va] > 0) == (y[va] == 1)).mean())
        if te.any():
            test_auc[L] = auroc(y[te], s[te])
            test_acc[L] = float(((s[te] > 0) == (y[te] == 1)).mean())
    pick = np.array(val_auc, float)
    pick[~np.isfinite(pick)] = -1
    best = int(np.argmax(pick))
    return LayerProbeResult(kind, val_auc, test_auc, best, test_acc, val_acc, models)

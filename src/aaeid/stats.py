"""Metrics with uncertainty. Small test sets need intervals, not point estimates."""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def wilson(k: int, n: int, alpha: float = 0.05):
    """Proportion with a Wilson score interval. Works for small n and for 0% / 100%."""
    if n == 0:
        return {"p": None, "lo": None, "hi": None, "k": 0, "n": 0}
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"p": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "k": int(k), "n": int(n)}


def auc_ci(y, s, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """AUC with a percentile bootstrap interval (resampling items)."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return {"auc": None, "lo": None, "hi": None, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": float(roc_auc_score(y, s)), "lo": float(lo), "hi": float(hi),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}


def gwet_ac1(a, b):
    """Gwet's AC1 for two raters, any number of categories. Robust to skewed labels."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.unique(np.concatenate([a, b]))
    po = np.mean(a == b)
    pi = np.array([(np.mean(a == c) + np.mean(b == c)) / 2 for c in cats])
    pe = np.sum(pi * (1 - pi)) / (len(cats) - 1)
    return float((po - pe) / (1 - pe))


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0, "n": len(xs)}

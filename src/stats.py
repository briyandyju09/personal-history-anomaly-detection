"""
Statistical analysis: uncertainty, paired comparisons, curve fitting and the
operational definitions of a personalisation threshold.

All resampling is over accounts, which are the independent units of the design.
Resampling transactions would treat rows from the same account as independent
and understate uncertainty.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple:
    """Percentile bootstrap CI for the mean, resampling accounts."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(v.mean()), float(lo), float(hi), len(v))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict:
    """
    Paired bootstrap of mean(a) - mean(b) over accounts. Accounts with a
    missing value in either arm are dropped pairwise, which keeps the pairing
    intact.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {
        "diff": float((a - b).mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n": int(len(a)),
    }


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Wilcoxon signed-rank test with a matched-pairs rank-biserial effect size.

    The p value answers whether a difference is detectable, not whether it
    matters. The effect size and the paired mean difference carry the practical
    question and should be read first.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    nz = d[d != 0]
    if len(nz) < 5:
        return {"stat": np.nan, "p": np.nan, "effect": np.nan, "n": int(len(a))}
    res = stats.wilcoxon(a, b, zero_method="wilcox")
    r = stats.rankdata(np.abs(nz))
    rb = (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()
    return {
        "stat": float(res.statistic),
        "p": float(res.pvalue),
        "effect": float(rb),
        "n": int(len(a)),
    }


def holm_correct(pvalues: dict) -> dict:
    """Holm step-down adjustment over a family of tests."""
    items = [(k, v) for k, v in pvalues.items() if not np.isnan(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = adj
    for k, v in pvalues.items():
        out.setdefault(k, np.nan)
    return out


# ---------------------------------------------------------------------------
# Saturation curve
# ---------------------------------------------------------------------------


def saturating_exponential(n, y0, ymax, tau):
    """y(n) = ymax - (ymax - y0) * exp(-n / tau)"""
    return ymax - (ymax - y0) * np.exp(-n / tau)


def fit_saturation(
    n: np.ndarray, y: np.ndarray
) -> dict:
    """
    Fit a saturating exponential to a performance curve.

    We report the fit only with its R^2 so a poor fit is visible. A curve is
    not forced onto data that does not support one: when R^2 is low the fitted
    asymptote and time constant should not be interpreted.
    """
    n = np.asarray(n, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(y)
    n, y = n[ok], y[ok]
    if len(n) < 4:
        return {"ok": False, "reason": "fewer than 4 usable points"}
    try:
        p0 = [float(y[0]), float(np.max(y)), float(np.median(n))]
        popt, _ = optimize.curve_fit(
            saturating_exponential, n, y, p0=p0, maxfev=20000
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"fit failed: {exc}"}
    resid = y - saturating_exponential(n, *popt)
    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {
        "ok": True,
        "y0": float(popt[0]),
        "ymax": float(popt[1]),
        "tau": float(popt[2]),
        "r2": float(r2),
    }


# ---------------------------------------------------------------------------
# Personalisation threshold definitions
# ---------------------------------------------------------------------------


def relative_threshold(
    sizes: np.ndarray, values: np.ndarray, fraction: float = 0.90
) -> dict:
    """
    Relative definition. The smallest evaluated history size whose performance
    reaches ``fraction`` of the best performance observed on the curve, and
    which is not later undercut at any larger size.

    Limitation: this is defined against the best OBSERVED value, not a true
    asymptote. If the curve is still rising at the largest evaluated size, the
    reference point is a lower bound and the threshold is optimistic.
    """
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    sizes, values = sizes[ok], values[ok]
    if len(sizes) == 0:
        return {"threshold": np.nan, "reference": np.nan}
    ref = float(np.max(values))
    target = fraction * ref
    for i in range(len(sizes)):
        if np.all(values[i:] >= target):
            return {
                "threshold": float(sizes[i]),
                "reference": ref,
                "target": float(target),
                "still_rising": bool(values[-1] >= ref - 1e-12),
            }
    return {"threshold": np.nan, "reference": ref, "target": float(target)}


def absolute_threshold(
    sizes: np.ndarray, values: np.ndarray, level: float
) -> dict:
    """
    Absolute definition. The smallest history size reaching a fixed externally
    chosen performance level and staying at or above it thereafter.

    Limitation: the level is a policy choice, not an empirical quantity. It
    should be justified by the cost of a false positive in the deployment
    context, which this study does not attempt to estimate.
    """
    sizes = np.asarray(sizes, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = ~np.isnan(values)
    sizes, values = sizes[ok], values[ok]
    for i in range(len(sizes)):
        if np.all(values[i:] >= level):
            return {"threshold": float(sizes[i]), "level": level}
    return {"threshold": np.nan, "level": level}


def stability_threshold(
    sizes: np.ndarray, spreads: np.ndarray, max_spread: float
) -> dict:
    """
    Stability definition. The smallest history size at which the across-account
    spread of performance falls below ``max_spread`` and remains below it.

    Limitation: a stable curve can be stably poor. This definition must be read
    together with a level-based definition, never on its own.
    """
    sizes = np.asarray(sizes, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    ok = ~np.isnan(spreads)
    sizes, spreads = sizes[ok], spreads[ok]
    for i in range(len(sizes)):
        if np.all(spreads[i:] <= max_spread):
            return {"threshold": float(sizes[i]), "max_spread": max_spread}
    return {"threshold": np.nan, "max_spread": max_spread}


def marginal_gains(sizes: np.ndarray, values: np.ndarray) -> list:
    """Absolute and per-doubling gain between consecutive evaluated sizes."""
    out = []
    for i in range(1, len(sizes)):
        n0, n1 = float(sizes[i - 1]), float(sizes[i])
        v0, v1 = float(values[i - 1]), float(values[i])
        doublings = np.log2(n1 / n0) if n0 > 0 else np.nan
        out.append(
            {
                "from": n0,
                "to": n1,
                "gain": v1 - v0,
                "gain_per_doubling": (v1 - v0) / doublings
                if doublings and not np.isnan(doublings)
                else np.nan,
            }
        )
    return out

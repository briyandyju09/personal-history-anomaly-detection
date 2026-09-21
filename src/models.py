"""
Anomaly detectors evaluated in this study.

Common interface
----------------
Every detector exposes ``fit(X)`` and ``score(X)`` where a higher score means
more anomalous. Detectors are fitted on an account's training history (personal),
on a pooled population sample (global), or on a combination (hybrid).

Degenerate regimes are handled explicitly rather than by silently failing. With
n = 1 a dispersion estimate does not exist; the affected detectors fall back to
a documented behaviour and the experiment records that the regime is degenerate.
That fallback is itself part of the finding, not a workaround hidden from view.

The hybrid mechanism
--------------------
``ShrinkageMahalanobis`` is the model that makes RQ8 precise. Personal location
and scatter estimates are shrunk toward population estimates with a weight that
depends only on the amount of personal history:

    mu_hybrid    = (n * mu_personal + kappa * mu_global) / (n + kappa)
    Sigma_hybrid = (n * S_personal + kappa * S_global) / (n + kappa) + eps * I

kappa is a pseudo-count expressing how many personal observations the
population prior is worth. At n = 0 the model is exactly the global model; as
n grows it converges to the purely personal model; kappa sets the crossover.
This is a standard conjugate-style shrinkage, not a tuned blend, and kappa is
fixed in advance rather than selected on test data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import IsolationForest


class Detector:
    """Base interface. Higher score means more anomalous."""

    name = "base"
    degenerate = False

    def fit(self, X: np.ndarray) -> "Detector":
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Univariate statistical baselines on log-amount (feature column 0)
# ---------------------------------------------------------------------------


class GaussianZ(Detector):
    """|z| against the mean and standard deviation of a single feature."""

    name = "gaussian_z"

    def __init__(self, column: int = 0):
        self.column = column
        self.mu = 0.0
        self.sigma = 1.0

    def fit(self, X: np.ndarray) -> "GaussianZ":
        v = X[:, self.column]
        self.mu = float(np.mean(v))
        if len(v) < 2:
            self.sigma = 1.0
            self.degenerate = True
        else:
            s = float(np.std(v, ddof=1))
            self.sigma = s if s > 1e-9 else 1.0
            self.degenerate = s <= 1e-9
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.abs(X[:, self.column] - self.mu) / self.sigma


class RobustZ(Detector):
    """|z| against the median and a MAD-based scale estimate."""

    name = "robust_z"

    def __init__(self, column: int = 0):
        self.column = column
        self.med = 0.0
        self.scale = 1.0

    def fit(self, X: np.ndarray) -> "RobustZ":
        v = X[:, self.column]
        self.med = float(np.median(v))
        mad = float(np.median(np.abs(v - self.med)))
        s = 1.4826 * mad
        if len(v) < 2 or s <= 1e-9:
            self.scale = 1.0
            self.degenerate = True
        else:
            self.scale = s
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.abs(X[:, self.column] - self.med) / self.scale


# ---------------------------------------------------------------------------
# Multivariate
# ---------------------------------------------------------------------------


@dataclass
class ShrinkageMahalanobis(Detector):
    """
    Mahalanobis distance with shrinkage toward a population prior.

    kappa = 0     purely personal (classic Mahalanobis with ridge)
    kappa = inf   purely global
    0 < kappa     hybrid; the personal estimate gains weight as n grows
    """

    kappa: float = 0.0
    ridge: float = 1e-3
    prior_mu: np.ndarray | None = None
    prior_cov: np.ndarray | None = None
    name: str = field(default="mahalanobis", init=False)

    def __post_init__(self):
        self.mu = None
        self.prec = None
        self.n = 0

    def set_prior(self, mu: np.ndarray, cov: np.ndarray) -> "ShrinkageMahalanobis":
        self.prior_mu = mu
        self.prior_cov = cov
        return self

    def fit(self, X: np.ndarray) -> "ShrinkageMahalanobis":
        n, d = X.shape
        self.n = n
        if n >= 1:
            mu_p = X.mean(axis=0)
            S_p = np.cov(X, rowvar=False, ddof=1) if n > 1 else np.zeros((d, d))
        else:
            mu_p = np.zeros(d)
            S_p = np.zeros((d, d))

        if self.kappa > 0 and self.prior_mu is not None:
            w = n + self.kappa
            mu = (n * mu_p + self.kappa * self.prior_mu) / w
            S = (n * S_p + self.kappa * self.prior_cov) / w
        else:
            mu, S = mu_p, S_p

        # A personal covariance from n observations in d dimensions is singular
        # whenever n <= d. The ridge keeps the distance defined; we record the
        # regime as degenerate so the results can be read with that in mind.
        self.degenerate = (self.kappa == 0) and (n <= d)
        S = S + self.ridge * np.eye(d) * max(1.0, float(np.trace(S)) / max(d, 1))
        self.mu = mu
        self.prec = np.linalg.pinv(S)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        delta = X - self.mu
        return np.einsum("ij,jk,ik->i", delta, self.prec, delta)


class KNNDistance(Detector):
    """Distance to the k-th nearest training point. Robust at very small n."""

    name = "knn"

    def __init__(self, k: int = 5):
        self.k = k
        self.X = None

    def fit(self, X: np.ndarray) -> "KNNDistance":
        self.X = X
        self.k_eff = max(1, min(self.k, len(X)))
        self.degenerate = len(X) < self.k
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        d = np.sqrt(
            np.maximum(
                ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2), 0.0
            )
        )
        d.sort(axis=1)
        return d[:, self.k_eff - 1]


class IForest(Detector):
    """Isolation Forest with a seeded, fixed configuration."""

    name = "iforest"

    def __init__(self, n_estimators: int = 200, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed
        self.model = None

    def fit(self, X: np.ndarray) -> "IForest":
        # max_samples cannot exceed the training size; with very small n the
        # ensemble degenerates toward memorising the few points it has.
        self.degenerate = len(X) < 8
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(256, max(1, len(X))),
            random_state=self.seed,
            n_jobs=1,
        ).fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        # score_samples is higher for normal points; negate for our convention.
        return -self.model.score_samples(X)


class TinyAutoencoder(Detector):
    """
    Small undercomplete autoencoder, d -> h -> d, tanh hidden layer, trained
    with full-batch Adam on mean squared reconstruction error. Implemented in
    NumPy to keep the dependency set small and the run fully deterministic.

    Score = reconstruction error.
    """

    name = "autoencoder"

    def __init__(
        self,
        hidden: int = 4,
        epochs: int = 300,
        lr: float = 0.01,
        seed: int = 0,
        weight_decay: float = 1e-4,
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.weight_decay = weight_decay

    def fit(self, X: np.ndarray) -> "TinyAutoencoder":
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        h = min(self.hidden, max(1, d - 1))
        self.degenerate = n < 4 * h

        W1 = rng.normal(0, 1.0 / np.sqrt(d), size=(d, h))
        b1 = np.zeros(h)
        W2 = rng.normal(0, 1.0 / np.sqrt(h), size=(h, d))
        b2 = np.zeros(d)
        params = [W1, b1, W2, b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1c, b2c, eps = 0.9, 0.999, 1e-8

        for t in range(1, self.epochs + 1):
            Z = np.tanh(X @ params[0] + params[1])
            R = Z @ params[2] + params[3]
            E = R - X
            gR = 2.0 * E / max(n, 1)
            gW2 = Z.T @ gR + self.weight_decay * params[2]
            gb2 = gR.sum(axis=0)
            gZ = gR @ params[2].T * (1.0 - Z**2)
            gW1 = X.T @ gZ + self.weight_decay * params[0]
            gb1 = gZ.sum(axis=0)
            grads = [gW1, gb1, gW2, gb2]
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = b1c * m[i] + (1 - b1c) * g
                v[i] = b2c * v[i] + (1 - b2c) * g * g
                mh = m[i] / (1 - b1c**t)
                vh = v[i] / (1 - b2c**t)
                params[i] = p - self.lr * mh / (np.sqrt(vh) + eps)

        self.params = params
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        W1, b1, W2, b2 = self.params
        Z = np.tanh(X @ W1 + b1)
        R = Z @ W2 + b2
        return ((R - X) ** 2).mean(axis=1)


# ---------------------------------------------------------------------------
# Score-level hybrid
# ---------------------------------------------------------------------------


class ScoreFusionHybrid(Detector):
    """
    Combine a personal and a global detector at score level after converting
    each raw score to a within-model rank on the evaluation batch.

    The fusion weight depends only on the amount of personal history:

        w_personal = n / (n + kappa)

    so the combination reduces to the global detector when no personal history
    exists and to the personal detector in the large-n limit. kappa is fixed in
    advance. Rank conversion is applied to the evaluation batch, which makes
    the two score scales comparable without using labels.
    """

    name = "fusion"

    def __init__(self, personal: Detector, global_model: Detector, kappa: float = 25.0):
        self.personal = personal
        self.global_model = global_model
        self.kappa = kappa
        self.n = 0

    def fit(self, X: np.ndarray) -> "ScoreFusionHybrid":
        self.n = len(X)
        self.personal.fit(X)
        self.degenerate = bool(getattr(self.personal, "degenerate", False))
        return self

    @staticmethod
    def _to_rank(s: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(s))
        return order / max(len(s) - 1, 1)

    def score(self, X: np.ndarray) -> np.ndarray:
        w = self.n / (self.n + self.kappa)
        sp = self._to_rank(self.personal.score(X))
        sg = self._to_rank(self.global_model.score(X))
        return w * sp + (1.0 - w) * sg


def personal_detector_factory(seed: int = 0) -> dict:
    """The set of personal detectors evaluated at every history size."""
    return {
        "gaussian_z": lambda: GaussianZ(column=0),
        "robust_z": lambda: RobustZ(column=0),
        "mahalanobis": lambda: ShrinkageMahalanobis(kappa=0.0),
        "knn": lambda: KNNDistance(k=5),
        "iforest": lambda: IForest(seed=seed),
        "autoencoder": lambda: TinyAutoencoder(seed=seed),
    }

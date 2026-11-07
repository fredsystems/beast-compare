"""Confidence intervals that respect the dependence between messages.

Messages from one aircraft visit are strongly correlated (same distance,
same bearing, same transponder), so resampling individual transmissions
would give intervals far too narrow. Visits are resampled instead: a rate
is ``sum(k) / sum(n)`` over the resampled visits, where ``n`` is the
visit's transmissions in the band and ``k`` those a receiver decoded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Floats = npt.NDArray[np.float64]

DEFAULT_RESAMPLES = 200
DEFAULT_SEED = 20261107
CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def visit_weights(
    visits: int, resamples: int, rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    """How often each visit is drawn in each resample, (resamples, visits)."""
    weights: npt.NDArray[np.int64] = rng.multinomial(
        visits, np.full(visits, 1 / visits), size=resamples
    )
    return weights


def _interval(estimate: float, samples: Floats) -> Interval:
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return Interval(estimate, np.nan, np.nan)
    tail = (1 - CONFIDENCE) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return Interval(estimate, float(low), float(high))


def ratio(n: Floats, k: Floats, weights: npt.NDArray[np.int64]) -> Interval:
    """``sum(k) / sum(n)`` with a visit-resampled interval."""
    total = n.sum()
    estimate = float(k.sum() / total) if total else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = (weights @ k) / (weights @ n)
    return _interval(estimate, samples)


def ratio_difference(
    n_a: Floats,
    k_a: Floats,
    n_b: Floats,
    k_b: Floats,
    weights: npt.NDArray[np.int64],
) -> Interval:
    """Rate of a minus rate of b, both resampled with the same visits."""
    estimate = float(k_a.sum() / n_a.sum() - k_b.sum() / n_b.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = (weights @ k_a) / (weights @ n_a) - (weights @ k_b) / (weights @ n_b)
    return _interval(estimate, samples)

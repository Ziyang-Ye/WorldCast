"""Bootstrap intervals, the paper's 95% intervals of 10,000 resamples: over clusters of views (the
ten-second windows, the two views of a window together; :func:`cluster_bootstrap`), and over the
units of a table cell (rounds or ten-second windows; :func:`unit_interval`).

Every call draws from a generator seeded anew, so all columns and rows of a table resample the same
clusters whenever they hold the same clusters in the same order. A resample draws as many clusters
as there are, with replacement, and takes the statistic over the values of the drawn clusters; the
interval is the 2.5th and 97.5th percentile (linear interpolation) of the resampled statistics.
"""

import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "INTERVAL_PERCENTILES",
    "PairedDifference",
    "cluster_bootstrap",
    "mean_over_units",
    "paired_difference",
    "unit_interval",
]

#: Resamples of an interval.
BOOTSTRAP_DRAWS = 10_000
#: Seed of the generator of every interval.
BOOTSTRAP_SEED = 20260922
#: Percentiles of the resampled statistics that bound the interval.
INTERVAL_PERCENTILES = (2.5, 97.5)


@dataclass(frozen=True)
class PairedDifference:
    """A paired difference of two arms over the views both score, with its interval.

    Attributes:
        point (float): the median or mean of the per-view differences.
        low (float): the interval's lower bound.
        high (float): the interval's upper bound.
        views (int): the views both arms score.
        clusters (int): the clusters of those views.
    """

    point: float
    low: float
    high: float
    views: int
    clusters: int


def cluster_bootstrap(
    values: Sequence[float], clusters: Sequence[Hashable], statistic: str = "median"
) -> tuple[float, float]:
    """The bootstrap interval of the median or mean of ``values``, resampling whole clusters.

    Clusters are numbered in the order of their first value, which fixes which cluster each draw
    picks: pass the values in a fixed order (the paper's: sorted by view).

    Args:
        values (Sequence[float]): one value per view (e.g. its Rendered, or a paired difference).
        clusters (Sequence[Hashable]): the cluster of each value (e.g. its window).
        statistic (str): ``median`` or ``mean``.

    Returns:
        tuple[float, float]: the interval's lower and upper bound.
    """
    if statistic not in ("median", "mean"):
        raise ValueError(f"statistic {statistic!r}: expected 'median' or 'mean'")
    if len(values) != len(clusters) or not len(values):
        raise ValueError(f"{len(values)} values and {len(clusters)} clusters")
    groups: dict[Hashable, list[float]] = {}
    for value, cluster in zip(values, clusters):
        groups.setdefault(cluster, []).append(float(value))
    arrays = [np.asarray(group, dtype=np.float64) for group in groups.values()]
    count = len(arrays)
    sizes = np.array([len(a) for a in arrays])
    sums = np.array([a.sum() for a in arrays])
    flat = np.concatenate(arrays)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = np.empty(BOOTSTRAP_DRAWS)
    for b in range(BOOTSTRAP_DRAWS):
        drawn = np.bincount(rng.integers(0, count, count), minlength=count)
        if statistic == "mean":
            resampled[b] = (drawn * sums).sum() / (drawn * sizes).sum()
        else:
            resampled[b] = np.median(np.repeat(flat, np.repeat(drawn, sizes)))
    low, high = np.percentile(resampled, INTERVAL_PERCENTILES)
    return float(low), float(high)


def paired_difference(
    first: Mapping[str, float],
    second: Mapping[str, float],
    cluster: Callable[[str], Hashable],
    statistic: str = "median",
) -> PairedDifference:
    """``first - second`` per view over the views both arms score, and its interval
    (:func:`cluster_bootstrap` over the views in sorted order).

    Args:
        first (Mapping[str, float]): view -> value of one arm (e.g. its Rendered).
        second (Mapping[str, float]): view -> value of the other arm.
        cluster (Callable[[str], Hashable]): the cluster of a view (e.g.
            :func:`~worldcast.engine.evaluation.entity.window_of`).
        statistic (str): ``median`` or ``mean``.

    Returns:
        PairedDifference: the statistic of the differences, its interval and the counts.
    """
    views = sorted(set(first) & set(second))
    values = [first[view] - second[view] for view in views]
    clusters = [cluster(view) for view in views]
    low, high = cluster_bootstrap(values, clusters, statistic)
    point = np.median(values) if statistic == "median" else np.mean(values)
    return PairedDifference(float(point), low, high, len(values), len(set(clusters)))


def mean_over_units(table: np.ndarray) -> np.ndarray:
    """A table cell: ``[U, S]`` per-unit values -> ``[S]`` their mean, each unit weighted equally
    and units without a value skipped."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(table, dtype=np.float64), 0)


def unit_interval(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 95% unit-cluster bootstrap interval of :func:`mean_over_units`.

    A unit is a round or a ten-second window. A resample draws as many units as there are, with
    replacement, whole rows at once, from a generator seeded anew on every call
    (:data:`BOOTSTRAP_SEED`), so every column and every table of the same units shares the draws;
    its mean skips the units without a value. The bounds are the 2.5th and 97.5th percentiles
    (linear) of the resampled means.

    Args:
        table (np.ndarray): ``[U, S]`` per-unit values, e.g. an arm difference per segment.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``[S]`` lower and upper bounds.
    """
    table = np.asarray(table, dtype=np.float64)
    count = len(table)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.array(
            [np.nanmean(table[rng.integers(0, count, count)], 0) for _ in range(BOOTSTRAP_DRAWS)]
        )
        low, high = np.nanpercentile(means, INTERVAL_PERCENTILES, axis=0)
    return low, high

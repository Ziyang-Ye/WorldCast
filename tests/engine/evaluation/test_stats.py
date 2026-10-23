"""The cluster bootstrap against its plain form, paired differences, and the mean over units with
its interval."""

import math

import numpy as np
import pytest

from worldcast.engine.evaluation.entity import window_of
from worldcast.engine.evaluation.stats import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    INTERVAL_PERCENTILES,
    cluster_bootstrap,
    mean_over_units,
    paired_difference,
    unit_interval,
)


def plain_bootstrap(values, clusters, statistic):
    """Each resample concatenates the values of the drawn clusters."""
    groups: dict = {}
    for value, cluster in zip(values, clusters):
        groups.setdefault(cluster, []).append(value)
    arrays = [np.asarray(group) for group in groups.values()]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = np.concatenate([arrays[k] for k in rng.integers(0, len(arrays), len(arrays))])
        resampled.append(np.median(sample) if statistic == "median" else sample.mean())
    return tuple(np.percentile(resampled, INTERVAL_PERCENTILES))


@pytest.fixture(scope="module")
def clustered() -> tuple[list[float], list[int]]:
    """60 values in 25 clusters of one to four values each."""
    rng = np.random.default_rng(0)
    clusters = sorted(int(c) for c in rng.integers(0, 25, 60))
    return [float(v) for v in rng.normal(size=60)], clusters


def test_the_median_interval_equals_the_plain_form(clustered):
    values, clusters = clustered
    assert cluster_bootstrap(values, clusters) == plain_bootstrap(values, clusters, "median")


def test_the_mean_interval_equals_the_plain_form(clustered):
    values, clusters = clustered
    low, high = cluster_bootstrap(values, clusters, "mean")
    assert (low, high) == pytest.approx(plain_bootstrap(values, clusters, "mean"), abs=1e-12)


def test_every_call_draws_the_same_clusters(clustered):
    values, clusters = clustered
    assert cluster_bootstrap(values, clusters) == cluster_bootstrap(values, clusters)
    shifted = [v + 1.0 for v in values]
    low, high = cluster_bootstrap(values, clusters)
    assert cluster_bootstrap(shifted, clusters) == pytest.approx((low + 1.0, high + 1.0))
    assert cluster_bootstrap([0.3, 0.3, 0.3], ["w", "w", "w"]) == (0.3, 0.3)


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="statistic"):
        cluster_bootstrap([1.0], ["w"], "mode")
    with pytest.raises(ValueError, match="clusters"):
        cluster_bootstrap([1.0, 2.0], ["w"])
    with pytest.raises(ValueError):
        cluster_bootstrap([], [])


def test_a_paired_difference_takes_the_views_both_arms_score():
    first = {"m-r01-p00@0": 0.5, "m-r01-p01@0": 0.7, "m-r02-p00@0": 0.2, "m-r03-p00@0": 0.9}
    second = {"m-r01-p00@0": 0.4, "m-r01-p01@0": 0.4, "m-r02-p00@0": 0.3, "m-r04-p00@0": 0.0}
    difference = paired_difference(first, second, window_of)
    assert (difference.views, difference.clusters) == (3, 2)
    assert difference.point == pytest.approx(0.1)
    values = [0.5 - 0.4, 0.7 - 0.4, 0.2 - 0.3]
    clusters = ["m-r01@0", "m-r01@0", "m-r02@0"]
    assert (difference.low, difference.high) == cluster_bootstrap(values, clusters)
    mean = paired_difference(first, second, window_of, "mean")
    assert mean.point == pytest.approx(0.1)
    assert (mean.low, mean.high) == cluster_bootstrap(values, clusters, "mean")


def test_the_mean_over_units_weights_units_equally_and_skips_missing():
    table = np.array([[1.0, math.nan], [3.0, 4.0], [5.0, 8.0]])
    assert mean_over_units(table).tolist() == [3.0, 6.0]


def test_the_unit_interval_resamples_whole_units_with_shared_draws():
    rng = np.random.default_rng(3)
    column = rng.normal(size=31)
    table = np.column_stack([column, 2 * column, column])
    table[4, 2] = math.nan
    low, high = unit_interval(table)
    draws = np.random.default_rng(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        means.append(np.nanmean(table[draws.integers(0, 31, 31)], 0))
    assert low.tolist() == np.percentile(means, 2.5, axis=0).tolist()
    assert high.tolist() == np.percentile(means, 97.5, axis=0).tolist()
    assert (low[1], high[1]) == pytest.approx((2 * low[0], 2 * high[0]))
    assert np.array_equal(np.array(unit_interval(table[:, :1])), np.array([low[:1], high[:1]]))

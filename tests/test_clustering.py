"""Tests for ress.core.clustering and ress.core.quality."""

import numpy as np

from ress.core.clustering import find_optimal_result, standardize_z4
from ress.core.quality import calculate_consensus, calculate_quality_index


def make_measurements(phi, dt, n, jitter_phi, jitter_dt, seed, t0=5.0):
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n):
        results.append(
            {
                "phi": phi + rng.normal(0, jitter_phi),
                "dt": dt + rng.normal(0, jitter_dt),
                "phi_err": abs(rng.normal(2.0, 0.5)),
                "dt_err": abs(rng.normal(0.005, 0.001)),
                "spol": 70.0,
                "CC_FS": 0.9,
                "CC_NE": 0.1,
                "L_R": 0.05,
                "window": (t0 - 0.3 - 0.001 * i, t0 + 0.5 + 0.001 * i),
            }
        )
    return results


def test_standardize_z4_constant_input():
    X = np.array([3.0, 3.0, 3.0])
    assert np.all(standardize_z4(X) == 1.0)


def test_find_optimal_result_recovers_stable_cluster():
    # a tight cluster at (40 deg, 0.12 s) plus scattered outliers
    stable = make_measurements(40.0, 0.12, 30, 1.0, 0.002, seed=1)
    outliers = make_measurements(-60.0, 0.30, 5, 20.0, 0.05, seed=2)
    cluster = find_optimal_result(stable + outliers, linkage="ward", k_max=10, n_min=5)

    assert cluster["success"]
    assert abs(cluster["phi"] - 40.0) < 5.0
    assert abs(cluster["dt"] - 0.12) < 0.01
    assert cluster["n_in_cluster"] >= 5


def test_find_optimal_result_too_few_measurements():
    results = make_measurements(40.0, 0.12, 2, 1.0, 0.002, seed=3)
    cluster = find_optimal_result(results)
    assert not cluster["success"]


def test_find_optimal_result_ignores_nan_errors():
    results = make_measurements(40.0, 0.12, 20, 1.0, 0.002, seed=4)
    for r in results[:18]:
        r["phi_err"] = np.nan
    cluster = find_optimal_result(results)
    assert not cluster["success"]


def test_quality_index_good_split():
    # RC and EV agree -> Delta ~ 1, Omega ~ 0 -> good split
    q = calculate_quality_index(phi_rc=40.0, dt_rc=0.12, phi_ev=41.0, dt_ev=0.12, gamma=0.75)
    assert q["Q"] >= 0.75
    assert q["Class"] == "Good Split"


def test_quality_index_good_null():
    # RC dt near zero and phi off by 45 -> null
    q = calculate_quality_index(phi_rc=85.0, dt_rc=0.001, phi_ev=40.0, dt_ev=0.12, gamma=0.75)
    assert q["Q"] <= -0.75
    assert q["Class"] == "Good Null"


def test_quality_index_poor():
    q = calculate_quality_index(phi_rc=60.0, dt_rc=0.06, phi_ev=40.0, dt_ev=0.12, gamma=0.75)
    assert -0.75 < q["Q"] < 0.75
    assert q["Class"] == "Poor/Noise"


def _filter_result(phi, dt, name):
    return {
        "phi": phi,
        "dt": dt,
        "phi_err": 3.0,
        "dt_err": 0.01,
        "spol": 70.0,
        "CC_FS": 0.9,
        "CC_NE": 0.1,
        "L_R": 0.05,
        "T_dominant": 0.25,
        "n_clusters": 2,
        "n_in_cluster": 15,
        "filter": name,
        "cluster_data": None,
        "window": (4.7, 5.5),
    }


def test_consensus_averages_within_tolerance():
    results = [
        _filter_result(40.0, 0.120, "1.0-8.0 Hz"),
        _filter_result(45.0, 0.125, "0.5-5.0 Hz"),  # within tolerance
        _filter_result(80.0, 0.300, "2.0-6.0 Hz"),  # excluded
    ]
    consensus = calculate_consensus(results, "EV", phi_tolerance=15.0, dt_tolerance=0.020)
    assert consensus["n_filters_used"] == 2
    assert abs(consensus["phi"] - 42.5) < 1e-9
    assert abs(consensus["dt"] - 0.1225) < 1e-9


def test_consensus_empty():
    assert calculate_consensus([], "EV") is None

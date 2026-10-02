"""Quality assessment and multi-filter consensus for splitting measurements.

Quality index after Wuestefeld et al. (2010): comparison of the
rotation-correlation and eigenvalue results classifies a measurement as
Good Split, Good Null or Poor/Noise.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_quality_index(phi_rc, dt_rc, phi_ev, dt_ev, gamma=0.75):
    """Quality index Q of a single measurement (Wuestefeld et al., 2010).

    :param phi_rc: fast polarization from the RC method (degrees)
    :param dt_rc: delay time from the RC method (seconds)
    :param phi_ev: fast polarization from the EV method (degrees)
    :param dt_ev: delay time from the EV method (seconds)
    :param gamma: |Q| threshold for the Good Split / Good Null classes
    :returns: dict with 'Q', 'Delta', 'Omega' and 'Class'
    """
    # Delta: ratio of delay times (1.0 for good splits, 0.0 for nulls)
    if dt_ev == 0:
        delta = 0
    else:
        delta = dt_rc / dt_ev

    # Omega: normalized polarization difference (0 for splits, 1 for nulls)
    diff_phi = abs(phi_rc - phi_ev)
    if diff_phi > 45:  # correct for the 90-degree periodicity
        diff_phi = 90 - diff_phi

    omega = abs(diff_phi) / 45.0

    # distances to the ideal Good (1, 0) and Null (0, 1) points
    dist_to_good = np.sqrt((delta - 1) ** 2 + omega**2)
    dist_to_null = np.sqrt(delta**2 + (omega - 1) ** 2)

    dist_to_good = min(dist_to_good, 1.0)
    dist_to_null = min(dist_to_null, 1.0)

    # negative if closer to NULL, positive if closer to GOOD
    if dist_to_null < dist_to_good:
        q_value = -(1.0 - dist_to_null)
    else:
        q_value = 1.0 - dist_to_good

    if q_value >= gamma:
        classification = "Good Split"
    elif q_value <= -gamma:
        classification = "Good Null"
    else:
        classification = "Poor/Noise"

    return {"Q": q_value, "Delta": delta, "Omega": omega, "Class": classification}


def calculate_event_quality(
    qwin_values,
    window_results_ev,
    window_results_rc,
    cluster_ev,
    omega1=0.5,
    omega2=0.5,
    gamma=0.75,
):
    """Event-level quality Qfinal = omega1 * Qmean + omega2 * Qbest.

    Qmean is the average QWin over the optimal cluster; Qbest is the
    quality of the cluster window with the highest |QWin|.

    :param qwin_values: pre-computed QWin values, aligned with the results lists
    :param window_results_ev: EV results per window (aligned)
    :param window_results_rc: RC results per window (aligned)
    :param cluster_ev: output of :func:`ress.core.clustering.find_optimal_result`
    :returns: dict with Qfinal, Qmean, Qbest, Delta, Omega and Class, or None
    """
    if not cluster_ev or not cluster_ev["success"]:
        return None

    cluster_mask = cluster_ev.get("cluster_mask", None)
    valid_results = cluster_ev.get("valid_results", None)

    if cluster_mask is None or valid_results is None:
        return None

    if len(qwin_values) == 0 or len(window_results_ev) == 0:
        return None

    if len(qwin_values) != len(window_results_ev):
        logger.warning(
            "Array length mismatch - qwin_values:%d vs window_results_ev:%d",
            len(qwin_values),
            len(window_results_ev),
        )
        return None

    # keep only windows that belong to the optimal cluster, matched by window bounds
    cluster_indices = []
    cluster_qwin = []
    for i, ev_res in enumerate(window_results_ev):
        for j, valid_res in enumerate(valid_results):
            if (
                ev_res["window"][0] == valid_res["window"][0]
                and ev_res["window"][1] == valid_res["window"][1]
            ):
                if cluster_mask[j]:
                    cluster_indices.append(i)
                    cluster_qwin.append(qwin_values[i])
                break

    if len(cluster_qwin) == 0:
        return None

    q_mean = np.mean(cluster_qwin)

    # window with the highest |QWin| within the optimal cluster
    best_cluster_idx = np.argmax(np.abs(cluster_qwin))
    best_window_idx = cluster_indices[best_cluster_idx]
    best_ev = window_results_ev[best_window_idx]
    best_rc = window_results_rc[best_window_idx]

    if best_rc is None:
        return None

    q_best_metric = calculate_quality_index(
        phi_rc=best_rc["phi"],
        dt_rc=best_rc["dt"],
        phi_ev=best_ev["phi"],
        dt_ev=best_ev["dt"],
        gamma=gamma,
    )
    q_best = q_best_metric["Q"]

    q_event = omega1 * q_mean + omega2 * q_best

    if q_event >= gamma:
        classification = "Good Split"
    elif q_event <= -gamma:
        classification = "Good Null"
    else:
        classification = "Poor/Noise"

    return {
        "Qfinal": q_event,
        "Qmean": q_mean,
        "Qbest": q_best,
        "N_windows": len(qwin_values),
        "N_cluster_windows": len(cluster_qwin),
        "best_window_idx": best_window_idx,
        "Delta": q_best_metric["Delta"],
        "Omega": q_best_metric["Omega"],
        "Class": classification,
        "omega1": omega1,
        "omega2": omega2,
    }


def calculate_consensus(filter_results, method_name, phi_tolerance=15.0, dt_tolerance=0.020):
    """Average per-filter results that agree with the best filter's result.

    :param filter_results: per-filter results ordered by filter score (best first)
    :param method_name: 'EV' or 'RC' (adds the method-specific QC metric)
    :param phi_tolerance: maximum |phi - phi_best| for inclusion (degrees)
    :param dt_tolerance: maximum |dt - dt_best| for inclusion (seconds)
    :returns: consensus dict, or None if there are no results
    """
    if len(filter_results) == 0:
        return None

    best = filter_results[0]
    to_average = [best]

    logger.debug(
        "%s: Best filter result: phi=%.1f, dt=%.1fms (%s)",
        method_name,
        best["phi"],
        best["dt"] * 1000,
        best["filter"],
    )

    for i, result in enumerate(filter_results[1:], start=2):
        d_phi = abs(result["phi"] - best["phi"])
        if d_phi > 90:  # handle wraparound
            d_phi = 180 - d_phi
        d_dt = abs(result["dt"] - best["dt"])

        within_tolerance = (d_phi <= phi_tolerance) and (d_dt <= dt_tolerance)

        logger.debug(
            "%s: Filter %d check: dphi=%.1f, ddt=%.1fms - %s",
            method_name,
            i,
            d_phi,
            d_dt * 1000,
            "INCLUDED" if within_tolerance else "EXCLUDED",
        )

        if within_tolerance:
            to_average.append(result)

    phi_avg = np.mean([r["phi"] for r in to_average])
    dt_avg = np.mean([r["dt"] for r in to_average])
    phi_err_avg = np.sqrt(np.mean([r["phi_err"] ** 2 for r in to_average]))
    dt_err_avg = np.sqrt(np.mean([r["dt_err"] ** 2 for r in to_average]))

    consensus = {
        "phi": phi_avg,
        "dt": dt_avg,
        "phi_err": phi_err_avg,
        "dt_err": dt_err_avg,
        "spol": best["spol"],  # best filter's values for the scalar metrics
        "CC_FS": best["CC_FS"],
        "CC_NE": best["CC_NE"],
        "T_dominant": best.get("T_dominant", np.nan),
        "n_filters_used": len(to_average),
        "n_clusters": best["n_clusters"],
        "n_in_cluster": best["n_in_cluster"],
        "filter": best["filter"],
    }

    if (
        "cluster_data" in best
        and best["cluster_data"] is not None
        and best["cluster_data"]["success"]
    ):
        consensus["window"] = best["cluster_data"]["result"]["window"]
    elif "window" in best:
        consensus["window"] = best["window"]

    if method_name == "EV" and "L_R" in best:
        consensus["L_R"] = best["L_R"]
    elif method_name == "RC":
        # corrected fast-slow cross-correlation as the RC quality metric
        consensus["CC_FS_corr"] = best.get("CC_FS", np.nan)

    logger.info(
        "%s: Consensus from %d/%d filter(s): phi=%.1f +/- %.1f, dt=%.1f +/- %.1fms",
        method_name,
        len(to_average),
        len(filter_results),
        phi_avg,
        phi_err_avg,
        dt_avg * 1000,
        dt_err_avg * 1000,
    )

    return consensus

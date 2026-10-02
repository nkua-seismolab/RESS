"""Cluster analysis of window measurements (Teanby et al., 2004)."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering


def standardize_z4(X):
    """Standardize a variable per the Z4 method (Milligan and Cooper, 1988)."""
    if np.all(X == X[0]):  # all values the same
        return np.ones_like(X)
    return X / (np.nanmax(X) - np.nanmin(X))


def get_cluster_labels(X, method="ward"):
    """Agglomerative cluster labels for every k from 1 to N-1.

    :param X: 2D array of (dt, phi) normalized coordinates
    :param method: linkage method ('single', 'complete', 'average', 'ward')
    :returns: dict {k: labels}
    """
    N = len(X)
    labels = {}
    for k in range(1, N):
        model = AgglomerativeClustering(n_clusters=k, linkage=method).fit(X)
        labels[k] = model.labels_
    return labels


def get_cluster_data(X, labels):
    """Data points of each cluster at each k.

    :returns: dict {k: {cluster_id: data_points}}
    """
    Z = {k: {} for k in labels.keys()}
    for k in sorted(labels):
        k_labels = labels[k]
        for i in range(k):
            Z[k][i] = X[k_labels == i]
    return Z


def calinski_harabasz_criterion(X, labels, k_max):
    """Calinski-Harabasz scores for k = 2..k_max (higher is better).

    Reference: Calinski & Harabasz (1974), doi: 10.1080/03610927408827101
    """
    k_values = np.arange(2, k_max + 1)
    scores = np.zeros(len(k_values))

    for i, k in enumerate(k_values):
        if k in labels:
            scores[i] = metrics.calinski_harabasz_score(X, labels[k])

    return k_values, scores


def duda_hart_criterion(X, Z, k, ccrit=3.2):
    """Duda-Hart criterion for the optimal number of clusters.

    :param X: 2D array of coordinates
    :param Z: cluster data from :func:`get_cluster_data`
    :param k: number of clusters to test
    :param ccrit: critical value threshold
    :returns: (optimal_k, score) or (None, None) if the criterion is not met

    Reference: Duda & Hart (1973), Pattern Classification and Scene Analysis.
    """
    p = 2  # number of parameters (phi and dt)
    N = len(X)

    for k1, k2 in itertools.combinations(range(k), 2):
        d1 = Z[k][k1]
        d2 = Z[k][k2]
        da = np.concatenate((d1, d2))

        dt1, phi1 = d1[:, 0], d1[:, 1]
        dt2, phi2 = d2[:, 0], d2[:, 1]
        dta, phia = da[:, 0], da[:, 1]

        # within-cluster sums of squares
        s22 = np.sum((dt1 - dt1.mean()) ** 2 + (phi1 - phi1.mean()) ** 2) + np.sum(
            (dt2 - dt2.mean()) ** 2 + (phi2 - phi2.mean()) ** 2
        )
        s21 = np.sum((dta - dta.mean()) ** 2 + (phia - phia.mean()) ** 2)

        c = (1 - (s22 / s21) - (2 / (np.pi * p))) * np.sqrt(
            (N * p) / (2 * (1 - (8 / (np.pi**2 * p))))
        )

        if c <= ccrit:
            return k, c

    return None, None


def cluster_variance(C, V):
    """Within-cluster and mean data variances of one cluster.

    :param C: cluster data (N x 2 array of dt, phi)
    :param V: errors (N x 2 array of sdt, sphi)
    :returns: (within_cluster_var, mean_data_var, overall_var)
    """
    dt, phi = C[:, 0], C[:, 1]
    sdt, sphi = V[:, 0], V[:, 1]
    Nj = len(C)

    s2cj = np.nansum((dt - np.nanmean(dt)) ** 2 + (phi - np.nanmean(phi)) ** 2) / Nj

    # mean data variance (guard against division by zero)
    sdt_sq_inv = 1 / (sdt**2)
    sphi_sq_inv = 1 / (sphi**2)
    sdt_sq_inv[~np.isfinite(sdt_sq_inv)] = 0
    sphi_sq_inv[~np.isfinite(sphi_sq_inv)] = 0

    sum_sdt = np.nansum(sdt_sq_inv)
    sum_sphi = np.nansum(sphi_sq_inv)

    s2dj = (1 / sum_sdt) + (1 / sum_sphi)

    s2oj = np.nanmax([s2cj, s2dj])

    return s2cj, s2dj, s2oj


def find_optimal_result(results, linkage="ward", k_max=5, n_min=3, ccrit=3.2):
    """Teanby et al. (2004) cluster analysis over window measurements.

    :param results: list of dicts from silver_and_chan()
    :param linkage: clustering linkage method
    :param k_max: maximum number of clusters to consider
    :param n_min: minimum number of points in a valid cluster
    :param ccrit: Duda-Hart critical value
    :returns: dict with the optimal result and clustering diagnostics

    Reference: Teanby, Kendall & van der Baan (2004), doi: 10.1785/0120030123
    """
    # discard measurements with NaN errors
    valid_results = [r for r in results if not np.isnan(r["phi_err"]) and not np.isnan(r["dt_err"])]

    if len(valid_results) < 3:
        return {"success": False, "reason": "Not enough valid measurements"}

    phis = np.array([r["phi"] for r in valid_results])
    dts = np.array([r["dt"] for r in valid_results])
    sphis = np.array([r["phi_err"] for r in valid_results])
    sdts = np.array([r["dt_err"] for r in valid_results])

    n_phis = standardize_z4(phis)
    n_dts = standardize_z4(dts)
    n_sphis = standardize_z4(sphis)
    n_sdts = standardize_z4(sdts)

    orig = np.column_stack((dts, phis))  # original data
    U = np.column_stack((sdts, sphis))  # original errors
    X = np.column_stack((n_dts, n_phis))  # normalized data
    S = np.column_stack((n_sdts, n_sphis))  # normalized errors

    L = get_cluster_labels(X, method=linkage)
    Z = get_cluster_data(X, L)

    # optimal number of clusters via Duda-Hart
    M_dh = np.inf
    for k in sorted(Z.keys()):
        if k < 2:
            continue
        M_dh, _ = duda_hart_criterion(X, Z, k, ccrit=ccrit)
        if M_dh is not None:
            break

    M_max = min(M_dh if M_dh is not None else k_max, k_max)

    # Calinski-Harabasz selects the final k
    k_vals, ch_scores = calinski_harabasz_criterion(X, L, M_max)

    if len(ch_scores) == 0 or np.all(ch_scores == 0):
        M_opt = 1
    else:
        best_idx = np.argmax(ch_scores)
        M_opt = k_vals[best_idx]

    # keep clusters with at least n_min points
    cluster_labels = L[M_opt]
    valid_clusters = []
    for c_id in range(M_opt):
        n_points = np.sum(cluster_labels == c_id)
        if n_points >= n_min:
            valid_clusters.append(c_id)

    if len(valid_clusters) == 0:
        valid_clusters = list(range(M_opt))

    # cluster with the minimum overall variance
    var_dict = {}
    for c_id in valid_clusters:
        mask = cluster_labels == c_id
        D = X[mask]
        V = S[mask]
        if len(D) > 0:
            _, _, var_dict[c_id] = cluster_variance(D, V)

    if len(var_dict) == 0:
        return {"success": False, "reason": "No valid clusters found"}

    C_opt = min(var_dict, key=var_dict.get)

    opt_mask = cluster_labels == C_opt
    D_opt = orig[opt_mask]
    U_opt = U[opt_mask]
    S_opt = S[opt_mask]

    # measurement with the minimum combined normalized error
    S_comb = S_opt[:, 0] + S_opt[:, 1]
    opt_idx = np.argmin(S_comb)

    valid_indices = np.where(opt_mask)[0]
    best_result_idx = valid_indices[opt_idx]
    best_result = valid_results[best_result_idx]

    return {
        "success": True,
        "result": best_result,
        "phi": D_opt[opt_idx, 1],
        "dt": D_opt[opt_idx, 0],
        "phi_err": U_opt[opt_idx, 1],
        "dt_err": U_opt[opt_idx, 0],
        "n_clusters": M_opt,
        "optimal_cluster": C_opt,
        "n_valid_measurements": len(valid_results),
        "n_in_cluster": np.sum(opt_mask),
        "all_phis": phis,
        "all_dts": dts,
        "cluster_labels": cluster_labels,
        "cluster_mask": opt_mask,  # boolean mask for windows in the optimal cluster
        "valid_results": valid_results,  # for mapping back to the original windows
        "ch_scores": (k_vals, ch_scores) if len(ch_scores) > 0 else None,
    }

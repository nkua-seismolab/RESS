"""Shear-wave splitting measurement methods.

Eigenvalue method (Silver & Chan, 1991) and rotation-correlation method
(Bowman & Ando, 1987), with error estimation after Walsh et al. (2013).
"""

from __future__ import annotations

import itertools

import numpy as np
from obspy.signal.cross_correlation import correlate
from obspy.signal.util import next_pow_2
from scipy.stats import f as fdistr
from scipy.stats import norm as norm_dist

from ress.core.quality import calculate_quality_index


def Rmatrix2D(angle):
    """2D clockwise rotation matrix for an angle in degrees."""
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, s], [-s, c]])


def Rmatrix3D(bazi, ain):
    """ZEN -> LQT rotation matrix (Plesinger et al., 1986).

    :param bazi: backazimuth in degrees
    :param ain: incidence angle in degrees
    """
    baz_rad = np.deg2rad(bazi)
    ain_rad = np.deg2rad(ain)

    cb, sb = np.cos(baz_rad), np.sin(baz_rad)
    ca, sa = np.cos(ain_rad), np.sin(ain_rad)

    return np.array(
        [
            [ca, -sa * sb, -sa * cb],  # L
            [sa, ca * sb, ca * cb],  # Q
            [0, -cb, sb],  # T
        ]
    )


def xcStatic(x, y):
    """Normalized correlation of two signals at zero time-shift."""
    return correlate(x, y, shift=0, demean=True, normalize="naive")[0]


def NDF(data):
    """Degrees of freedom of a signal, corrected after Walsh et al. (2013)."""
    npts = data.size
    nfft = next_pow_2(npts)
    A = np.abs(np.fft.fft(data, nfft))

    F2 = (A**2).sum() - ((A[0] ** 2) - (A[-1] ** 2)) / 2
    F4 = (4 / 3) * (A**4).sum() - ((A[0] ** 4) - (A[-1] ** 4)) / 3  # Walsh et al. (2013) eq. 28

    N = int(round((2 * (F2**2) / F4) - 1))
    return min(N, npts)  # cannot exceed number of points


def confRegion(cMin, N, K=2, alpha=0.05):
    """100*(1-alpha)% confidence threshold for the critical value.

    :param cMin: minimum critical value (eigenvalue ratio or energy)
    :param N: degrees of freedom
    :param K: number of parameters (phi and time-delay = 2)
    :param alpha: 0.05 for the 95% confidence region
    """
    return cMin * (1 + ((K / (N - K)) * fdistr.isf(alpha, K, N - K)))


def estimate_errors(td, phi, td_trials, phi_trials, c_norm):
    """Approximate 1-sigma errors from the 95% confidence contour.

    :param c_norm: normalized critical array (values <= 1 are within the CI)
    :returns: (phi_error, td_error) with td_error in samples
    """
    within_ci = c_norm <= 1.0

    phi_in_ci = phi_trials[np.any(within_ci, axis=1)]
    td_in_ci = td_trials[np.any(within_ci, axis=0)]

    if len(phi_in_ci) > 0:
        phi_err = (phi_in_ci.max() - phi_in_ci.min()) / 4.0  # approximate 1-sigma
    else:
        phi_err = np.nan

    if len(td_in_ci) > 0:
        td_err = (td_in_ci.max() - td_in_ci.min()) / 4.0  # approximate 1-sigma
    else:
        td_err = np.nan

    return phi_err, td_err


def _rotate_to_qt(ZNE, bazi, ain):
    """Rotate ZNE to QT (via LQT when an incidence angle is given, else RT)."""
    if ain is not None and ain != 0:
        M = Rmatrix3D(bazi, ain)
        ZEN = ZNE.copy()
        ZEN[1] = ZNE[2].copy()  # swap N<->E for ZEN order
        ZEN[2] = ZNE[1].copy()
        LQT = np.dot(M, ZEN)
        return LQT[1:]  # Q and T components
    M = Rmatrix2D(bazi)
    return np.dot(M, ZNE[1:])


def _extract_zne(stream):
    return np.array(
        [
            stream.select(component="Z")[0].data,
            stream.select(component="N")[0].data,
            stream.select(component="E")[0].data,
        ]
    )


def silver_and_chan(stream1, bazi, pickwin, ain=None, max_delay=0.250, method="EV"):
    """Silver & Chan (1991) eigenvalue method for shear-wave splitting.

    :param stream1: ObsPy Stream with ZNE components
    :param bazi: backazimuth in degrees
    :param pickwin: (start, end) of the analysis window in seconds from stream start
    :param ain: incidence angle in degrees; None uses ZRT instead of LQT rotation
    :param max_delay: maximum time-delay of the grid search in seconds
    :param method: 'EV' for eigenvalue or 'ME' for minimum energy
    :returns: dict with phi, dt, errors, spol, correlations and grid diagnostics
    """
    stream = stream1.copy()

    sps = stream[0].stats.sampling_rate
    delta = 1.0 / sps
    max_delay_samp = int(max_delay * sps)

    window_length = pickwin[1] - pickwin[0]
    if window_length <= max_delay + 1e-6:
        raise ValueError(
            f"Window length ({window_length:.3f}s) must exceed max_delay ({max_delay:.3f}s)!"
        )

    step = 1  # 1 degree for phi, 1 sample for td
    phi_trials = np.arange(-90, 91, step, dtype=int)
    td_trials = np.arange(0, max_delay_samp + step, step, dtype=int)

    n_phi = len(phi_trials)
    n_td = len(td_trials)

    E_array = np.zeros((n_phi, n_td))  # energy in transverse
    L2_array = np.zeros((n_phi, n_td))  # second eigenvalue
    L1_array = np.zeros((n_phi, n_td))  # first eigenvalue
    V1_array = np.zeros((n_phi, n_td, 2))  # principal eigenvector

    ZNE = _extract_zne(stream)
    QT = _rotate_to_qt(ZNE, bazi, ain)

    sel_window = np.array([int(pickwin[0] * sps), int(pickwin[1] * sps)])

    for i, j in itertools.product(range(n_phi), range(n_td)):
        phi = phi_trials[i]
        td = td_trials[j]

        del_window = sel_window + td

        # rotate QT to fast-slow, delay the slow component, rotate back
        R = Rmatrix2D(phi)
        FS1 = np.dot(R, QT)
        FS2 = np.array(
            [
                FS1[0][sel_window[0] : sel_window[1]],
                FS1[1][del_window[0] : del_window[1]],
            ]
        )
        QT2 = np.dot(R.T, FS2)

        E_array[i, j] = np.sum(QT2[1] ** 2)

        C = np.cov(QT2[0], QT2[1])
        w, v = np.linalg.eig(C)

        L2_array[i, j] = w.min()
        L1_array[i, j] = w.max()
        V1_array[i, j] = v[:, np.argmax(w)]

    L_array = L2_array / L1_array

    c_array = L_array if method == "EV" else E_array

    idx = np.unravel_index(np.nanargmin(c_array), c_array.shape)
    phi_sc = phi_trials[idx[0]]
    td_sc = td_trials[idx[1]]
    L_R = L_array[idx]  # eigenvalue ratio at the optimal solution

    # corrected waveforms at the optimal solution
    R = Rmatrix2D(phi_sc)
    del_window = sel_window + td_sc

    FS_corr1 = np.dot(R, QT)
    FS_corr2 = np.array(
        [FS_corr1[0][sel_window[0] : sel_window[1]], FS_corr1[1][del_window[0] : del_window[1]]]
    )
    QT_corr = np.dot(R.T, FS_corr2)

    Tvar = np.var(QT_corr[1] / np.abs(QT_corr[1]).max())

    N = NDF(QT_corr[1])
    K = 2  # number of parameters (phi and td)

    if N <= K:
        c95 = c_array[idx]
        phi_err = np.nan
        dt_err = np.nan
    else:
        c95 = confRegion(c_array.min(), N, K)
        c_norm = c_array / c95
        phi_err, dt_err = estimate_errors(td_sc, phi_sc, td_trials, phi_trials, c_norm)

    # convert phi to geographic coordinates
    phi_final = (phi_sc + bazi) % 180

    R_final = Rmatrix2D(phi_final)
    FS_final = np.dot(R_final, ZNE[1:])
    FS_win = np.array(
        [FS_final[0][sel_window[0] : sel_window[1]], FS_final[1][del_window[0] : del_window[1]]]
    )
    CC_FS = xcStatic(FS_win[0], FS_win[1])

    NE_corr = np.dot(R_final.T, FS_win)
    CC_NE = xcStatic(NE_corr[0], NE_corr[1])

    # S-wave polarization from the corrected waveforms
    C_ne = np.cov(NE_corr[0], NE_corr[1])
    w_ne, v_ne = np.linalg.eig(C_ne)
    V1_ne = v_ne[:, np.argmax(w_ne)]
    spol = np.rad2deg(np.arctan2(V1_ne[1], V1_ne[0])) % 180

    dt_sc = td_sc * delta
    dt_err = dt_err * delta if not np.isnan(dt_err) else np.nan

    return {
        "phi": float(phi_final),
        "dt": float(dt_sc),
        "phi_err": float(phi_err),
        "dt_err": float(dt_err),
        "spol": float(spol),
        "CC_FS": float(CC_FS),
        "CC_NE": float(CC_NE),
        "Tvar": float(Tvar),
        "c95": float(c95),
        "L_R": float(L_R),
        "cArray": c_array,
        "phiTrials": phi_trials,
        "tdTrials": td_trials,
    }


def r2z(r):
    """Fisher transformation from R-space to Z-space."""
    return np.log((1 + r) / (1 - r)) / 2.0


def z2r(z):
    """Inverse Fisher transformation from Z-space to R-space."""
    e = np.exp(2 * z)
    return (e - 1) / (e + 1)


def lowCI(r, alpha, n):
    """Lower bound of the 100*(1-alpha)% CI of a correlation coefficient."""
    z = r2z(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm_dist.ppf(1 - alpha)
    lo = z - z_crit * se
    return z2r(lo)


def rotation_correlation(stream1, bazi, pickwin, ain=None, max_delay=0.250):
    """Rotation-correlation method for shear-wave splitting (Bowman & Ando, 1987).

    Searches for the phi and time-delay that maximize the cross-correlation
    between the fast and slow components.

    :param stream1: ObsPy Stream with ZNE components
    :param bazi: backazimuth in degrees
    :param pickwin: (start, end) of the analysis window in seconds from stream start
    :param ain: incidence angle in degrees; None uses ZRT instead of LQT rotation
    :param max_delay: maximum time-delay of the grid search in seconds
    :returns: dict with phi, dt, errors, spol, correlations and grid diagnostics
    """
    stream = stream1.copy()

    sps = stream[0].stats.sampling_rate
    delta = 1.0 / sps
    max_delay_samp = int(np.ceil(max_delay * sps))

    window_length = pickwin[1] - pickwin[0]
    if window_length <= max_delay + 1e-6:
        raise ValueError(
            f"Window length ({window_length:.3f}s) must exceed max_delay ({max_delay:.3f}s)!"
        )

    step = 1
    phi_trials = np.arange(-90, 91, step, dtype=int)
    td_trials = np.arange(0, max_delay_samp + step, step, dtype=int)

    n_phi = len(phi_trials)
    n_td = len(td_trials)

    c_array = np.zeros((n_phi, n_td))

    ZNE = _extract_zne(stream)
    QT = _rotate_to_qt(ZNE, bazi, ain)

    sel_window = np.array([int(pickwin[0] * sps), int(pickwin[1] * sps)])

    for i, j in itertools.product(range(n_phi), range(n_td)):
        phi = phi_trials[i]
        td = td_trials[j]

        del_window = sel_window + td

        R = Rmatrix2D(phi)
        FS1 = np.dot(R, QT)
        FS2 = np.array(
            [
                FS1[0][sel_window[0] : sel_window[1]],
                FS1[1][del_window[0] : del_window[1]],
            ]
        )

        c_array[i, j] = xcStatic(FS2[0], FS2[1])

    # use absolute value (anticorrelated solutions count too)
    c_array = np.abs(c_array)

    idx = np.unravel_index(np.nanargmax(c_array), c_array.shape)
    phi_rc = phi_trials[idx[0]]
    td_rc = td_trials[idx[1]]
    cc_max = c_array[idx]

    c95 = lowCI(cc_max, 0.05, c_array.size - 1)
    c_norm = c_array / c95

    phi_err, dt_err = estimate_errors(td_rc, phi_rc, td_trials, phi_trials, c_norm)

    # convert phi to geographic coordinates
    phi_final = (phi_rc + bazi) % 180

    R_final = Rmatrix2D(phi_final)
    del_window = sel_window + td_rc

    FS_corr = np.dot(R_final, ZNE[1:])
    FS_win = np.array(
        [FS_corr[0][sel_window[0] : sel_window[1]], FS_corr[1][del_window[0] : del_window[1]]]
    )
    CC_FS = xcStatic(FS_win[0], FS_win[1])

    NE_corr = np.dot(R_final.T, FS_win)
    CC_NE = xcStatic(NE_corr[0], NE_corr[1])

    C_ne = np.cov(NE_corr[0], NE_corr[1])
    w_ne, v_ne = np.linalg.eig(C_ne)
    V1_ne = v_ne[:, np.argmax(w_ne)]
    spol = np.rad2deg(np.arctan2(V1_ne[1], V1_ne[0])) % 180

    dt_rc = td_rc * delta
    dt_err = dt_err * delta if not np.isnan(dt_err) else np.nan

    return {
        "phi": float(phi_final),
        "dt": float(dt_rc),
        "phi_err": float(phi_err),
        "dt_err": float(dt_err),
        "spol": float(spol),
        "CC_FS": float(CC_FS),
        "CC_NE": float(CC_NE),
        "Tvar": np.nan,  # not applicable for RC
        "c95": float(c95),
        "cArray": c_array,
        "phiTrials": phi_trials,
        "tdTrials": td_trials,
    }


def calculate_dominant_period(trace_data, sampling_rate):
    """Dominant period of a signal from the FFT power spectrum, in seconds."""
    fft = np.fft.rfft(trace_data)
    freqs = np.fft.rfftfreq(len(trace_data), 1.0 / sampling_rate)

    power = np.abs(fft[1:]) ** 2  # skip the DC component

    peak_idx = np.argmax(power) + 1
    f_dominant = freqs[peak_idx]

    if f_dominant > 0:
        return 1.0 / f_dominant
    return np.nan


def process_window(args):
    """Run both EV and RC on one window and compute its quality index QWin.

    Executed in a worker pool; args is a picklable tuple
    (st_filtered, bazi, pickwin, ain, max_delay_s, gamma).

    :returns: dict with 'ev', 'rc', 'qwin', 'window' and 'T_dominant'
    """
    st_filtered, bazi, pickwin, ain, max_delay_s, gamma = args

    result = {"ev": None, "rc": None, "qwin": None, "window": pickwin, "T_dominant": np.nan}

    # adaptive mode: derive max_delay from the dominant period of the window
    max_delay_adaptive = max_delay_s
    if max_delay_s is None or max_delay_s == 0:
        try:
            sps = st_filtered[0].stats.sampling_rate

            n_data = st_filtered.select(component="N")[0].data
            e_data = st_filtered.select(component="E")[0].data

            R_bazi = Rmatrix2D(bazi)
            NE = np.array([n_data, e_data])
            RT = np.dot(R_bazi, NE)

            i_start = int(pickwin[0] * sps)
            i_end = int(pickwin[1] * sps)
            r_windowed = RT[0][i_start:i_end]
            t_windowed = RT[1][i_start:i_end]

            T_r = calculate_dominant_period(r_windowed, sps)
            T_t = calculate_dominant_period(t_windowed, sps)

            if not np.isnan(T_r) and not np.isnan(T_t):
                T_dominant = (T_r + T_t) / 2.0
            elif not np.isnan(T_r):
                T_dominant = T_r
            elif not np.isnan(T_t):
                T_dominant = T_t
            else:
                T_dominant = np.nan

            result["T_dominant"] = T_dominant

            if not np.isnan(T_dominant) and T_dominant > 0:
                max_delay_adaptive = T_dominant / 2.0
            else:
                max_delay_adaptive = 0.3
        except Exception:
            max_delay_adaptive = 0.3

    try:
        ev_result = silver_and_chan(
            st_filtered,
            bazi=bazi,
            pickwin=pickwin,
            ain=ain,
            max_delay=max_delay_adaptive,
            method="EV",
        )
        ev_result["window"] = pickwin
        ev_result["T_dominant"] = result["T_dominant"]
        result["ev"] = ev_result
    except Exception:
        pass

    try:
        rc_result = rotation_correlation(
            st_filtered, bazi=bazi, pickwin=pickwin, ain=ain, max_delay=max_delay_adaptive
        )
        rc_result["window"] = pickwin
        rc_result["T_dominant"] = result["T_dominant"]
        result["rc"] = rc_result
    except Exception:
        pass

    # QWin only when both methods produced valid errors
    if result["ev"] and result["rc"]:
        if (
            not np.isnan(result["ev"].get("phi_err", np.nan))
            and not np.isnan(result["ev"].get("dt_err", np.nan))
            and not np.isnan(result["rc"].get("phi_err", np.nan))
            and not np.isnan(result["rc"].get("dt_err", np.nan))
        ):
            qwin_result = calculate_quality_index(
                phi_rc=result["rc"]["phi"],
                dt_rc=result["rc"]["dt"],
                phi_ev=result["ev"]["phi"],
                dt_ev=result["ev"]["dt"],
                gamma=gamma,
            )
            result["qwin"] = qwin_result["Q"]

    return result

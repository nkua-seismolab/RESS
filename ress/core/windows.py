"""Analysis window generation, SNR measurement and adaptive filter selection."""

from __future__ import annotations

import itertools
import logging

import numpy as np

logger = logging.getLogger(__name__)


def make_windows(
    s_pick,
    max_delay,
    tbeg0=-0.2,
    tbeg1=-0.05,
    tend0=0.1,
    tend1=0.5,
    dtbeg=0.02,
    dtend=0.02,
    nbeg=10,
    nend=20,
):
    """Generate analysis windows around the S pick (Teanby et al., 2004).

    All window parameters are relative to the S pick, in seconds.

    :param s_pick: S-arrival time relative to stream start (s)
    :param max_delay: maximum splitting time-delay; windows must exceed it (s)
    :param tbeg0: earliest window start
    :param tbeg1: latest window start
    :param tend0: earliest window end
    :param tend1: latest window end
    :param dtbeg: step between window starts
    :param dtend: step between window ends
    :param nbeg: number of start points
    :param nend: number of end points
    :returns: array of (start, end) pairs in absolute stream time, sorted by
        window length (ascending)
    """
    windows = []

    for i, j in itertools.product(range(nbeg), range(nend)):
        i += 1
        j += 1
        tbeg = tbeg1 - (i - 1) * dtbeg
        tend = tend0 + (j - 1) * dtend

        if (tbeg < tbeg0) or (tend > tend1):
            continue

        # window length must strictly exceed max_delay (epsilon for float precision)
        if (tend - tbeg) <= max_delay + 1e-6:
            continue

        windows.append((tbeg, tend))

    windows = np.asarray(windows)

    if len(windows) > 0:
        dur = windows[:, 1] - windows[:, 0]
        windows = windows[np.argsort(dur)]
        windows = windows + s_pick  # convert to absolute stream time

    return windows


def calculate_snr(stream, s_time):
    """SNR as the RMS signal/noise ratio averaged over the components.

    Noise window: S-arrival - 1.0 s to S-arrival - 0.1 s
    Signal window: S-arrival - 0.1 s to S-arrival + 1.0 s

    :param stream: ObsPy Stream (filtered)
    :param s_time: S-arrival time (UTCDateTime)
    """
    snr_values = []

    t_noise_start = s_time - 1.0
    t_noise_end = s_time - 0.1
    t_sig_start = s_time - 0.1
    t_sig_end = s_time + 1.0

    for trace in stream:
        try:
            noise_data = trace.slice(t_noise_start, t_noise_end).data
            sig_data = trace.slice(t_sig_start, t_sig_end).data

            if len(noise_data) < 10 or len(sig_data) < 10:
                continue

            rms_noise = np.sqrt(np.mean(noise_data**2))
            rms_signal = np.sqrt(np.mean(sig_data**2))

            if rms_noise > 0:
                snr_values.append(rms_signal / rms_noise)

        except Exception:
            continue

    if not snr_values:
        return 0.0
    return np.mean(snr_values)


def adaptive_filter(stream, s_time, filter_bank, snr_min):
    """Score every filter of the bank following Savage et al. (2010).

    Filters are scored by SNR * bandwidth(octaves) to balance signal
    quality against filter width (narrow bands are prone to cycle skipping).

    :param stream: ObsPy Stream (raw, unfiltered)
    :param s_time: S-arrival time (UTCDateTime)
    :param filter_bank: iterable of (f_low, f_high) pairs in Hz
    :param snr_min: minimum acceptable SNR
    :returns: list of dicts with freqmin, freqmax, snr, bandwidth, score,
        stream and passes_snr, sorted by score (best first)
    """
    sps = stream[0].stats.sampling_rate
    nyquist = sps / 2.0

    results = []

    for f_low, f_high in filter_bank:
        # skip filters near Nyquist
        if f_high >= nyquist * 0.9:
            logger.debug("Filter %.1f-%.1f Hz skipped (Nyquist)", f_low, f_high)
            continue

        try:
            st_filt = stream.copy()
            st_filt.filter("bandpass", freqmin=f_low, freqmax=f_high, corners=2, zerophase=True)
        except Exception:
            logger.debug("Filter %.1f-%.1f Hz failed", f_low, f_high)
            continue

        snr = calculate_snr(st_filt, s_time)
        bw_octaves = np.log2(f_high / f_low)
        score = snr * bw_octaves
        passes_snr = snr >= snr_min

        logger.debug(
            "Filter %.1f-%.1f Hz | SNR %.2f | BW %.2f oct | score %.2f | %s",
            f_low,
            f_high,
            snr,
            bw_octaves,
            score,
            "OK" if passes_snr else f"SNR<{snr_min}",
        )

        results.append(
            {
                "freqmin": f_low,
                "freqmax": f_high,
                "snr": snr,
                "bandwidth": bw_octaves,
                "score": score,
                "stream": st_filt,
                "passes_snr": passes_snr,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)

    best_passed = next((r for r in results if r["passes_snr"]), None)
    if best_passed:
        logger.debug(
            "Best filter: %.1f-%.1f Hz (score=%.2f)",
            best_passed["freqmin"],
            best_passed["freqmax"],
            best_passed["score"],
        )

    return results

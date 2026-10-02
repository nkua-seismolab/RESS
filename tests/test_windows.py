"""Tests for ress.core.windows."""

import numpy as np
from obspy import Stream, Trace, UTCDateTime

from ress.core.windows import adaptive_filter, calculate_snr, make_windows


def test_make_windows_bounds_and_sorting():
    windows = make_windows(
        5.0,
        max_delay=0.4,
        tbeg0=-0.50,
        tbeg1=-0.10,
        tend0=0.10,
        tend1=0.40,
        dtbeg=0.05,
        dtend=0.01,
        nbeg=10,
        nend=20,
    )
    assert len(windows) > 0
    durations = windows[:, 1] - windows[:, 0]
    # every window strictly exceeds max_delay and is sorted by length
    assert np.all(durations > 0.4)
    assert np.all(np.diff(durations) >= -1e-9)
    # bounds are relative to the pick at 5.0 s
    assert np.all(windows[:, 0] >= 5.0 - 0.50 - 1e-9)
    assert np.all(windows[:, 1] <= 5.0 + 0.40 + 1e-9)


def test_make_windows_none_when_too_short():
    windows = make_windows(
        5.0,
        max_delay=2.0,
        tbeg0=-0.50,
        tbeg1=-0.10,
        tend0=0.10,
        tend1=0.40,
        dtbeg=0.05,
        dtend=0.01,
        nbeg=10,
        nend=20,
    )
    assert len(windows) == 0


def _snr_stream(snr_amplitude):
    """Noise-only before the pick, sine of given amplitude after."""
    sps = 100.0
    rng = np.random.default_rng(1)
    t = np.arange(0, 20, 1 / sps)
    start = UTCDateTime(2026, 1, 1)
    s_time = start + 10.0

    traces = []
    for comp in "ZNE":
        data = rng.normal(0, 1.0, t.size)
        signal_mask = t >= 9.9
        data[signal_mask] += snr_amplitude * np.sin(2 * np.pi * 5.0 * t[signal_mask])
        tr = Trace(data=data)
        tr.stats.sampling_rate = sps
        tr.stats.starttime = start
        tr.stats.channel = "HH" + comp
        traces.append(tr)
    return Stream(traces), s_time


def test_calculate_snr_high_vs_low():
    st_high, s_time = _snr_stream(20.0)
    st_low, _ = _snr_stream(0.1)
    assert calculate_snr(st_high, s_time) > 5.0
    assert calculate_snr(st_low, s_time) < 3.0


def test_adaptive_filter_scores_and_sorting():
    st, s_time = _snr_stream(20.0)
    results = adaptive_filter(st, s_time, [(1.0, 10.0), (0.5, 2.0), (40.0, 49.0)], snr_min=3.0)

    # the 40-49 Hz filter is dropped (>= 0.9 * Nyquist)
    assert all(r["freqmax"] < 45.0 for r in results)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # the 5 Hz signal passes in the 1-10 Hz band
    best = results[0]
    assert best["freqmin"] == 1.0
    assert best["passes_snr"]

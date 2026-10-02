"""Tests for ress.core.splitting: recovery of known splitting parameters."""

import numpy as np
import pytest
from obspy import Stream, Trace

from ress.core.splitting import (
    Rmatrix2D,
    calculate_dominant_period,
    process_window,
    rotation_correlation,
    silver_and_chan,
)

SPS = 100.0
S_ARRIVAL = 5.0  # seconds from stream start


def ricker(t, t0, f0):
    """Ricker wavelet centered at t0 with peak frequency f0."""
    a = (np.pi * f0 * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def make_split_stream(phi_deg, dt_s, spol_deg, bazi=0.0, noise=1e-4, seed=0):
    """Synthesize a ZNE stream with a split S wave of known (phi, dt).

    The source wavelet with polarization spol is projected onto the fast
    (phi) and slow axes and the slow component is delayed by dt; the fast
    and slow components are then rotated back to N/E.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, 10, 1 / SPS)
    f0 = 4.0

    angle = np.deg2rad(spol_deg - phi_deg)
    fast = ricker(t, S_ARRIVAL, f0) * np.cos(angle)
    slow = ricker(t, S_ARRIVAL + dt_s, f0) * np.sin(angle)

    # [N, E] = R(phi).T @ [F, S]
    R = Rmatrix2D(phi_deg)
    NE = np.dot(R.T, np.array([fast, slow]))

    traces = []
    for data, comp in ((rng.normal(0, noise, t.size), "Z"), (NE[0], "N"), (NE[1], "E")):
        tr = Trace(data=data + rng.normal(0, noise, t.size))
        tr.stats.sampling_rate = SPS
        tr.stats.channel = "HH" + comp
        traces.append(tr)
    return Stream(traces)


PICKWIN = (S_ARRIVAL - 0.4, S_ARRIVAL + 0.8)


def angular_diff(a, b):
    """Smallest difference between two axial angles (mod 180)."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def test_silver_and_chan_recovers_split():
    phi_true, dt_true, spol = 40.0, 0.12, 70.0
    st = make_split_stream(phi_true, dt_true, spol)

    result = silver_and_chan(st, bazi=0.0, pickwin=PICKWIN, max_delay=0.25)

    assert angular_diff(result["phi"], phi_true) < 5.0
    assert abs(result["dt"] - dt_true) < 0.02


def test_silver_and_chan_with_backazimuth():
    phi_true, dt_true, spol = 40.0, 0.12, 70.0
    st = make_split_stream(phi_true, dt_true, spol)

    result = silver_and_chan(st, bazi=30.0, pickwin=PICKWIN, max_delay=0.25)

    assert angular_diff(result["phi"], phi_true) < 5.0
    assert abs(result["dt"] - dt_true) < 0.02


def test_rotation_correlation_recovers_split():
    phi_true, dt_true, spol = -30.0, 0.10, 10.0
    st = make_split_stream(phi_true, dt_true, spol)

    result = rotation_correlation(st, bazi=0.0, pickwin=PICKWIN, max_delay=0.25)

    assert angular_diff(result["phi"], phi_true) < 5.0
    assert abs(result["dt"] - dt_true) < 0.02


def test_window_shorter_than_max_delay_rejected():
    st = make_split_stream(40.0, 0.12, 70.0)
    with pytest.raises(ValueError, match="must exceed max_delay"):
        silver_and_chan(st, bazi=0.0, pickwin=(5.0, 5.2), max_delay=0.25)


def test_process_window_computes_qwin():
    st = make_split_stream(40.0, 0.12, 70.0)
    result = process_window((st, 0.0, PICKWIN, None, 0.25, 0.75))

    assert result["ev"] is not None
    assert result["rc"] is not None
    # EV and RC agree on a clean synthetic split -> positive quality
    assert result["qwin"] is not None
    assert result["qwin"] > 0.5


def test_calculate_dominant_period():
    t = np.arange(0, 10, 1 / SPS)
    data = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz -> 0.5 s
    period = calculate_dominant_period(data, SPS)
    assert abs(period - 0.5) < 0.05


def test_read_only_velocity_model(monkeypatch, tmp_path):
    import shutil
    from pathlib import Path

    from ress.core import raypath

    source = tmp_path / "model.nd"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "velocity_models/model_rigo1996_half.nd", source
    )
    build = raypath.build_taup_model

    def read_only_build(model_path, directory, verbose=False):
        if Path(directory) == tmp_path:
            raise PermissionError("read-only model directory")
        return build(model_path, directory, verbose=verbose)

    monkeypatch.setattr(raypath, "build_taup_model", read_only_build)
    calculator = raypath.RaypathCalculator(str(source), max_ain=45, min_tkf=90)
    assert calculator.model.get_travel_times(5, 0.1, phase_list=["s", "S"])
    assert not source.with_suffix(".npz").exists()

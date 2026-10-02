"""Boundary dataclasses exchanged between the client and core modules."""

from __future__ import annotations

from dataclasses import dataclass, field

from obspy import UTCDateTime


@dataclass
class SwsTask:
    """Everything the core needs to run splitting analysis on one S pick.

    Station coordinates and channel orientations are snapshotted from the
    SeisComP inventory on the main thread so the worker process never
    touches SeisComP objects.
    """

    pick_id: str
    seed_id: str  # NET.STA.LOC.CHA
    pick_time: UTCDateTime
    # event / origin metadata
    event_id: str
    origin_id: str
    origin_time: UTCDateTime
    event_latitude: float
    event_longitude: float
    event_depth_km: float
    magnitude: float | None = None
    magnitude_type: str | None = None
    # station metadata
    station_latitude: float | None = None
    station_longitude: float | None = None
    station_elevation_m: float | None = None
    # channel code -> (azimuth_deg, dip_deg) for ->ZNE rotation and QC
    channel_orientations: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class SwsResult:
    """Outcome of the splitting analysis for one S pick."""

    pick_id: str
    error: str | None = None

    # path geometry (worker-computed)
    epi_distance_km: float | None = None
    hypo_distance_km: float | None = None
    backazimuth: float | None = None
    incidence_angle: float | None = None
    takeoff_angle: float | None = None
    s_travel_time_s: float | None = None

    # eigenvalue method (Silver & Chan, 1991) - primary
    ev_phi: float | None = None
    ev_phi_err: float | None = None
    ev_dt_ms: float | None = None
    ev_dt_err_ms: float | None = None
    ev_spol: float | None = None
    ev_cc_fs: float | None = None
    ev_cc_ne: float | None = None
    ev_linearity: float | None = None

    # rotation-correlation method (Bowman & Ando, 1987) - supportive
    rc_phi: float | None = None
    rc_phi_err: float | None = None
    rc_dt_ms: float | None = None
    rc_dt_err_ms: float | None = None
    rc_spol: float | None = None
    rc_cc_fs: float | None = None
    rc_cc_ne: float | None = None

    # quality (Wuestefeld et al., 2010)
    qfinal: float | None = None
    qmean: float | None = None
    qbest: float | None = None
    delta: float | None = None
    omega: float | None = None
    quality_class: str | None = None

    # clustering / processing diagnostics
    n_windows: int | None = None
    n_cluster_windows: int | None = None
    n_clusters: int | None = None
    n_in_cluster: int | None = None
    filter_low: float | None = None
    filter_high: float | None = None
    snr: float | None = None
    dominant_period_s: float | None = None
    rotation_mode: str | None = None
    consensus_used: bool = False
    n_filters_used: int | None = None

    # per-filter / per-cluster diagnostics persisted as JSON
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

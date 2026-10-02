"""SQLAlchemy 2.0 models for the RESS results store."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on PostgreSQL, generic JSON elsewhere (e.g. SQLite in tests)
JsonType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One RESS invocation: configuration snapshot and counters."""

    __tablename__ = "runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    finished_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    software_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    n_ok: Mapped[int] = mapped_column(Integer, default=0)
    n_failed: Mapped[int] = mapped_column(Integer, default=0)


class SplittingResult(Base):
    """One shear-wave splitting measurement per event-station pair."""

    __tablename__ = "results"
    __table_args__ = (
        UniqueConstraint("event_id", "network", "station", name="uq_results_event_station"),
        Index("ix_results_origin_time", "origin_time"),
        Index("ix_results_network_station", "network", "station"),
        Index("ix_results_quality_class", "quality_class"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # event layer
    event_id: Mapped[str] = mapped_column(String(255))
    origin_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    origin_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    event_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    event_depth_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    magnitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    magnitude_type: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # station layer
    network: Mapped[str] = mapped_column(String(8))
    station: Mapped[str] = mapped_column(String(8))
    location: Mapped[str | None] = mapped_column(String(8), nullable=True)
    channel: Mapped[str | None] = mapped_column(String(8), nullable=True)
    station_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    station_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    station_elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)

    # path layer
    epi_distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    hypo_distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    backazimuth: Mapped[float | None] = mapped_column(Float, nullable=True)
    incidence_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    takeoff_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    s_travel_time_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    # eigenvalue method (Silver & Chan, 1991)
    ev_phi: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_phi_err: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_dt_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_dt_err_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_spol: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_cc_fs: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_cc_ne: Mapped[float | None] = mapped_column(Float, nullable=True)
    ev_linearity: Mapped[float | None] = mapped_column(Float, nullable=True)

    # rotation-correlation method (Bowman & Ando, 1987)
    rc_phi: Mapped[float | None] = mapped_column(Float, nullable=True)
    rc_phi_err: Mapped[float | None] = mapped_column(Float, nullable=True)
    rc_dt_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rc_dt_err_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rc_spol: Mapped[float | None] = mapped_column(Float, nullable=True)
    rc_cc_fs: Mapped[float | None] = mapped_column(Float, nullable=True)
    rc_cc_ne: Mapped[float | None] = mapped_column(Float, nullable=True)

    # quality layer (Wuestefeld et al., 2010)
    qfinal: Mapped[float | None] = mapped_column(Float, nullable=True)
    qmean: Mapped[float | None] = mapped_column(Float, nullable=True)
    qbest: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    omega: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_class: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # processing layer
    n_windows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_cluster_windows: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_clusters: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_in_cluster: Mapped[int | None] = mapped_column(Integer, nullable=True)
    filter_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    filter_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    snr: Mapped[float | None] = mapped_column(Float, nullable=True)
    dominant_period_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    rotation_mode: Mapped[str | None] = mapped_column(String(8), nullable=True)
    consensus_used: Mapped[bool] = mapped_column(Boolean, default=False)
    n_filters_used: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # bookkeeping
    pick_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    run_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("runs.run_id"), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta_json: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    updated_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

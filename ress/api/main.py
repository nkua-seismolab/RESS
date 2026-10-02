"""RESS results API.

Read-only, intended for internal use (no authentication). Results are
returned as CSV by default, or JSON with ?format=json.

Run with:
    uvicorn ress.api.main:app --host 0.0.0.0 --port 8000
The database connection comes from the POSTGRES_* environment (.env).
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from ress import __version__
from ress.db.models import Run, SplittingResult
from ress.db.session import database_url_from_env, init_db, make_engine, make_session_factory

MAX_LIMIT = 100_000

# scalar columns returned by the JSON format (meta_json added separately)
JSON_COLUMNS = [c.name for c in SplittingResult.__table__.columns if c.name != "meta_json"]

# fixed CSV column order of the results export
CSV_COLUMNS = [
    "id",
    # event layer
    "event_id",
    "event_origin_time_utc",
    "event_latitude_deg",
    "event_longitude_deg",
    "event_depth_km",
    "event_magnitude",
    "event_magnitude_type",
    # station layer
    "station_network_code",
    "station_code",
    "station_latitude_deg",
    "station_longitude_deg",
    "station_elevation_m",
    # path layer
    "path_epicentral_km",
    "path_backazimuth_deg",
    "path_incidence_deg",
    "path_takeoff_deg",
    "path_travel_time_P_s",
    "path_travel_time_S_s",
    # ev layer
    "ev_phi_deg",
    "ev_phi_err_deg",
    "ev_dt_ms",
    "ev_dt_err_ms",
    "ev_spol_deg",
    "ev_cc_fs",
    "ev_cc_ne",
    "ev_l_r",
    "ev_t_dominant_s",
    # rc layer
    "rc_phi_deg",
    "rc_phi_err_deg",
    "rc_dt_ms",
    "rc_dt_err_ms",
    "rc_spol_deg",
    "rc_cc_fs",
    "rc_cc_ne",
    "rc_cc_fs_corr",
    "rc_t_dominant_s",
    # qc layer
    "qc_qfinal",
    "qc_qmean",
    "qc_qbest",
    "qc_n_windows",
    "qc_delta",
    "qc_omega",
    "qc_class",
    # processing info
    "linearity_test",
    "filter",
    "filter_snr",
    "filter_bandwidth_oct",
    "filter_score",
    "n_windows",
    # bookkeeping
    "status",
    "comment",
    "run_id",
    "updated_utc",
]

# EV linearity threshold used for the linearity_test text field
LINEARITY_THRESHOLD = 0.15

app = FastAPI(
    title="RESS API",
    version=__version__,
    description="Read-only access to shear-wave splitting results.",
)

_session_factory = None


def _get_session() -> Iterator[Session]:
    global _session_factory
    if _session_factory is None:
        engine = make_engine(database_url_from_env())
        init_db(engine)
        _session_factory = make_session_factory(engine)
    with _session_factory() as session:
        yield session


SessionDep = Annotated[Session, Depends(_get_session)]


def _row_to_dict(row: SplittingResult, include_meta: bool) -> dict:
    data = {name: getattr(row, name) for name in JSON_COLUMNS}
    for key, value in data.items():
        if isinstance(value, datetime):
            data[key] = value.isoformat()
    if include_meta:
        data["meta_json"] = row.meta_json
    return data


def _csv_row(row: SplittingResult) -> dict:
    """Map a results row onto the CSV export columns."""
    if row.filter_low is not None and row.filter_high is not None:
        filter_name = f"{row.filter_low:.1f}-{row.filter_high:.1f} Hz"
        bandwidth = math.log2(row.filter_high / row.filter_low)
        score = row.snr * bandwidth if row.snr is not None else None
    else:
        filter_name = bandwidth = score = None

    linearity_test = None
    if row.ev_linearity is not None:
        linearity_test = "linear" if row.ev_linearity <= LINEARITY_THRESHOLD else "non-linear"

    return {
        "id": row.id,
        "event_id": row.event_id,
        "event_origin_time_utc": row.origin_time.isoformat() if row.origin_time else None,
        "event_latitude_deg": row.event_latitude,
        "event_longitude_deg": row.event_longitude,
        "event_depth_km": row.event_depth_km,
        "event_magnitude": row.magnitude,
        "event_magnitude_type": row.magnitude_type,
        "station_network_code": row.network,
        "station_code": row.station,
        "station_latitude_deg": row.station_latitude,
        "station_longitude_deg": row.station_longitude,
        "station_elevation_m": row.station_elevation_m,
        "path_epicentral_km": row.epi_distance_km,
        "path_backazimuth_deg": row.backazimuth,
        "path_incidence_deg": row.incidence_angle,
        "path_takeoff_deg": row.takeoff_angle,
        "path_travel_time_P_s": None,  # RESS does not compute the P travel time
        "path_travel_time_S_s": row.s_travel_time_s,
        "ev_phi_deg": row.ev_phi,
        "ev_phi_err_deg": row.ev_phi_err,
        "ev_dt_ms": row.ev_dt_ms,
        "ev_dt_err_ms": row.ev_dt_err_ms,
        "ev_spol_deg": row.ev_spol,
        "ev_cc_fs": row.ev_cc_fs,
        "ev_cc_ne": row.ev_cc_ne,
        "ev_l_r": row.ev_linearity,
        "ev_t_dominant_s": row.dominant_period_s,
        "rc_phi_deg": row.rc_phi,
        "rc_phi_err_deg": row.rc_phi_err,
        "rc_dt_ms": row.rc_dt_ms,
        "rc_dt_err_ms": row.rc_dt_err_ms,
        "rc_spol_deg": row.rc_spol,
        "rc_cc_fs": row.rc_cc_fs,
        "rc_cc_ne": row.rc_cc_ne,
        "rc_cc_fs_corr": row.rc_cc_fs,  # RC consensus keeps the corrected FS correlation
        "rc_t_dominant_s": row.dominant_period_s,
        "qc_qfinal": row.qfinal,
        "qc_qmean": row.qmean,
        "qc_qbest": row.qbest,
        "qc_n_windows": row.n_cluster_windows,
        "qc_delta": row.delta,
        "qc_omega": row.omega,
        "qc_class": row.quality_class,
        "linearity_test": linearity_test,
        "filter": filter_name,
        "filter_snr": row.snr,
        "filter_bandwidth_oct": bandwidth,
        "filter_score": score,
        "n_windows": row.n_windows,
        "status": row.status,
        "comment": row.comment,
        "run_id": row.run_id,
        "updated_utc": row.updated_utc.isoformat() if row.updated_utc else None,
    }


def _csv_stream(rows: list[SplittingResult]) -> Iterator[str]:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    yield buffer.getvalue()
    for row in rows:
        buffer.seek(0)
        buffer.truncate()
        writer.writerow(_csv_row(row))
        yield buffer.getvalue()


@app.get("/health")
def health(session: SessionDep) -> dict:
    session.execute(select(1))
    return {"status": "ok", "version": __version__}


@app.get("/performance")
def performance(
    session: SessionDep, limit: int = Query(default=100, ge=1, le=MAX_LIMIT)
) -> list[dict]:
    """Per-run processing statistics (one row per RESS invocation)."""
    stmt = select(Run).order_by(Run.run_id.desc()).limit(limit)
    return [
        {
            "run_id": run.run_id,
            "started_utc": run.started_utc.isoformat() if run.started_utc else None,
            "finished_utc": run.finished_utc.isoformat() if run.finished_utc else None,
            "software_version": run.software_version,
            "n_ok": run.n_ok,
            "n_failed": run.n_failed,
        }
        for run in session.execute(stmt).scalars()
    ]


@app.get("/results")
def results(
    session: SessionDep,
    # event filters
    event_id: str | None = None,
    start_time: datetime | None = Query(default=None, description="Origin time >= (ISO 8601)"),
    end_time: datetime | None = Query(default=None, description="Origin time <= (ISO 8601)"),
    min_event_latitude: float | None = None,
    max_event_latitude: float | None = None,
    min_event_longitude: float | None = None,
    max_event_longitude: float | None = None,
    min_event_depth_km: float | None = None,
    max_event_depth_km: float | None = None,
    min_magnitude: float | None = None,
    max_magnitude: float | None = None,
    # station filters
    network: str | None = None,
    station: str | None = None,
    min_station_latitude: float | None = None,
    max_station_latitude: float | None = None,
    min_station_longitude: float | None = None,
    max_station_longitude: float | None = None,
    # path / quality filters
    min_epi_distance_km: float | None = None,
    max_epi_distance_km: float | None = None,
    min_qfinal: float | None = None,
    max_qfinal: float | None = None,
    quality_class: str | None = Query(
        default=None, description="Good Split, Good Null or Poor/Noise"
    ),
    status: str | None = Query(default="ok", description="ok, failed or 'all'"),
    # output
    limit: int = Query(default=10_000, ge=1, le=MAX_LIMIT),
    format: str = Query(default="csv", pattern="^(csv|json)$"),
):
    """Query splitting results; returns CSV (default) or JSON."""
    stmt = select(SplittingResult)

    if event_id is not None:
        stmt = stmt.where(SplittingResult.event_id == event_id)
    if start_time is not None:
        stmt = stmt.where(SplittingResult.origin_time >= start_time)
    if end_time is not None:
        stmt = stmt.where(SplittingResult.origin_time <= end_time)
    if min_event_latitude is not None:
        stmt = stmt.where(SplittingResult.event_latitude >= min_event_latitude)
    if max_event_latitude is not None:
        stmt = stmt.where(SplittingResult.event_latitude <= max_event_latitude)
    if min_event_longitude is not None:
        stmt = stmt.where(SplittingResult.event_longitude >= min_event_longitude)
    if max_event_longitude is not None:
        stmt = stmt.where(SplittingResult.event_longitude <= max_event_longitude)
    if min_event_depth_km is not None:
        stmt = stmt.where(SplittingResult.event_depth_km >= min_event_depth_km)
    if max_event_depth_km is not None:
        stmt = stmt.where(SplittingResult.event_depth_km <= max_event_depth_km)
    if min_magnitude is not None:
        stmt = stmt.where(SplittingResult.magnitude >= min_magnitude)
    if max_magnitude is not None:
        stmt = stmt.where(SplittingResult.magnitude <= max_magnitude)
    if network is not None:
        stmt = stmt.where(SplittingResult.network == network)
    if station is not None:
        stmt = stmt.where(SplittingResult.station == station)
    if min_station_latitude is not None:
        stmt = stmt.where(SplittingResult.station_latitude >= min_station_latitude)
    if max_station_latitude is not None:
        stmt = stmt.where(SplittingResult.station_latitude <= max_station_latitude)
    if min_station_longitude is not None:
        stmt = stmt.where(SplittingResult.station_longitude >= min_station_longitude)
    if max_station_longitude is not None:
        stmt = stmt.where(SplittingResult.station_longitude <= max_station_longitude)
    if min_epi_distance_km is not None:
        stmt = stmt.where(SplittingResult.epi_distance_km >= min_epi_distance_km)
    if max_epi_distance_km is not None:
        stmt = stmt.where(SplittingResult.epi_distance_km <= max_epi_distance_km)
    if min_qfinal is not None:
        stmt = stmt.where(SplittingResult.qfinal >= min_qfinal)
    if max_qfinal is not None:
        stmt = stmt.where(SplittingResult.qfinal <= max_qfinal)
    if quality_class is not None:
        stmt = stmt.where(SplittingResult.quality_class == quality_class)
    if status is not None and status != "all":
        if status not in ("ok", "failed"):
            raise HTTPException(status_code=422, detail="status must be ok, failed or all")
        stmt = stmt.where(SplittingResult.status == status)

    stmt = stmt.order_by(SplittingResult.origin_time.desc()).limit(limit)
    rows = list(session.execute(stmt).scalars())

    if format == "json":
        return [_row_to_dict(row, include_meta=True) for row in rows]

    return StreamingResponse(
        _csv_stream(rows),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ress_results.csv"},
    )

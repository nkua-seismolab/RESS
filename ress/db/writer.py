"""Best-effort writer of splitting results to PostgreSQL.

The connection is established lazily and every failure is raised to the
caller (ress.app logs and moves on), so a database outage never blocks
the SeisComP publication path.
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from sqlalchemy import select

from ress import __version__
from ress.config import Config
from ress.core.models import SwsResult, SwsTask
from ress.db.models import Run, SplittingResult
from ress.db.session import database_url_from_env, init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)


class DbWriter:
    """Upserts one row per event-station pair into the results table."""

    def __init__(self, config: Config, url: str | None = None):
        # url override is for tests; production reads the POSTGRES_* environment
        self._url = url or database_url_from_env()
        self._config_snapshot = {
            name: asdict(getattr(config, name))
            for name in ("velocity_model", "waveform", "sws", "processing")
        }
        self._config_snapshot["seiscomp"] = {
            "wait_time": config.seiscomp.wait_time,
            "reprocess": config.seiscomp.reprocess,
        }
        self._engine = None
        self._session_factory = None
        self._run_id: int | None = None

    def _ensure_connection(self) -> None:
        if self._engine is None:
            engine = make_engine(self._url)
            init_db(engine)
            self._engine = engine
            self._session_factory = make_session_factory(engine)
            logger.info("Connected to results database")
        if self._run_id is None:
            with self._session_factory() as session:
                run = Run(software_version=__version__, config_json=self._config_snapshot)
                session.add(run)
                session.commit()
                self._run_id = run.run_id
            logger.info("Started run %d", self._run_id)

    def write_result(self, task: SwsTask, result: SwsResult) -> None:
        """Insert or update the row for the task's event-station pair."""
        self._ensure_connection()

        net, sta, loc, cha = task.seed_id.split(".")
        values = {
            "event_id": task.event_id,
            "origin_id": task.origin_id,
            "origin_time": task.origin_time.datetime,
            "event_latitude": task.event_latitude,
            "event_longitude": task.event_longitude,
            "event_depth_km": task.event_depth_km,
            "magnitude": task.magnitude,
            "magnitude_type": task.magnitude_type,
            "network": net,
            "station": sta,
            "location": loc,
            "channel": cha,
            "station_latitude": task.station_latitude,
            "station_longitude": task.station_longitude,
            "station_elevation_m": task.station_elevation_m,
            "epi_distance_km": result.epi_distance_km,
            "hypo_distance_km": result.hypo_distance_km,
            "backazimuth": result.backazimuth,
            "incidence_angle": result.incidence_angle,
            "takeoff_angle": result.takeoff_angle,
            "s_travel_time_s": result.s_travel_time_s,
            "ev_phi": result.ev_phi,
            "ev_phi_err": result.ev_phi_err,
            "ev_dt_ms": result.ev_dt_ms,
            "ev_dt_err_ms": result.ev_dt_err_ms,
            "ev_spol": result.ev_spol,
            "ev_cc_fs": result.ev_cc_fs,
            "ev_cc_ne": result.ev_cc_ne,
            "ev_linearity": result.ev_linearity,
            "rc_phi": result.rc_phi,
            "rc_phi_err": result.rc_phi_err,
            "rc_dt_ms": result.rc_dt_ms,
            "rc_dt_err_ms": result.rc_dt_err_ms,
            "rc_spol": result.rc_spol,
            "rc_cc_fs": result.rc_cc_fs,
            "rc_cc_ne": result.rc_cc_ne,
            "qfinal": result.qfinal,
            "qmean": result.qmean,
            "qbest": result.qbest,
            "delta": result.delta,
            "omega": result.omega,
            "quality_class": result.quality_class,
            "n_windows": result.n_windows,
            "n_cluster_windows": result.n_cluster_windows,
            "n_clusters": result.n_clusters,
            "n_in_cluster": result.n_in_cluster,
            "filter_low": result.filter_low,
            "filter_high": result.filter_high,
            "snr": result.snr,
            "dominant_period_s": result.dominant_period_s,
            "rotation_mode": result.rotation_mode,
            "consensus_used": result.consensus_used,
            "n_filters_used": result.n_filters_used,
            "pick_id": task.pick_id,
            "run_id": self._run_id,
            "status": "ok" if result.ok else "failed",
            "comment": result.error,
            "meta_json": result.meta or None,
        }

        with self._session_factory() as session:
            row = session.execute(
                select(SplittingResult).where(
                    SplittingResult.event_id == task.event_id,
                    SplittingResult.network == net,
                    SplittingResult.station == sta,
                )
            ).scalar_one_or_none()
            if row is None:
                session.add(SplittingResult(**values))
            else:
                for key, value in values.items():
                    setattr(row, key, value)

            run = session.get(Run, self._run_id)
            if run is not None:
                if result.ok:
                    run.n_ok += 1
                else:
                    run.n_failed += 1

            session.commit()
        logger.debug("Persisted result for %s / %s.%s", task.event_id, net, sta)

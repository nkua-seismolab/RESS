"""Tests for ress.db against an SQLite backend."""

import json

from obspy import UTCDateTime
from sqlalchemy import select

from ress.config import Config
from ress.core.models import SwsResult, SwsTask
from ress.db.models import Run, SplittingResult
from ress.db.session import init_db, make_engine, make_session_factory
from ress.db.writer import DbWriter


def make_task(event_id="ev1", station="ATH"):
    return SwsTask(
        pick_id=f"pick-{event_id}-{station}",
        seed_id=f"HL.{station}..HHZ",
        pick_time=UTCDateTime(2026, 1, 1, 0, 0, 10),
        event_id=event_id,
        origin_id="orig1",
        origin_time=UTCDateTime(2026, 1, 1, 0, 0, 0),
        event_latitude=38.0,
        event_longitude=23.7,
        event_depth_km=10.0,
        magnitude=3.5,
        magnitude_type="ML",
        station_latitude=38.1,
        station_longitude=23.8,
        station_elevation_m=100.0,
    )


def make_result(pick_id, qfinal=0.8):
    return SwsResult(
        pick_id=pick_id,
        ev_phi=40.0,
        ev_dt_ms=120.0,
        qfinal=qfinal,
        quality_class="Good Split",
        meta={"filters": [{"filter": "1.0-8.0 Hz"}]},
    )


def make_writer(tmp_path):
    # sqlite override bypasses the POSTGRES_* environment
    return DbWriter(Config(), url=f"sqlite:///{tmp_path}/ress.db")


def test_write_and_read_result(tmp_path):
    writer = make_writer(tmp_path)
    task = make_task()
    writer.write_result(task, make_result(task.pick_id))

    engine = make_engine(f"sqlite:///{tmp_path}/ress.db")
    with make_session_factory(engine)() as session:
        row = session.execute(select(SplittingResult)).scalar_one()
        assert row.event_id == "ev1"
        assert row.network == "HL"
        assert row.station == "ATH"
        assert row.ev_phi == 40.0
        assert row.qfinal == 0.8
        assert row.status == "ok"
        assert row.meta_json["filters"][0]["filter"] == "1.0-8.0 Hz"

        run = session.execute(select(Run)).scalar_one()
        assert run.n_ok == 1
        assert run.n_failed == 0


def test_upsert_same_event_station(tmp_path):
    writer = make_writer(tmp_path)
    task = make_task()
    writer.write_result(task, make_result(task.pick_id, qfinal=0.5))
    writer.write_result(task, make_result(task.pick_id, qfinal=0.9))

    engine = make_engine(f"sqlite:///{tmp_path}/ress.db")
    with make_session_factory(engine)() as session:
        row = session.execute(select(SplittingResult)).scalar_one()
        assert row.qfinal == 0.9


def test_failed_result_persisted_with_status(tmp_path):
    writer = make_writer(tmp_path)
    task = make_task()
    writer.write_result(task, SwsResult(pick_id=task.pick_id, error="No filters passed"))

    engine = make_engine(f"sqlite:///{tmp_path}/ress.db")
    with make_session_factory(engine)() as session:
        row = session.execute(select(SplittingResult)).scalar_one()
        assert row.status == "failed"
        assert row.comment == "No filters passed"

        run = session.execute(select(Run)).scalar_one()
        assert run.n_failed == 1


def test_init_db_idempotent(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/ress.db")
    init_db(engine)
    init_db(engine)


def test_run_snapshot_excludes_credentials(tmp_path, monkeypatch):
    config = Config()
    config.seiscomp.database = "mysql://sentinel_user:sentinel_password@db/seiscomp"
    config.seiscomp.host = "scmp://sentinel_token@broker"
    monkeypatch.setenv("POSTGRES_PASSWORD", "sentinel_postgres_password")
    url = f"sqlite:///{tmp_path}/ress.db"
    writer = DbWriter(config, url=url)
    task = make_task()
    writer.write_result(task, make_result(task.pick_id))

    with make_session_factory(make_engine(url))() as session:
        snapshot = session.execute(select(Run)).scalar_one().config_json
    assert "sentinel" not in json.dumps(snapshot)
    assert snapshot["seiscomp"] == {"wait_time": 300, "reprocess": False}
    assert snapshot["sws"]["max_delay_ms"] == config.sws.max_delay_ms
    assert "logging" not in snapshot
    assert "sds" not in snapshot
    assert config.seiscomp.database.startswith("mysql://sentinel_user:")

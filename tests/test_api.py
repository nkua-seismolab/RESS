"""Tests for the RESS results API against an SQLite backend."""

import csv
import io
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import ress.api.main as api_main
from ress.db.models import SplittingResult
from ress.db.session import init_db, make_engine, make_session_factory


@pytest.fixture
def client(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/api.db")
    init_db(engine)
    factory = make_session_factory(engine)

    with factory() as session:
        session.add_all(
            [
                SplittingResult(
                    event_id="ev1",
                    origin_time=datetime(2026, 1, 1, tzinfo=UTC),
                    event_latitude=38.0,
                    event_longitude=23.7,
                    event_depth_km=10.0,
                    magnitude=3.5,
                    network="HL",
                    station="ATH",
                    station_latitude=38.1,
                    station_longitude=23.8,
                    epi_distance_km=15.0,
                    ev_phi=40.0,
                    ev_dt_ms=120.0,
                    qfinal=0.85,
                    quality_class="Good Split",
                    status="ok",
                ),
                SplittingResult(
                    event_id="ev2",
                    origin_time=datetime(2026, 2, 1, tzinfo=UTC),
                    event_latitude=39.5,
                    event_longitude=22.0,
                    event_depth_km=25.0,
                    magnitude=4.2,
                    network="HP",
                    station="SERG",
                    station_latitude=38.4,
                    station_longitude=22.1,
                    epi_distance_km=60.0,
                    ev_phi=-20.0,
                    ev_dt_ms=80.0,
                    qfinal=-0.9,
                    quality_class="Good Null",
                    status="ok",
                ),
                SplittingResult(
                    event_id="ev3",
                    network="HL",
                    station="ATH",
                    status="failed",
                    comment="No filters passed",
                ),
            ]
        )
        session.commit()

    api_main._session_factory = factory
    yield TestClient(api_main.app)
    api_main._session_factory = None


def parse_csv(response):
    assert response.headers["content-type"].startswith("text/csv")
    return list(csv.DictReader(io.StringIO(response.text)))


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_results_csv_default_excludes_failed(client):
    rows = parse_csv(client.get("/results"))
    assert len(rows) == 2
    assert {row["event_id"] for row in rows} == {"ev1", "ev2"}


def test_results_csv_column_layout(client):
    from ress.api.main import CSV_COLUMNS

    response = client.get("/results", params={"event_id": "ev1"})
    header = response.text.splitlines()[0].split(",")
    assert header == CSV_COLUMNS

    row = parse_csv(response)[0]
    assert row["station_network_code"] == "HL"
    assert row["station_code"] == "ATH"
    assert row["ev_phi_deg"] == "40.0"
    assert row["qc_qfinal"] == "0.85"
    assert row["path_epicentral_km"] == "15.0"


def test_results_status_all(client):
    rows = parse_csv(client.get("/results", params={"status": "all"}))
    assert len(rows) == 3


def test_results_filter_network_station(client):
    rows = parse_csv(client.get("/results", params={"network": "HL", "station": "ATH"}))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "ev1"


def test_results_filter_origin_time(client):
    rows = parse_csv(client.get("/results", params={"start_time": "2026-01-15T00:00:00"}))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "ev2"


def test_results_filter_event_box_and_magnitude(client):
    params = {
        "min_event_latitude": 39.0,
        "max_event_latitude": 40.0,
        "min_magnitude": 4.0,
    }
    rows = parse_csv(client.get("/results", params=params))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "ev2"


def test_results_filter_quality(client):
    rows = parse_csv(client.get("/results", params={"min_qfinal": 0.75}))
    assert len(rows) == 1
    assert rows[0]["qc_class"] == "Good Split"

    rows = parse_csv(client.get("/results", params={"quality_class": "Good Null"}))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "ev2"


def test_results_filter_epi_distance(client):
    rows = parse_csv(client.get("/results", params={"max_epi_distance_km": 20.0}))
    assert len(rows) == 1
    assert rows[0]["event_id"] == "ev1"


def test_results_json_format(client):
    response = client.get("/results", params={"format": "json", "event_id": "ev1"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["ev_phi"] == 40.0


def test_results_invalid_format_rejected(client):
    assert client.get("/results", params={"format": "xml"}).status_code == 422

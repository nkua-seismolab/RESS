"""Conversion between SeisComP Pick objects and the SWS core dataclasses.

Snapshots everything the worker process needs (event, station and channel
metadata) into picklable SwsTask objects on the main thread, and writes
SwsResult objects back onto picks as JSON comments.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from seiscomp import client, datamodel

from ress import __version__
from ress.scclient import pick_io

if TYPE_CHECKING:
    from ress.core.models import SwsResult, SwsTask

logger = logging.getLogger(__name__)

# comment ids look like <pickID>/comment/ress#<UTC time in compact ISO format>
SWS_COMMENT_MARKER = "/comment/ress#"
SWS_COMMENT_TIME_FORMAT = "%Y%m%dT%H%M%S.%fZ"


def is_s_pick(pick, arrival) -> bool:
    """A pick is treated as S if its arrival phase (or phase hint) starts with S."""
    phase = ""
    if arrival is not None:
        try:
            phase = arrival.phase().code()
        except Exception:
            phase = ""
    if not phase:
        try:
            phase = pick.phaseHint().code()
        except Exception:
            phase = ""
    return phase.upper().startswith("S")


def has_sws_comment(pick) -> bool:
    """True if the pick already carries a RESS splitting comment."""
    for i in range(pick.commentCount()):
        try:
            if SWS_COMMENT_MARKER in pick.comment(i).id():
                return True
        except Exception:
            continue
    return False


def station_coords(network: str, station: str, time) -> tuple[float, float, float] | None:
    """(latitude, longitude, elevation_m) from the loaded inventory, or None."""
    try:
        sta = client.Inventory.Instance().getStation(network, station, time)
        return (sta.latitude(), sta.longitude(), sta.elevation())
    except Exception:
        return None


def channel_orientations(
    network: str, station: str, location: str, channel_base: str, time
) -> dict[str, tuple[float, float]]:
    """Map channel code -> (azimuth_deg, dip_deg) for one sensor location.

    Only channels sharing the pick's band+instrument code (e.g. 'HH') and
    whose epoch covers the pick time are returned.
    """
    orientations: dict[str, tuple[float, float]] = {}
    try:
        inv = client.Inventory.Instance().inventory()
    except Exception:
        return orientations
    if inv is None:
        return orientations

    for i in range(inv.networkCount()):
        net = inv.network(i)
        if net.code() != network:
            continue
        for j in range(net.stationCount()):
            sta = net.station(j)
            if sta.code() != station:
                continue
            for k in range(sta.sensorLocationCount()):
                loc = sta.sensorLocation(k)
                if loc.code() != location:
                    continue
                for m in range(loc.streamCount()):
                    stream = loc.stream(m)
                    code = stream.code()
                    if not code.startswith(channel_base):
                        continue
                    try:
                        if stream.start() > time:
                            continue
                    except Exception:
                        continue
                    try:
                        if stream.end() < time:
                            continue
                    except Exception:
                        pass  # open epoch
                    try:
                        orientations[code] = (stream.azimuth(), stream.dip())
                    except Exception:
                        logger.debug("Channel %s has no orientation", code)
    return orientations


def _preferred_magnitude(event) -> tuple[float | None, str | None]:
    try:
        magnitude = datamodel.Magnitude.Find(event.preferredMagnitudeID())
        if magnitude is None:
            return None, None
        return magnitude.magnitude().value(), magnitude.type()
    except Exception:
        return None, None


def build_task(pick, event, origin) -> SwsTask:
    """Build the core-facing task for one S pick (main thread only)."""
    from obspy import UTCDateTime

    from ress.core.models import SwsTask

    wfid = pick.waveformID()
    net, sta, loc, cha = (
        wfid.networkCode(),
        wfid.stationCode(),
        wfid.locationCode(),
        wfid.channelCode(),
    )
    sc_time = pick.time().value()

    coords = station_coords(net, sta, sc_time)
    magnitude, magnitude_type = _preferred_magnitude(event)

    return SwsTask(
        pick_id=pick.publicID(),
        seed_id=pick_io.seed_id(pick),
        pick_time=pick_io.pick_time(pick),
        event_id=event.publicID(),
        origin_id=origin.publicID(),
        origin_time=UTCDateTime(origin.time().value().toString("%Y-%m-%dT%H:%M:%S.%f")),
        event_latitude=origin.latitude().value(),
        event_longitude=origin.longitude().value(),
        event_depth_km=origin.depth().value(),
        magnitude=magnitude,
        magnitude_type=magnitude_type,
        station_latitude=coords[0] if coords else None,
        station_longitude=coords[1] if coords else None,
        station_elevation_m=coords[2] if coords else None,
        channel_orientations=channel_orientations(net, sta, loc, cha[:-1], sc_time),
    )


def apply_result(pick, result: SwsResult) -> bool:
    """Attach the splitting JSON comment to a Pick. Returns True if modified."""
    from obspy import UTCDateTime

    if not result.ok or result.quality_class is None:
        return False

    stored_at = UTCDateTime.utcnow()
    payload = {
        "provenance": {
            "software": "RESS",
            "version": __version__,
            "stored_at": stored_at.isoformat(),
        },
        "splitting": {
            "ev_phi_deg": _round(result.ev_phi, 2),
            "ev_phi_err_deg": _round(result.ev_phi_err, 2),
            "ev_dt_ms": _round(result.ev_dt_ms, 2),
            "ev_dt_err_ms": _round(result.ev_dt_err_ms, 2),
            "ev_spol_deg": _round(result.ev_spol, 2),
            "qfinal": _round(result.qfinal, 4),
            "qmean": _round(result.qmean, 4),
            "qbest": _round(result.qbest, 4),
            "omega": _round(result.omega, 4),
            "delta": _round(result.delta, 4),
            "class": result.quality_class,
        },
    }
    comment = datamodel.Comment()
    comment.setText(json.dumps(payload, allow_nan=True, ensure_ascii=True))
    comment.setId(
        f"{pick.publicID()}{SWS_COMMENT_MARKER}{stored_at.strftime(SWS_COMMENT_TIME_FORMAT)}"
    )
    pick.add(comment)
    return True


def _round(value: float | None, ndigits: int) -> float | None:
    return round(value, ndigits) if value is not None else None

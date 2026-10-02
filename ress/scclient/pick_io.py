"""Pick identifiers, times and arrival lookup for splitting tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from obspy import UTCDateTime


def seed_id(pick) -> str:
    wfid = pick.waveformID()
    return f"{wfid.networkCode()}.{wfid.stationCode()}.{wfid.locationCode()}.{wfid.channelCode()}"


def pick_time(pick) -> UTCDateTime:
    from obspy import UTCDateTime

    return UTCDateTime(pick.time().value().toString("%Y-%m-%dT%H:%M:%S.%f"))


def arrival_index(origin) -> dict:
    """Map pickID -> Arrival for all arrivals of an origin."""
    return {origin.arrival(i).pickID(): origin.arrival(i) for i in range(origin.arrivalCount())}

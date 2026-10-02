"""SDS waveform retrieval, QC and component rotation."""

from __future__ import annotations

import logging

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.signal.rotate import rotate2zne

from ress.config import WaveformConfig

logger = logging.getLogger(__name__)

# channels within this dip of horizontal count as horizontal components
HORIZONTAL_DIP_TOLERANCE = 10.0
# max absolute amplitude at or below this indicates a dead sensor
DEAD_SENSOR_TOLERANCE = 1e-6
# max spread (%) between normalized horizontal peak amplitudes
HORIZONTAL_AMPLITUDE_SPREAD = 30.0


def warm_up_processing() -> None:
    """Trigger ObsPy's lazy entry-point imports once, single-threaded.

    detrend/filter/taper/decimate import their backends on first use;
    concurrent first calls from worker threads deadlock on the import lock.
    """
    tr = Trace(data=np.random.default_rng(0).normal(size=256))
    tr.stats.sampling_rate = 100.0
    st = Stream([tr])
    st.detrend("linear").detrend("demean")
    st.taper(0.001, type="cosine")
    st.filter("bandpass", freqmin=1.0, freqmax=10.0, zerophase=True)
    st.decimate(factor=2, strict_length=False, no_filter=True)
    logger.debug("ObsPy processing backends warmed up")


class WaveformError(Exception):
    """Raised when waveform data are missing or fail quality control."""


class WaveformFetcher:
    """Retrieves all components of a pick's stream from an SDS archive."""

    def __init__(self, archive: str, config: WaveformConfig):
        self.client = SDSClient(archive)
        self.config = config
        logger.info("Initialized SDS client with archive: %s", archive)

    def fetch(self, seed_id: str, pick_time: UTCDateTime) -> Stream:
        net, sta, loc, cha = seed_id.split(".")
        st = self.client.get_waveforms(
            network=net,
            station=sta,
            location=loc,
            channel=cha[:-1] + "?",
            starttime=pick_time - self.config.t_before,
            endtime=pick_time + self.config.t_after,
        )
        if not st:
            raise WaveformError(f"No waveform data for {seed_id} at {pick_time}")
        return st


def horizontal_channels(orientations: dict[str, tuple[float, float]]) -> list[str]:
    """Channel codes that are horizontal per their inventory dip."""
    return [
        code
        for code, (_azimuth, dip) in orientations.items()
        if abs(dip) <= HORIZONTAL_DIP_TOLERANCE and not code.endswith("Z")
    ]


def quality_control(st: Stream, min_gap: float, orientations: dict) -> Stream:
    """Run the pre-analysis QC pipeline; raises WaveformError on rejection.

    Checks: 3 components present, no gaps, live sensors, comparable
    horizontal amplitudes. Detrends and merges the stream in place.
    """
    channels = {tr.stats.channel for tr in st}
    if len(channels) < 3:
        raise WaveformError("Less than 3 channels found in waveform data")

    gaps = st.get_gaps(min_gap=min_gap)
    if len(gaps) > 0:
        raise WaveformError("Gaps found in waveform data")

    st.detrend("linear").detrend("demean")
    st.merge(method=1, fill_value=0)

    # dead sensor check
    max_abs_amplitude = np.asarray([np.max(np.abs(tr.data)) for tr in st])
    if np.any(max_abs_amplitude <= DEAD_SENSOR_TOLERANCE):
        raise WaveformError("Dead sensor detected in waveform data")

    # horizontal components must have comparable maximum amplitudes
    channels_horiz = horizontal_channels(orientations)
    max_abs_horiz = np.asarray(
        [np.max(np.abs(tr.data)) for tr in st if tr.stats.channel in channels_horiz]
    )
    if len(max_abs_horiz) >= 2:
        max_abs_horiz_norm = 100 * max_abs_horiz / max(max_abs_horiz)
        if np.ptp(max_abs_horiz_norm) > HORIZONTAL_AMPLITUDE_SPREAD:
            raise WaveformError("Horizontal components have non-comparable amplitudes")

    return st


def rotate_to_zne(st: Stream, orientations: dict[str, tuple[float, float]]) -> Stream:
    """Rotate a stream to ZNE using the channel orientations, if necessary.

    Handles 1-2-3 and Z-1-2 style component combinations via the inventory
    azimuth/dip of each channel.
    """
    comps = "".join(sorted({tr.stats.channel[-1] for tr in st}))
    if comps == "ENZ":
        return st

    traces = sorted(st, key=lambda tr: tr.stats.channel)[:3]
    if len(traces) < 3:
        raise WaveformError("Need 3 components to rotate to ZNE")

    # trim to a common number of samples
    n_min = min(len(tr.data) for tr in traces)

    args = []
    for tr in traces:
        orientation = orientations.get(tr.stats.channel)
        if orientation is None:
            raise WaveformError(f"No inventory orientation for channel {tr.stats.channel}")
        azimuth, dip = orientation
        args.extend([tr.data[:n_min], azimuth, dip])

    try:
        z, n, e = rotate2zne(*args)
    except Exception as exc:
        raise WaveformError(f"Failed to rotate waveform data: {exc}") from exc

    base = traces[0].stats.channel[:-1]
    rotated = Stream()
    for data, comp in ((z, "Z"), (n, "N"), (e, "E")):
        tr = traces[0].copy()
        tr.data = data
        tr.stats.channel = base + comp
        rotated += tr

    comps_after = "".join(sorted({tr.stats.channel[-1] for tr in rotated}))
    if comps_after != "ENZ":
        raise WaveformError("Failed to obtain ENZ components")
    return rotated


def ensure_sampling_rate(st: Stream, target: float, decimate: bool) -> Stream:
    """Bring a stream to the target sampling rate, or raise WaveformError."""
    sps = st[0].stats.sampling_rate
    if sps == target:
        return st
    if sps < target:
        raise WaveformError(f"Sampling rate {sps} Hz is below the target {target} Hz")
    if not decimate:
        raise WaveformError(
            f"Sampling rate {sps} Hz exceeds the target {target} Hz and decimation is disabled"
        )
    if sps % target:
        raise WaveformError(f"Sampling rate {sps} Hz is not an integer multiple of {target} Hz")
    st.decimate(factor=int(sps // target), strict_length=False, no_filter=True)
    return st

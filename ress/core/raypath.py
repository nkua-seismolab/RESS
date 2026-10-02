"""Theoretical ray-path computation and shear-wave window screening."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model

logger = logging.getLogger(__name__)


class RaypathError(Exception):
    """Raised when a pick fails the ray-path screening."""


class RaypathCalculator:
    """TauP travel-time wrapper enforcing the shear-wave window criterion."""

    def __init__(self, model_path: str, max_ain: float, min_tkf: float):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Velocity model file not found: {path}")
        npz = path.with_suffix(".npz")
        if not npz.exists():
            logger.info("Compiling velocity model %s for TauP...", path)
            try:
                build_taup_model(str(path), str(path.parent), verbose=False)
            except OSError:
                with tempfile.TemporaryDirectory(prefix="ress-taup-") as directory:
                    build_taup_model(str(path), directory, verbose=False)
                    self.model = TauPyModel(str(Path(directory) / npz.name))
            else:
                self.model = TauPyModel(str(npz))
        else:
            self.model = TauPyModel(str(npz))
        self.max_ain = max_ain
        self.min_tkf = min_tkf

    def compute(
        self,
        event_lat: float,
        event_lon: float,
        event_depth_km: float,
        station_lat: float,
        station_lon: float,
    ) -> dict:
        """Theoretical S geometry for one event-station pair.

        :returns: dict with ain, tkf, s_travel_time, backazimuth,
            epi_distance_km and hypo_distance_km
        :raises RaypathError: if no S arrival exists or the shear-wave
            window criterion is violated
        """
        arrivals = self.model.get_travel_times_geo(
            event_depth_km, event_lat, event_lon, station_lat, station_lon, phase_list=["s", "S"]
        )
        if len(arrivals) == 0:
            raise RaypathError("No theoretical S arrivals found")

        # earliest arrival: local-scale context
        arrival = min(arrivals, key=lambda a: a.time)

        if arrival.name not in ("s", "S"):
            raise RaypathError("Theoretical arrival is not an S phase")

        ain = arrival.incident_angle
        tkf = arrival.takeoff_angle

        # shear-wave window criterion
        if ain > self.max_ain:
            raise RaypathError(f"Incident angle too wide ({ain:.1f} > {self.max_ain:.1f})")
        if tkf < self.min_tkf:
            raise RaypathError(f"Takeoff angle too small ({tkf:.1f} < {self.min_tkf:.1f})")

        dist_m, _az, baz = gps2dist_azimuth(event_lat, event_lon, station_lat, station_lon)
        epi_km = dist_m * 1e-3
        hypo_km = float(np.sqrt(epi_km**2 + event_depth_km**2))

        return {
            "ain": float(ain),
            "tkf": float(tkf),
            "s_travel_time": float(arrival.time),
            "backazimuth": float(baz),
            "epi_distance_km": float(epi_km),
            "hypo_distance_km": hypo_km,
        }

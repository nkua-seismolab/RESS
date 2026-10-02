"""Per-pick shear-wave splitting pipeline (SeisComP-free).

Processing order: ray-path screening,
waveform QC, adaptive filter selection, per-window EV/RC measurements,
cluster analysis, multi-filter consensus and event-level quality.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ress.config import Config
from ress.core.clustering import find_optimal_result
from ress.core.models import SwsResult, SwsTask
from ress.core.quality import calculate_consensus, calculate_event_quality
from ress.core.raypath import RaypathCalculator, RaypathError
from ress.core.splitting import process_window
from ress.core.waveforms import (
    WaveformError,
    WaveformFetcher,
    ensure_sampling_rate,
    quality_control,
    rotate_to_zne,
    warm_up_processing,
)
from ress.core.windows import adaptive_filter, make_windows

logger = logging.getLogger(__name__)

# hard cap on window-analysis processes
MAX_WINDOW_WORKERS = 4


class ProcessingError(Exception):
    """Raised when a pick cannot produce a splitting measurement."""


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


class SwsProcessor:
    """Runs the complete splitting analysis for SwsTask objects."""

    def __init__(self, config: Config):
        self.config = config
        self.fetcher = WaveformFetcher(config.sds.archive, config.waveform)
        self.raypath = RaypathCalculator(
            config.velocity_model.path, config.sws.max_ain, config.sws.min_tkf
        )
        warm_up_processing()
        n_cores = config.processing.n_cores
        max_cores = mp.cpu_count() if n_cores == -1 else n_cores
        self.n_workers = min(MAX_WINDOW_WORKERS, max_cores)

    # ------------------------------------------------------------------
    # batch entry point
    # ------------------------------------------------------------------
    def process_batch(self, tasks: list[SwsTask]) -> dict[str, SwsResult]:
        results: dict[str, SwsResult] = {}
        for task in tasks:
            try:
                results[task.pick_id] = self.process(task)
            except (ProcessingError, RaypathError, WaveformError) as exc:
                logger.info("Pick %s rejected: %s", task.pick_id, exc)
                results[task.pick_id] = SwsResult(pick_id=task.pick_id, error=str(exc))
            except Exception as exc:
                logger.exception("Pick %s failed", task.pick_id)
                results[task.pick_id] = SwsResult(pick_id=task.pick_id, error=str(exc))
        return results

    # ------------------------------------------------------------------
    # single-pick pipeline
    # ------------------------------------------------------------------
    def process(self, task: SwsTask) -> SwsResult:
        sws = self.config.sws
        net, sta, _loc, _cha = task.seed_id.split(".")

        if sta in sws.blacklist_stations:
            raise ProcessingError(f"Station {sta} is blacklisted")
        if task.station_latitude is None or task.station_longitude is None:
            raise ProcessingError("Station coordinates not found in inventory")

        # ray-path screening (shear-wave window criterion)
        path = self.raypath.compute(
            task.event_latitude,
            task.event_longitude,
            task.event_depth_km,
            task.station_latitude,
            task.station_longitude,
        )

        # waveform retrieval and QC
        st = self.fetcher.fetch(task.seed_id, task.pick_time)
        st = quality_control(st, self.config.waveform.min_gap, task.channel_orientations)
        st = rotate_to_zne(st, task.channel_orientations)
        st = ensure_sampling_rate(
            st, self.config.waveform.target_sampling_rate, self.config.waveform.decimate
        )

        # adaptive filter selection
        all_filters = adaptive_filter(st, task.pick_time, sws.filter_bank, sws.snr_min)
        passed_filters = [f for f in all_filters if f["passes_snr"]]
        if not passed_filters:
            raise ProcessingError("No filters passed the SNR test")

        if sws.consensus.enabled:
            filters_for_analysis = passed_filters[: sws.consensus.n_best_filters]
        else:
            filters_for_analysis = passed_filters[:1]

        # analysis windows
        max_delay_s = sws.max_delay_ms / 1000.0
        win = sws.windows
        pick_time_rel = task.pick_time - st[0].stats.starttime
        windows = make_windows(
            pick_time_rel,
            max_delay=max_delay_s,
            tbeg0=win.tbeg0,
            tbeg1=win.tbeg1,
            tend0=win.tend0,
            tend1=win.tend1,
            dtbeg=win.dtbeg,
            dtend=win.dtend,
            nbeg=win.nbeg,
            nend=win.nend,
        )
        if len(windows) == 0:
            raise ProcessingError("No analysis windows generated")

        ain_to_use = path["ain"] if sws.rotation_mode == "LQT" else None

        # per-filter window analysis + clustering
        ev_filter_results: list[dict] = []
        rc_filter_results: list[dict] = []
        meta_filters: list[dict] = []
        best_cluster_ev = None
        best_aligned: tuple[list, list, list] | None = None

        for filt in filters_for_analysis:
            filter_name = f"{filt['freqmin']:.1f}-{filt['freqmax']:.1f} Hz"
            window_results = self._analyze_windows(
                filt["stream"], path["backazimuth"], windows, ain_to_use, max_delay_s
            )

            results_ev = [r["ev"] for r in window_results if r["ev"] is not None]
            results_rc = [r["rc"] for r in window_results if r["rc"] is not None]

            # aligned arrays: windows where both EV and QWin exist
            aligned_ev, aligned_rc, aligned_qwin = [], [], []
            for r in window_results:
                if r["ev"] is not None and r["qwin"] is not None:
                    aligned_ev.append(r["ev"])
                    aligned_rc.append(r["rc"])
                    aligned_qwin.append(r["qwin"])

            logger.debug(
                "Filter %s: EV %d/%d, RC %d/%d windows successful",
                filter_name,
                len(results_ev),
                len(windows),
                len(results_rc),
                len(windows),
            )

            filter_meta = {
                "filter": filter_name,
                "snr": _float_or_none(filt["snr"]),
                "score": _float_or_none(filt["score"]),
                "n_windows_ev": len(results_ev),
                "n_windows_rc": len(results_rc),
                "clustering": None,
            }

            if len(results_ev) > 0:
                cluster_ev = find_optimal_result(
                    results_ev,
                    linkage=sws.cluster.linkage,
                    k_max=sws.cluster.k_max,
                    n_min=sws.cluster.n_min,
                    ccrit=sws.cluster.ccrit,
                )
                if cluster_ev["success"]:
                    optimal_window = cluster_ev["result"]["window"]
                    ev_filter_results.append(
                        {
                            "phi": cluster_ev["phi"],
                            "dt": cluster_ev["dt"],
                            "phi_err": cluster_ev["phi_err"],
                            "dt_err": cluster_ev["dt_err"],
                            "spol": cluster_ev["result"]["spol"],
                            "CC_FS": cluster_ev["result"]["CC_FS"],
                            "CC_NE": cluster_ev["result"]["CC_NE"],
                            "L_R": cluster_ev["result"]["L_R"],
                            "T_dominant": cluster_ev["result"].get("T_dominant", np.nan),
                            "n_clusters": cluster_ev["n_clusters"],
                            "n_in_cluster": cluster_ev["n_in_cluster"],
                            "filter": filter_name,
                            "cluster_data": cluster_ev,
                        }
                    )
                    # quality assessment uses the best (first successful) filter
                    if best_cluster_ev is None:
                        best_cluster_ev = cluster_ev
                        best_aligned = (aligned_ev, aligned_rc, aligned_qwin)

                    filter_meta["clustering"] = {
                        "phi": _float_or_none(cluster_ev["phi"]),
                        "dt_ms": _float_or_none(cluster_ev["dt"] * 1000),
                        "phi_err": _float_or_none(cluster_ev["phi_err"]),
                        "dt_err_ms": _float_or_none(cluster_ev["dt_err"] * 1000),
                        "n_clusters": int(cluster_ev["n_clusters"]),
                        "n_in_cluster": int(cluster_ev["n_in_cluster"]),
                        "n_valid_measurements": int(cluster_ev["n_valid_measurements"]),
                        "window": [float(optimal_window[0]), float(optimal_window[1])],
                    }

                    # RC result from the EV-selected window (RC is supportive QC)
                    rc_from_ev_window = next(
                        (
                            rc_res
                            for rc_res in results_rc
                            if rc_res["window"][0] == optimal_window[0]
                            and rc_res["window"][1] == optimal_window[1]
                        ),
                        None,
                    )
                    if rc_from_ev_window is not None:
                        rc_filter_results.append(
                            {
                                "phi": rc_from_ev_window["phi"],
                                "dt": rc_from_ev_window["dt"],
                                "phi_err": rc_from_ev_window["phi_err"],
                                "dt_err": rc_from_ev_window["dt_err"],
                                "spol": rc_from_ev_window["spol"],
                                "CC_FS": rc_from_ev_window["CC_FS"],
                                "CC_NE": rc_from_ev_window["CC_NE"],
                                "T_dominant": rc_from_ev_window.get("T_dominant", np.nan),
                                "window": optimal_window,
                                "filter": filter_name,
                                "n_clusters": 1,  # RC does not cluster
                                "n_in_cluster": 1,
                                "cluster_data": None,
                            }
                        )
                else:
                    logger.debug(
                        "Filter %s: EV clustering failed: %s", filter_name, cluster_ev["reason"]
                    )

            meta_filters.append(filter_meta)

        # multi-filter consensus (single-filter mode averages one result)
        ev_result = calculate_consensus(
            ev_filter_results,
            "EV",
            phi_tolerance=sws.consensus.phi_tolerance,
            dt_tolerance=sws.consensus.dt_tolerance,
        )
        rc_result = calculate_consensus(
            rc_filter_results,
            "RC",
            phi_tolerance=sws.consensus.phi_tolerance,
            dt_tolerance=sws.consensus.dt_tolerance,
        )

        if ev_result is None and rc_result is None:
            raise ProcessingError("No valid EV or RC results after multi-filter consensus")

        # event-level quality (Wuestefeld et al., 2010)
        quality = None
        if ev_result and rc_result and best_cluster_ev and best_aligned:
            aligned_ev, aligned_rc, aligned_qwin = best_aligned
            quality = calculate_event_quality(
                qwin_values=aligned_qwin,
                window_results_ev=aligned_ev,
                window_results_rc=aligned_rc,
                cluster_ev=best_cluster_ev,
                omega1=sws.quality.omega1,
                omega2=sws.quality.omega2,
                gamma=sws.quality.gamma,
            )

        best_filter = filters_for_analysis[0]
        result = SwsResult(
            pick_id=task.pick_id,
            epi_distance_km=path["epi_distance_km"],
            hypo_distance_km=path["hypo_distance_km"],
            backazimuth=path["backazimuth"],
            incidence_angle=path["ain"],
            takeoff_angle=path["tkf"],
            s_travel_time_s=path["s_travel_time"],
            n_windows=len(windows),
            filter_low=float(best_filter["freqmin"]),
            filter_high=float(best_filter["freqmax"]),
            snr=_float_or_none(best_filter["snr"]),
            rotation_mode=sws.rotation_mode,
            consensus_used=sws.consensus.enabled,
            meta={
                "filters": meta_filters,
                "n_filters_passed_snr": len(passed_filters),
                "n_windows_generated": len(windows),
            },
        )

        if ev_result:
            result.ev_phi = _float_or_none(ev_result["phi"])
            result.ev_phi_err = _float_or_none(ev_result["phi_err"])
            result.ev_dt_ms = _float_or_none(ev_result["dt"] * 1000)
            result.ev_dt_err_ms = _float_or_none(ev_result["dt_err"] * 1000)
            result.ev_spol = _float_or_none(ev_result["spol"])
            result.ev_cc_fs = _float_or_none(ev_result["CC_FS"])
            result.ev_cc_ne = _float_or_none(ev_result["CC_NE"])
            result.ev_linearity = _float_or_none(ev_result.get("L_R"))
            result.n_clusters = int(ev_result["n_clusters"])
            result.n_in_cluster = int(ev_result["n_in_cluster"])
            result.n_filters_used = int(ev_result["n_filters_used"])
            result.dominant_period_s = _float_or_none(ev_result.get("T_dominant"))

        if rc_result:
            result.rc_phi = _float_or_none(rc_result["phi"])
            result.rc_phi_err = _float_or_none(rc_result["phi_err"])
            result.rc_dt_ms = _float_or_none(rc_result["dt"] * 1000)
            result.rc_dt_err_ms = _float_or_none(rc_result["dt_err"] * 1000)
            result.rc_spol = _float_or_none(rc_result["spol"])
            result.rc_cc_fs = _float_or_none(rc_result["CC_FS"])
            result.rc_cc_ne = _float_or_none(rc_result["CC_NE"])

        if quality:
            result.qfinal = _float_or_none(quality["Qfinal"])
            result.qmean = _float_or_none(quality["Qmean"])
            result.qbest = _float_or_none(quality["Qbest"])
            result.delta = _float_or_none(quality["Delta"])
            result.omega = _float_or_none(quality["Omega"])
            result.quality_class = quality["Class"]
            result.n_cluster_windows = int(quality["N_cluster_windows"])

        logger.info(
            "Pick %s (%s.%s): EV phi=%s dt=%sms Qfinal=%s class=%s",
            task.pick_id,
            net,
            sta,
            f"{result.ev_phi:.1f}" if result.ev_phi is not None else "N/A",
            f"{result.ev_dt_ms:.1f}" if result.ev_dt_ms is not None else "N/A",
            f"{result.qfinal:+.2f}" if result.qfinal is not None else "N/A",
            result.quality_class or "N/A",
        )
        return result

    # ------------------------------------------------------------------
    # window-level parallelism
    # ------------------------------------------------------------------
    def _analyze_windows(self, st_filtered, bazi, windows, ain, max_delay_s) -> list[dict]:
        """Run both splitting methods on every window in a process pool."""
        gamma = self.config.sws.quality.gamma
        window_args = [
            (st_filtered, bazi, tuple(pickwin), ain, max_delay_s, gamma) for pickwin in windows
        ]
        n_workers = min(self.n_workers, len(window_args))

        t_start = time.time()
        window_results = []
        if n_workers <= 1:
            window_results = [process_window(args) for args in window_args]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(process_window, args) for args in window_args]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        window_results.append(result)
        logger.debug(
            "Analyzed %d windows in %.2fs with %d workers",
            len(window_args),
            time.time() - t_start,
            n_workers,
        )
        return window_results

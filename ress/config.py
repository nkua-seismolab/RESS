"""Configuration loading and validation for RESS (config.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from pathlib import Path
from types import UnionType
from typing import get_args, get_origin, get_type_hints

import yaml


class ConfigError(Exception):
    """Raised when config.yaml is missing, malformed or fails validation."""


@dataclass
class SeisCompConfig:
    """Options consumed by the scclient module."""

    host: str | None = None
    database: str | None = None
    wait_time: int = 300
    reprocess: bool = False


@dataclass
class SDSConfig:
    archive: str = "/data/archive"


@dataclass
class VelocityModelConfig:
    # TauP-compatible layered model (.nd or .tvel); compiled to .npz on first use
    path: str = "velocity_models/rigo_crustal.nd"


@dataclass
class WaveformConfig:
    t_before: float = 10.0
    t_after: float = 60.0
    target_sampling_rate: float = 100.0
    decimate: bool = True
    # minimum duration of gaps that disqualify a stream (s)
    min_gap: float = 1.0


# Filter bank from Savage et al. (2010) Table 1
DEFAULT_FILTER_BANK = [
    [0.4, 4.0],
    [0.5, 5.0],
    [0.2, 7.5],
    [0.3, 3.0],
    [0.5, 4.0],
    [0.6, 3.0],
    [0.8, 6.0],
    [1.0, 3.0],
    [0.5, 2.5],
    [1.0, 8.0],
    [2.0, 3.0],
    [2.0, 6.0],
    [3.0, 8.0],
    [4.0, 10.0],
    [1.0, 10.0],
    [1.0, 20.0],
    [1.0, 8.0],
    [0.5, 10.0],
    [0.5, 20.0],
    [0.5, 8.0],
]


@dataclass
class WindowConfig:
    """Analysis window generation around the S pick (Teanby et al., 2004)."""

    tbeg0: float = -0.50  # earliest window start relative to S (s)
    tbeg1: float = -0.10  # latest window start (s)
    tend0: float = 0.10  # earliest window end (s)
    tend1: float = 0.40  # latest window end (s)
    dtbeg: float = 0.05  # step between window starts (s)
    dtend: float = 0.01  # step between window ends (s)
    nbeg: int = 10  # number of start points
    nend: int = 20  # number of end points


@dataclass
class ClusterConfig:
    """Cluster analysis of window measurements (Teanby et al., 2004)."""

    linkage: str = "ward"
    k_max: int = 20
    n_min: int = 10
    ccrit: float = 3.2  # Duda-Hart critical value


@dataclass
class ConsensusConfig:
    """Multi-filter consensus tolerances."""

    enabled: bool = False
    n_best_filters: int = 1
    phi_tolerance: float = 15.0  # degrees
    dt_tolerance: float = 0.020  # seconds


@dataclass
class QualityConfig:
    """Event-level quality weighting (Wuestefeld et al., 2010)."""

    gamma: float = 0.75  # |Q| threshold for Good Split / Good Null
    omega1: float = 1 / 3  # weight for the mean window quality (Qmean)
    omega2: float = 2 / 3  # weight for the best measurement quality (Qbest)


@dataclass
class SwsConfig:
    """Shear-wave splitting analysis parameters."""

    blacklist_stations: list[str] = field(default_factory=list)
    max_ain: float = 60.0  # shear-wave window: maximum incidence angle (deg)
    min_tkf: float = 90.0  # minimum takeoff angle; >90 ensures upgoing rays (deg)
    max_delay_ms: float = 400.0  # maximum time-delay of the grid search
    snr_min: float = 3.0  # minimum SNR for filter acceptance
    rotation_mode: str = "ZRT"  # 'ZRT' (backazimuth only) or 'LQT' (with incidence)
    filter_bank: list[list[float]] = field(
        default_factory=lambda: [list(f) for f in DEFAULT_FILTER_BANK]
    )
    windows: WindowConfig = field(default_factory=WindowConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)


@dataclass
class LoggingConfig:
    level: str = "DEBUG"
    file: str | None = None


@dataclass
class ProcessingConfig:
    # worker processes for the per-window grid searches (-1 = all cores, capped at 4)
    n_cores: int = -1


@dataclass
class Config:
    seiscomp: SeisCompConfig = field(default_factory=SeisCompConfig)
    sds: SDSConfig = field(default_factory=SDSConfig)
    velocity_model: VelocityModelConfig = field(default_factory=VelocityModelConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    sws: SwsConfig = field(default_factory=SwsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


def _matches_type(value, annotation) -> bool:
    if get_origin(annotation) is UnionType:
        return any(_matches_type(value, option) for option in get_args(annotation))
    if get_origin(annotation) is list:
        return isinstance(value, list) and all(
            _matches_type(item, get_args(annotation)[0]) for item in value
        )
    if annotation is float:
        return type(value) in (int, float) and isfinite(value)
    return type(value) is annotation


def _build_section(cls, name: str, data: dict):
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"Unknown keys in '{name}' section: {sorted(unknown)}")

    hints = get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        f = known[key]
        # nested dataclass sections (e.g. sws.windows) are mappings in the YAML
        if is_dataclass(f.default_factory):
            if not isinstance(value, dict):
                raise ConfigError(f"'{name}.{key}' must be a mapping")
            kwargs[key] = _build_section(f.default_factory, f"{name}.{key}", value)
        else:
            if not _matches_type(value, hints[key]):
                raise ConfigError(f"{name}.{key} has an invalid type or non-finite value")
            kwargs[key] = value
    return cls(**kwargs)


def _validate(config: Config) -> None:
    if config.logging.level.upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ConfigError("logging.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
    if config.seiscomp.wait_time < 0:
        raise ConfigError("seiscomp.wait_time must be >= 0")
    if config.waveform.target_sampling_rate <= 0:
        raise ConfigError("waveform.target_sampling_rate must be > 0")
    if config.waveform.min_gap <= 0:
        raise ConfigError("waveform.min_gap must be > 0")
    if config.sws.rotation_mode not in ("ZRT", "LQT"):
        raise ConfigError("sws.rotation_mode must be 'ZRT' or 'LQT'")
    if config.sws.max_delay_ms <= 0:
        raise ConfigError("sws.max_delay_ms must be > 0")
    if config.sws.snr_min < 0:
        raise ConfigError("sws.snr_min must be >= 0")
    for pair in config.sws.filter_bank:
        if len(pair) != 2 or not pair[0] < pair[1]:
            raise ConfigError(f"sws.filter_bank entries must be [freqmin, freqmax], got {pair}")
    win = config.sws.windows
    if min(win.dtbeg, win.dtend, win.nbeg, win.nend) <= 0:
        raise ConfigError("sws.windows steps and counts must be > 0")
    if not win.tbeg0 <= win.tbeg1 <= 0:
        raise ConfigError("sws.windows requires tbeg0 <= tbeg1 <= 0")
    if not 0 <= win.tend0 <= win.tend1:
        raise ConfigError("sws.windows requires 0 <= tend0 <= tend1")
    cluster = config.sws.cluster
    if cluster.linkage not in ("single", "complete", "average", "ward"):
        raise ConfigError("sws.cluster.linkage must be single/complete/average/ward")
    if cluster.k_max < 2 or cluster.n_min < 1:
        raise ConfigError("sws.cluster requires k_max >= 2 and n_min >= 1")
    consensus = config.sws.consensus
    if consensus.n_best_filters < 1:
        raise ConfigError("sws.consensus.n_best_filters must be >= 1")
    quality = config.sws.quality
    if not 0 < quality.gamma <= 1:
        raise ConfigError("sws.quality.gamma must be in (0, 1]")
    if abs(quality.omega1 + quality.omega2 - 1.0) > 1e-9:
        raise ConfigError("sws.quality.omega1 + omega2 must equal 1")
    if config.processing.n_cores == 0 or config.processing.n_cores < -1:
        raise ConfigError("processing.n_cores must be -1 or a positive integer")


def load_config(path: str | Path) -> Config:
    """Load and validate a config.yaml file. Missing sections/keys use defaults."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path) as fid:
            raw = yaml.safe_load(fid)
    except yaml.YAMLError:
        raise ConfigError(f"Invalid YAML in {path}") from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top level of {path} must be a mapping")

    sections = {f.name for f in fields(Config)}
    unknown = set(raw) - sections
    if unknown:
        raise ConfigError(f"Unknown top-level sections: {sorted(unknown)}")

    kwargs = {}
    for f in fields(Config):
        data = raw.get(f.name, {})
        if not isinstance(data, dict):
            raise ConfigError(f"Section '{f.name}' must be a mapping")
        kwargs[f.name] = _build_section(f.default_factory, f.name, data)

    config = Config(**kwargs)
    _validate(config)
    return config

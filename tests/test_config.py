"""Tests for ress.config."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ress import __main__ as cli
from ress.config import ConfigError, load_config


def write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "text",
    [
        "false",
        "[]",
        "seiscomp: false",
        "seiscomp: []",
        "seiscomp: null",
        "seiscomp: {wait_time: true}",
        "seiscomp: {wait_time: -1}",
        "seiscomp: {reprocess: 'false'}",
        "logging: {level: nonsense}",
        "waveform: {target_sampling_rate: .nan}",
        "logging: [",
    ],
)
def test_invalid_config_rejected(tmp_path, text):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, text))


def test_release_example():
    config = load_config(Path(__file__).resolve().parents[1] / "config.example.yaml")
    assert config.logging.level == "DEBUG"


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_informational_cli_needs_no_config(monkeypatch, option):
    monkeypatch.setattr(cli.sys, "argv", ["ress", option, "--config", "/missing.yaml"])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0


@pytest.mark.parametrize("args", [["--wait-time", "-1"], ["-w-1"], ["--config"]])
def test_invalid_cli_rejected(args):
    with pytest.raises(SystemExit) as result:
        cli._config_path_from_argv(["ress", *args])
    assert result.value.code == 2


def test_cli_precedence(tmp_path):
    config = load_config(write(tmp_path, "seiscomp: {host: yaml-host, database: yaml-db}"))
    for arguments in (
        ["-H", "cli-host", "-d", "cli-db"],
        ["-Hcli-host", "-dcli-db"],
        ["--host=cli-host", "--database=cli-db"],
    ):
        argv = ["ress", *arguments]
        assert cli._inject_connection_args(argv, config) == argv


def test_worker_failure_stops_before_listener(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(cli.sys, "argv", ["ress", "--config", str(write(tmp_path, ""))])
    monkeypatch.setattr(cli.faulthandler, "register", Mock())
    executor = Mock()
    executor.submit.return_value.result.side_effect = RuntimeError("failed")
    monkeypatch.setattr(cli, "ProcessPoolExecutor", Mock(return_value=executor))
    listener = Mock()
    monkeypatch.setitem(
        cli.sys.modules, "ress.app", SimpleNamespace(make_dispatch=Mock(), make_finalize=Mock())
    )
    monkeypatch.setitem(
        cli.sys.modules, "ress.worker", SimpleNamespace(init_worker=Mock(), run_batch=Mock())
    )
    monkeypatch.setitem(
        cli.sys.modules, "ress.scclient.listener", SimpleNamespace(EventListenerApp=listener)
    )
    assert cli.main() == 1
    listener.assert_not_called()
    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
    assert "Worker initialization failed" in caplog.text
    assert "Worker process ready" not in caplog.text


def test_defaults_from_empty_file(tmp_path):
    config = load_config(write(tmp_path, ""))
    assert config.seiscomp.wait_time == 300
    assert config.logging.level == "DEBUG"
    assert config.sws.max_ain == 60.0
    assert config.sws.rotation_mode == "ZRT"
    assert len(config.sws.filter_bank) == 20
    assert config.sws.windows.nbeg == 10
    assert config.sws.cluster.linkage == "ward"


def test_partial_override(tmp_path):
    config = load_config(
        write(tmp_path, "seiscomp:\n  wait_time: 60\nsws:\n  max_ain: 45.0\n  snr_min: 4.0\n")
    )
    assert config.seiscomp.wait_time == 60
    assert config.sws.max_ain == 45.0
    assert config.sws.snr_min == 4.0
    # untouched sections and nested defaults are kept
    assert config.sds.archive == "/data/archive"
    assert config.sws.windows.tend1 == 0.40


def test_nested_override(tmp_path):
    config = load_config(
        write(tmp_path, "sws:\n  windows:\n    tbeg0: -0.8\n  cluster:\n    k_max: 10\n")
    )
    assert config.sws.windows.tbeg0 == -0.8
    assert config.sws.windows.nend == 20
    assert config.sws.cluster.k_max == 10


@pytest.mark.parametrize(
    "key", ["bogus", "subscriptions", "pick_group", "focmech_group", "load_inventory"]
)
def test_unknown_key_rejected(tmp_path, key):
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(write(tmp_path, f"seiscomp:\n  {key}: 1\n"))


def test_unknown_nested_key_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(write(tmp_path, "sws:\n  windows:\n    bogus: 1\n"))


def test_unknown_section_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown top-level"):
        load_config(write(tmp_path, "bogus:\n  a: 1\n"))


def test_invalid_rotation_mode_rejected(tmp_path):
    with pytest.raises(ConfigError, match="rotation_mode"):
        load_config(write(tmp_path, "sws:\n  rotation_mode: XYZ\n"))


def test_invalid_filter_bank_rejected(tmp_path):
    with pytest.raises(ConfigError, match="filter_bank"):
        load_config(write(tmp_path, "sws:\n  filter_bank: [[5.0, 1.0]]\n"))


def test_invalid_quality_weights_rejected(tmp_path):
    with pytest.raises(ConfigError, match="omega1"):
        load_config(write(tmp_path, "sws:\n  quality:\n    omega1: 0.5\n    omega2: 0.7\n"))


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")

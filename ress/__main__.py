"""RESS entrypoint: python -m ress [--config config.yaml] [SeisComP options]."""

from __future__ import annotations

import argparse
import faulthandler
import logging
import multiprocessing
import signal
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor

from ress import __version__
from ress.config import Config, ConfigError, load_config


def _config_path_from_argv(argv: list[str]) -> str:
    """Pre-parse --config before the SeisComP application takes over argv."""
    parser = argparse.ArgumentParser(prog="ress", allow_abbrev=False)
    parser.add_argument("--version", action="version", version=f"RESS {__version__}")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="YAML file; paths are relative to the working directory",
    )
    parser.add_argument("-H", "--host", help="Override the SeisComP messaging host")
    parser.add_argument("-d", "--database", help="Override the SeisComP database URI")
    parser.add_argument("-w", "--wait-time", type=int, help="Override event wait time in seconds")
    args, _ = parser.parse_known_args(argv[1:])
    if args.wait_time is not None and args.wait_time < 0:
        parser.error("--wait-time must be >= 0")
    return args.config


def _setup_logging(config: Config) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.logging.file:
        handlers.append(logging.FileHandler(config.logging.file))
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def _inject_connection_args(argv: list[str], config: Config) -> list[str]:
    """Default -H / -d from config.yaml unless given on the command line."""
    argv = list(argv)
    # SeisComP derives the messaging client name from basename(argv[0]);
    # with `python -m ress` that would be "__main__", clashing with any
    # other module-run SeisComP client on the same scmaster.
    argv[0] = "ress"
    if config.seiscomp.host and not any(
        a.startswith("-H") or a == "--host" or a.startswith("--host=") for a in argv[1:]
    ):
        argv += ["-H", config.seiscomp.host]
    if config.seiscomp.database and not any(
        a.startswith("-d") or a == "--database" or a.startswith("--database=") for a in argv[1:]
    ):
        argv += ["-d", config.seiscomp.database]
    return argv


def main() -> int:
    try:
        config = load_config(_config_path_from_argv(sys.argv))
        _setup_logging(config)
    except (ConfigError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # dump all thread stacks on SIGUSR1 (docker kill -s USR1 ress)
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    # imported here so config/logging errors surface before the heavy imports
    from ress.app import make_dispatch, make_finalize
    from ress.scclient.listener import EventListenerApp
    from ress.worker import init_worker, run_batch

    # Application.run() holds the GIL, starving in-process threads, so the
    # pipeline runs in its own process; spawn keeps the child clean.
    executor = ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(config,),
    )
    # start the worker now, not on the first event
    try:
        try:
            executor.submit(run_batch, []).result()
        except Exception:
            logging.getLogger("ress").error("Worker initialization failed; stopping")
            return 1
        logging.getLogger("ress").info("Worker process ready")
        argv = _inject_connection_args(sys.argv, config)
        app = EventListenerApp(
            len(argv),
            argv,
            config.seiscomp,
            dispatch=make_dispatch(config),
            process_batch=run_batch,
            finalize=make_finalize(config),
            executor=executor,
        )
        return app()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()
        sys.exit(1)

"""Worker-process entry points for the processing pipeline.

Application.run() holds the GIL almost continuously, starving any Python
thread in the same process. Processing therefore runs in a separate
process with its own GIL; only picklable SwsTask/SwsResult objects
cross the boundary. Heavy imports (ObsPy, scikit-learn) happen only
here, keeping the parent process light.
"""

from __future__ import annotations

import logging

from ress.config import Config
from ress.core.models import SwsResult, SwsTask

_processor = None


def init_worker(config: Config) -> None:
    """ProcessPoolExecutor initializer: build the pipeline in the child."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.DEBUG),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    from ress.core.processor import SwsProcessor

    global _processor
    _processor = SwsProcessor(config)


def run_batch(tasks: list[SwsTask]) -> dict[str, SwsResult]:
    return _processor.process_batch(tasks)

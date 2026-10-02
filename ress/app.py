"""Wires the scclient listener to the core splitting processor.

dispatch runs on the main thread (SeisComP objects), the processor's
process_batch runs on the listener's worker executor (core objects only),
and finalize runs on the main thread again. Database persistence is
best-effort: a failed write never blocks the SeisComP publication.
"""

from __future__ import annotations

import logging

from seiscomp import datamodel

from ress.config import Config
from ress.scclient import OUTPUT_GROUP, pick_io, sws_io

logger = logging.getLogger(__name__)


def make_dispatch(config: Config):
    """Build the main-thread callable that turns an event into SwsTasks."""

    def dispatch(app, event, origin, picks):
        arrivals = pick_io.arrival_index(origin)

        # select the S picks to process
        targets = []
        for pick in picks:
            if not sws_io.is_s_pick(pick, arrivals.get(pick.publicID())):
                continue
            if sws_io.has_sws_comment(pick) and not config.seiscomp.reprocess:
                logger.debug("Pick %s already has a splitting result, skipping", pick.publicID())
                continue
            targets.append(pick)
        logger.info("Processing %d of %d picks", len(targets), len(picks))

        tasks = [sws_io.build_task(pick, event, origin) for pick in targets]
        # the pick objects wait on the main thread until finalize
        context = (targets, {task.pick_id: task for task in tasks})
        return tasks, context

    return dispatch


def make_finalize(config: Config):
    """Build the main-thread callable that publishes and persists results."""
    from ress.db.writer import DbWriter

    # writes are best-effort: failures are logged in finalize, never raised
    writer = DbWriter(config)

    def finalize(app, context, results):
        targets, tasks_by_id = context

        # attach picks to a parent before enabling the notifier so that
        # modifications are tracked as UPDATE operations
        ep = datamodel.EventParameters()
        for pick in targets:
            ep.add(pick)

        modified = 0
        datamodel.Notifier.Enable()
        try:
            for pick in targets:
                result = results.get(pick.publicID())
                if result is None:
                    continue
                if sws_io.apply_result(pick, result):
                    pick.update()
                    modified += 1
                    logger.info(
                        "Pick %s (%s): phi=%s dt=%s ms class=%s",
                        pick.publicID(),
                        pick_io.seed_id(pick),
                        f"{result.ev_phi:.1f}" if result.ev_phi is not None else "N/A",
                        f"{result.ev_dt_ms:.1f}" if result.ev_dt_ms is not None else "N/A",
                        result.quality_class,
                    )
        finally:
            msg = datamodel.Notifier.GetMessage()
            datamodel.Notifier.Disable()

        if modified:
            app.send_message(OUTPUT_GROUP, msg)
        else:
            logger.info("No picks were modified, nothing to send")

        # database persistence is best-effort and must never block messaging
        if writer is not None:
            for pick_id, result in results.items():
                task = tasks_by_id.get(pick_id)
                if task is None:
                    continue
                try:
                    writer.write_result(task, result)
                except Exception:
                    logger.exception("Failed to persist result for pick %s", pick_id)

    return finalize

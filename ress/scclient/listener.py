"""Generic SeisComP event listener application.

Subscribes to messaging, buffers events for a configurable wait time while
collecting updates, then resolves the preferred origin and its picks and
hands them to three injected callables:

    dispatch(app, event, origin, picks) -> (tasks, context)   [main thread]
    process_batch(tasks) -> results                           [worker executor]
    finalize(app, context, results) -> None                   [main thread]

Processing runs on an injected executor so the messaging loop never
blocks. Application.run() holds the GIL, so the executor should be a
ProcessPoolExecutor; tasks and results must be picklable. SeisComP
objects must stay on the main thread: dispatch snapshots them into plain
tasks, and only finalize touches them again.
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import Executor, Future

from seiscomp import client, datamodel

from ress import __version__
from ress.config import SeisCompConfig
from ress.scclient import LOAD_INVENTORY, OUTPUT_GROUP, SUBSCRIPTIONS

logger = logging.getLogger(__name__)


class EventListenerApp(client.Application):
    def __init__(
        self,
        argc,
        argv,
        config: SeisCompConfig,
        dispatch,
        process_batch,
        finalize,
        executor: Executor,
    ):
        client.Application.__init__(self, argc, argv)
        self._config = config
        self._dispatch = dispatch
        self._process_batch = process_batch
        self._finalize = finalize
        self._executor = executor

        self.setMessagingEnabled(True)
        for group in SUBSCRIPTIONS:
            self.addMessagingSubscription(group)
        self.setPrimaryMessagingGroup(OUTPUT_GROUP)

        self.setDatabaseEnabled(True, True)
        self.setLoadConfigModuleEnabled(False)
        self.setLoadInventoryEnabled(LOAD_INVENTORY)
        self.setLoggingToStdErr(True)

        # eventID -> (event_object, first_seen_time)
        self.pending_events: dict[str, tuple] = {}
        self.processed_events: set[str] = set()
        self.wait_time = config.wait_time

        self._futures: dict[str, Future] = {}
        self._in_flight: dict[str, object] = {}  # eventID -> dispatch context (main thread only)

    # reported to scmaster on connect and shown per-client in scm
    def version(self):
        return __version__

    # ------------------------------------------------------------------
    # command line
    # ------------------------------------------------------------------
    def createCommandLineDescription(self):
        client.Application.createCommandLineDescription(self)
        self.commandline().addGroup("Processing")
        self.commandline().addIntOption(
            "Processing",
            "wait-time,w",
            f"Wait time in seconds before processing an event (default: {self.wait_time})",
        )
        self.commandline().addStringOption(
            "Processing",
            "config",
            "Path to the RESS config.yaml (default: config.yaml)",
        )
        return True

    def validateParameters(self):
        if not client.Application.validateParameters(self):
            return False
        try:
            self.wait_time = self.commandline().optionInt("wait-time")
            logger.info("Wait time overridden from the command line: %ds", self.wait_time)
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # messaging callbacks
    # ------------------------------------------------------------------
    def addObject(self, parentID, obj):
        event = datamodel.Event.Cast(obj)
        if event:
            event_id = event.publicID()
            now = time.time()
            if event_id in self.processed_events:
                return
            if event_id not in self.pending_events:
                logger.info("Received NEW event: %s", event_id)
                self.pending_events[event_id] = (event, now)
            else:
                _, first_seen = self.pending_events[event_id]
                logger.info(
                    "Received UPDATE for event: %s (waiting %ds / %ds)",
                    event_id,
                    int(now - first_seen),
                    self.wait_time,
                )
                self.pending_events[event_id] = (event, first_seen)
            return

        origin = datamodel.Origin.Cast(obj)
        if origin:
            # keeps the origin in the memory cache for later lookup
            logger.debug("Received Origin: %s", origin.publicID())

    def updateObject(self, parentID, obj):
        self.addObject(parentID, obj)

    # ------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------
    def handleTimeout(self):
        self._dispatch_ready()
        self._collect_finished()

    def _dispatch_ready(self):
        now = time.time()
        ready = [
            event_id
            for event_id, (_, first_seen) in self.pending_events.items()
            if now - first_seen >= self.wait_time
        ]
        for event_id in ready:
            event, first_seen = self.pending_events.pop(event_id)
            logger.info("Dispatching event %s after %ds wait", event_id, int(now - first_seen))
            self._dispatch_event(event)
            self.processed_events.add(event_id)

    def _collect_finished(self):
        done = [event_id for event_id, future in self._futures.items() if future.done()]
        for event_id in done:
            future = self._futures.pop(event_id)
            context = self._in_flight.pop(event_id)
            try:
                results = future.result()
            except Exception:
                logger.error("ERROR processing event %s:\n%s", event_id, traceback.format_exc())
                continue
            try:
                self._finalize(self, context, results)
            except Exception:
                logger.error("ERROR finalizing event %s:\n%s", event_id, traceback.format_exc())

    # ------------------------------------------------------------------
    # event processing
    # ------------------------------------------------------------------
    def _load_origin(self, origin_id):
        origin = datamodel.Origin.Find(origin_id)
        if not origin:
            logger.debug("Origin not in memory cache, loading from database...")
            obj = self.query().loadObject(datamodel.Origin.TypeInfo(), origin_id)
            origin = datamodel.Origin.Cast(obj)
        if origin and origin.arrivalCount() == 0:
            self.query().loadArrivals(origin)
        return origin

    def _load_picks(self, origin) -> list:
        # exhaust the iterator immediately so the DB cursor is closed
        # before any further database access
        db_objects = list(self.query().getPicks(origin.publicID()))
        picks = []
        for obj in db_objects:
            pick = datamodel.Pick.Cast(obj)
            if pick:
                self.query().loadComments(pick)
                picks.append(pick)
        return picks

    def _dispatch_event(self, event):
        event_id = event.publicID()
        try:
            origin_id = event.preferredOriginID()
            origin = self._load_origin(origin_id)
            if not origin:
                logger.warning("Could not load origin %s for event %s", origin_id, event_id)
                return
            logger.info(
                "Event %s | origin %s | %s | lat %.4f lon %.4f depth %.1f km | %d arrivals",
                event_id,
                origin.publicID(),
                origin.time().value().toString("%Y-%m-%d %H:%M:%S"),
                origin.latitude().value(),
                origin.longitude().value(),
                origin.depth().value(),
                origin.arrivalCount(),
            )
            picks = self._load_picks(origin)
            logger.info("Loaded %d picks for origin %s", len(picks), origin.publicID())
            tasks, context = self._dispatch(self, event, origin, picks)
            if not tasks:
                logger.info("No picks to process for event %s", event_id)
                return
            self._in_flight[event_id] = context
            self._futures[event_id] = self._executor.submit(self._process_batch, tasks)
        except Exception:
            logger.error("ERROR dispatching event %s:\n%s", event_id, traceback.format_exc())

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------
    def send_message(self, group: str, msg) -> bool:
        if not self.connection():
            logger.error("No messaging connection available")
            return False
        if not msg:
            logger.warning("Empty message, nothing to send")
            return False
        if self.connection().send(group, msg):
            logger.info("Sent message with %d notifiers to group %s", msg.size(), group)
            return True
        logger.error("Failed to send message to group %s", group)
        return False

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self):
        logger.info(
            "Connected. Watching for events (wait time: %ds). Press Ctrl+C to exit.",
            self.wait_time,
        )
        # check pending events every second
        self.enableTimer(1)
        return client.Application.run(self)

    def done(self):
        if self._futures:
            logger.warning("Shutting down with %d unfinished event(s)", len(self._futures))
        self._executor.shutdown(wait=True, cancel_futures=True)
        client.Application.done(self)

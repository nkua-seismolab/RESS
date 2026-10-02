"""Generic SeisComP client: event listening, pick retrieval and publishing.

This package owns every transaction with SeisComP (messaging + database).
It is science-agnostic: EventListenerApp delegates ready events to an
injected handler callable.
"""

SUBSCRIPTIONS = ("EVENT", "LOCATION")
OUTPUT_GROUP = "PICK"
LOAD_INVENTORY = True

"""Outbox — fire-and-forget HTTP notification when a drawer changes in a
tracked wing (xc-tracker push sync). Fail-soft by contract: a palace write
must never fail or slow down because the consumer is unreachable."""

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)
_TIMEOUT_S = 2


def emit(config, wing: str, drawer_id: str, event: str) -> None:
    try:
        ob = config.outbox
        url = ob.get("url")
        if not url or wing not in ob.get("wings", []):
            return
        req = urllib.request.Request(
            url,
            data=json.dumps({"wing": wing, "drawer_id": drawer_id, "event": event}).encode(),
            headers={"Content-Type": "application/json", "x-palace-secret": ob.get("secret", "")},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S):
            pass
    except Exception as e:  # noqa: BLE001 — deliberately swallow everything
        logger.debug("outbox emit failed (ignored): %s", e)

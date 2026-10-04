"""Hide control-plane polling traffic from uvicorn's access log.

Clients poll a handful of read-only endpoints every 1-2s to keep a UI current: GET /health
(lifecycle), GET /v1/stats (runtime metrics), GET /v1/requests (request ring, carries a
?since=&limit= query string), GET /v1/cache/status, and a bare GET /v1 liveness probe.
Uvicorn's uvicorn.access logger logs every one of these at INFO, flooding the server's
output with lines nobody reads. They are treated as debug-level noise: suppressed unless
debug logging is on. Every other request, such as POST /v1/chat/completions, is always
logged.

Port of FreeToken server/access_log_filter.py. Divergence: debug logging is on when the
ftmetal loggers are enabled for DEBUG (this engine has no LOG_LEVEL variable).
"""

from __future__ import annotations

import logging

# Path prefixes considered polling traffic. Matched against the request path with any query
# string stripped (uvicorn's access record embeds the query string in the logged path, e.g.
# "/v1/requests?since=123&limit=50").
POLLING_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/v1/stats",
    "/v1/requests",
    "/v1/cache/status",
)

# The bare "/v1" probe is matched exactly, never as a prefix: every path above (and other
# real endpoints like "/v1/models") also starts with "/v1".
BARE_PROBE_PATH = "/v1"


class PollingAccessLogFilter(logging.Filter):
    """logging.Filter for the uvicorn.access logger: drops the polling endpoints' records
    unless debug logging is on. Records that don't look like uvicorn's access log shape, or
    whose path isn't a polling path, always pass through unchanged."""

    def filter(self, record: logging.LogRecord) -> bool:
        if logging.getLogger("ftmetal").isEnabledFor(logging.DEBUG):
            return True

        # uvicorn's h11/httptools protocols log access lines as:
        #   access_logger.info('%s - "%s %s HTTP/%s" %d', client_addr, method, full_path,
        #                       http_version, status)
        # i.e. record.args == (client_addr, method, full_path, http_version, status).
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3:
            return True

        full_path = args[2]
        if not isinstance(full_path, str):
            return True

        path = full_path.split("?", 1)[0]

        if path == BARE_PROBE_PATH:
            return False
        return not any(path.startswith(prefix) for prefix in POLLING_PATH_PREFIXES)


def install_polling_access_log_filter() -> None:
    """Attach PollingAccessLogFilter to uvicorn's access logger, once: idempotent, so building
    several apps in one process (tests do) does not pile up duplicate filters."""
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, PollingAccessLogFilter) for f in access_logger.filters):
        access_logger.addFilter(PollingAccessLogFilter())


__all__ = ["PollingAccessLogFilter", "install_polling_access_log_filter"]

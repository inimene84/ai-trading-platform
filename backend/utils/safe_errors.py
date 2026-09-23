"""Safe error responses and log sanitising.

API responses must never echo raw exception text (it can leak stack details,
file paths, hostnames or upstream payloads). Instead we log the full exception
server-side under a short correlation id and return only that id to the caller,
so an operator can grep the backend logs for the exact failure.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from fastapi.responses import JSONResponse

# CR/LF and other C0 control characters (except tab) let an attacker forge
# fake log lines when user-controlled text is written to a log record.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0a-\x1f\x7f]")


def new_error_id() -> str:
    return uuid.uuid4().hex[:12]


def log_exception(logger: logging.Logger, context: str, exc: BaseException) -> str:
    """Log ``exc`` with a correlation id and return that id."""
    error_id = new_error_id()
    logger.error("%s [error_id=%s]: %s", context, error_id, exc, exc_info=exc)
    return error_id


def internal_error_response(
    logger: logging.Logger,
    context: str,
    exc: BaseException,
    status_code: int = 500,
    **extra: Any,
) -> JSONResponse:
    """Log ``exc`` and return a generic JSON error that carries only an id."""
    error_id = log_exception(logger, context, exc)
    body = {"error": f"{context} (internal error)", "error_id": error_id}
    body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


def sanitize_log_text(value: str) -> str:
    """Replace control characters (CR, LF, ...) so one record stays one line."""
    return _CONTROL_CHARS.sub(lambda m: "\\n" if m.group() == "\n" else "\\r" if m.group() == "\r" else "?", value)


class LogInjectionFilter(logging.Filter):
    """Neutralise log-forging characters in every record's message and args.

    Installed on the root handlers at startup, so it covers every logger in the
    process (including the f-string log calls CodeQL flags as py/log-injection).
    Tracebacks (``exc_info``) are formatted separately and stay multi-line.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = sanitize_log_text(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: sanitize_log_text(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                else:
                    record.args = tuple(
                        sanitize_log_text(a) if isinstance(a, str) else a
                        for a in record.args
                    )
        except Exception:  # never let logging hygiene break logging itself
            pass
        return True


def install_log_injection_filter() -> None:
    """Attach :class:`LogInjectionFilter` to all root handlers (idempotent)."""
    root = logging.getLogger()
    for handler in root.handlers:
        if not any(isinstance(f, LogInjectionFilter) for f in handler.filters):
            handler.addFilter(LogInjectionFilter())

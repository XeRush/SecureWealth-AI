"""
Middleware stack for SecureWealth Twin AI.

Applied in order (Starlette reverses add_middleware order):
1. TraceIDMiddleware   — generates UUID trace_id, attaches to request.state and X-Trace-ID header
2. ConsentValidationMiddleware — checks consent_token on PII-bearing endpoints; returns 403 if absent
3. PIIMaskingMiddleware — installs a logging filter that masks account numbers with truncated SHA-256 hashes
"""

import hashlib
import json
import logging
import re
import uuid

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# ---------------------------------------------------------------------------
# PII masking helpers
# ---------------------------------------------------------------------------

# Matches standalone 10–18 digit sequences (account numbers, card numbers, etc.)
ACCOUNT_NUMBER_PATTERN = re.compile(r"\b\d{10,18}\b")


def _mask_pii(text: str) -> str:
    """Replace account-number-like digit sequences with truncated SHA-256 hashes."""

    def replace_match(m: re.Match) -> str:
        h = hashlib.sha256(m.group().encode()).hexdigest()[:8]
        return f"[masked:{h}]"

    return ACCOUNT_NUMBER_PATTERN.sub(replace_match, text)


class PIIMaskingFilter(logging.Filter):
    """Logging filter that masks PII in log record messages and args."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _mask_pii(str(record.msg))
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(_mask_pii(str(a)) for a in record.args)
            # dict-style args (rare) — leave as-is to avoid breaking format strings
        return True


# ---------------------------------------------------------------------------
# Middleware classes
# ---------------------------------------------------------------------------


class TraceIDMiddleware(BaseHTTPMiddleware):
    """Generate a UUID trace_id per request; attach to request.state and X-Trace-ID header."""

    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response


# Endpoints that carry personally identifiable financial data
PII_ENDPOINTS = {"/analyze-user", "/simulate", "/risk-check", "/decision", "/chat"}


class ConsentValidationMiddleware(BaseHTTPMiddleware):
    """
    Check that a consent_token is present in the request body for PII-bearing endpoints.
    Returns HTTP 403 if the token is absent or empty.
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PII_ENDPOINTS:
            body_bytes = await request.body()
            try:
                body_json = json.loads(body_bytes)
                # consent_token may be at the top level or nested inside a "profile" object
                consent_token = body_json.get("consent_token") or (
                    body_json.get("profile") or {}
                ).get("consent_token")
                if not consent_token:
                    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "consent_required",
                            "message": "Data processing consent is required.",
                            "trace_id": trace_id,
                        },
                    )
            except Exception:
                # Malformed JSON — let Pydantic validation handle it downstream
                pass

            # Re-inject the body so downstream handlers can read it again
            async def receive():
                return {"type": "http.request", "body": body_bytes}

            request._receive = receive  # type: ignore[attr-defined]

        return await call_next(request)


class PIIMaskingMiddleware(BaseHTTPMiddleware):
    """
    Install a PII masking filter on the root logger so that account numbers and
    other identifiable digit sequences are replaced with truncated SHA-256 hashes
    in all log output.
    """

    def __init__(self, app):
        super().__init__(app)
        pii_filter = PIIMaskingFilter()
        logging.getLogger().addFilter(pii_filter)

    async def dispatch(self, request: Request, call_next):
        return await call_next(request)

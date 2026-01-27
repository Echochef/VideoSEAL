from __future__ import annotations

from typing import Any, Optional


_REQUEST_ID_HEADERS = (
    "x-request-id",
    "x-requestid",
    "request-id",
    "apim-request-id",
    "x-ms-request-id",
    "x-dashscope-request-id",
    "x-dashscope-requestid",
    "x-amzn-requestid",
    "x-amz-request-id",
    "x-trace-id",
)


def extract_request_id_from_headers(headers: Any) -> Optional[str]:
    if not headers:
        return None
    try:
        items = list(headers.items())
    except Exception:
        try:
            items = list(dict(headers).items())
        except Exception:
            return None

    for k, v in items:
        lk = str(k).strip().lower()
        if lk in _REQUEST_ID_HEADERS:
            s = str(v).strip()
            if s:
                return s

    for k, v in items:
        lk = str(k).strip().lower()
        if lk.endswith(("request-id", "requestid")):
            s = str(v).strip()
            if s:
                return s
    return None


def extract_request_id(exc: BaseException) -> Optional[str]:
    for attr in ("request_id", "requestId"):
        rid = getattr(exc, attr, None)
        if rid is not None:
            s = str(rid).strip()
            if s:
                return s

    resp = getattr(exc, "response", None)
    if resp is not None:
        rid = extract_request_id_from_headers(getattr(resp, "headers", None))
        if rid:
            return rid

    rid = extract_request_id_from_headers(getattr(exc, "headers", None))
    if rid:
        return rid
    return None


def format_exception_with_request_id(exc: BaseException) -> str:
    rid = extract_request_id(exc)
    base = f"{type(exc).__name__}: {exc}"
    if rid:
        return f"{base} (request_id={rid})"
    return base


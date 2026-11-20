"""Helpers that keep secrets and customer text out of log files."""

from __future__ import annotations

from urllib.parse import urlsplit


def redact_url(url: object) -> str:
    """Return a log-safe description of a (possibly signed) URL.

    Signed upload URLs carry bearer credentials in the query string and can
    also encode object names in the path, so only the scheme and host are kept.
    """
    if not isinstance(url, str) or not url:
        return "<no url>"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url redacted>"
    if not parts.scheme or not parts.hostname:
        return "<url redacted>"
    return f"{parts.scheme}://{parts.hostname}/<redacted>"

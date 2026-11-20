from __future__ import annotations

from urllib.parse import urlsplit


def redact_url(url: object) -> str:
    if not isinstance(url, str) or not url:
        return "<no url>"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<unparseable url redacted>"
    if not parts.scheme or not parts.hostname:
        return "<url redacted>"
    return f"{parts.scheme}://{parts.hostname}/<redacted>"

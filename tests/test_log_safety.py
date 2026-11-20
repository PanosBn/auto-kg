import pytest

from preprocessing.log_safety import redact_url


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.example.invalid/bucket/customer-file.json?X-Goog-Signature=abc123&X-Goog-Credential=svc%40x",
        "https://user:pa55@storage.example.invalid/path?sig=abc123",
    ],
)
def test_redact_url_drops_credentials_path_and_query(url):
    redacted = redact_url(url)
    assert redacted == "https://storage.example.invalid/<redacted>"
    for secret in ("abc123", "Signature", "customer-file", "pa55", "user"):
        assert secret not in redacted


@pytest.mark.parametrize("value", [None, "", 42, "not a url", "http://[::1"])
def test_redact_url_handles_odd_values(value):
    out = redact_url(value)
    assert "not a url" not in out
    assert out.startswith("<")

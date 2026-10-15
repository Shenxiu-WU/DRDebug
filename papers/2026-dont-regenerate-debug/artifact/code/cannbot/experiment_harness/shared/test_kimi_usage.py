#!/usr/bin/env python3
"""Offline tests for the independent Kimi quota query module."""

from __future__ import annotations

import io
import json
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import patch

from kimi_usage import Provider, load_providers, query_kimi_usage, query_pool


class Response:
    def __init__(self, body: dict, status: int = 200):
        self.status = status
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit: int) -> bytes:
        return self.body


SUCCESS = {
    "usage": {"limit": "100", "used": "35", "remaining": "65", "resetTime": "2026-07-18T13:27:27Z"},
    "limits": [
        {"detail": {"limit": "100", "used": "2", "remaining": "98", "resetTime": "2026-07-17T14:27:27Z"}}
    ],
    "totalQuota": {"limit": "100", "remaining": "99"},
    "parallel": {"limit": "20"},
}


def main() -> int:
    provider = Provider("safe-name", "example-token", "k3", "https://api.kimi.com/coding/")
    with patch("urllib.request.urlopen", return_value=Response(SUCCESS)):
        row = query_kimi_usage(provider)
    assert row["success"] is True
    assert row["weekly"]["remaining_pct"] == 65.0
    assert row["five_hour"]["remaining"] == "98"
    assert row["parallel_limit"] == "20"
    assert "example-token" not in json.dumps(row)

    error = urllib.error.HTTPError(
        "https://api.kimi.com/coding/v1/usages",
        429,
        "quota",
        {},
        io.BytesIO(b'{"error":{"message":"quota reached"}}'),
    )
    with patch("urllib.request.urlopen", side_effect=error):
        failed = query_kimi_usage(provider)
    assert failed["success"] is False and failed["http_status"] == 429

    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "keys.json"
        config.write_text(
            json.dumps(
                {
                    "model": "k3",
                    "base_url": "https://api.kimi.com/coding/",
                    "keys": [{"name": "safe-name", "api_key": "example-token"}],
                }
            )
        )
        assert load_providers(config)[0].model == "k3"
        with patch("urllib.request.urlopen", return_value=Response(SUCCESS)):
            payload = query_pool(config)
        assert payload["providers"][0]["weekly"]["reset_time_local"].endswith("+08:00")
        assert "example-token" not in json.dumps(payload)

    print("PASS: independent quota parsing, errors, timezone conversion, and key redaction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

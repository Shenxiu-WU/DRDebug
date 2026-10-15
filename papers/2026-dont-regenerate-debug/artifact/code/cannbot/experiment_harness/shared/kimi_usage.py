#!/usr/bin/env python3
"""Read Kimi Code quota state for a key pool without exposing API keys.

This module is self-contained.  It does not import or read AscendOpGenAgent.
The query is a read-only GET and does not invoke a model or consume tokens.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


USAGE_URL = "https://api.kimi.com/coding/v1/usages"
DEFAULT_KEY_CONFIG = Path(
    "/workspace/regenerate_experiment/.secrets/provider_config.json"
)


@dataclass(frozen=True)
class Provider:
    name: str
    api_key: str
    model: str
    base_url: str


def load_providers(path: Path) -> list[Provider]:
    data = json.loads(path.read_text())
    default_model = str(data.get("model") or "k3")
    default_base_url = str(data.get("base_url") or "https://api.kimi.com/coding/")
    providers = [
        Provider(
            name=str(item["name"]),
            api_key=str(item["api_key"]),
            model=str(item.get("model") or default_model),
            base_url=str(item.get("base_url") or default_base_url),
        )
        for item in data.get("keys", [])
        if isinstance(item, dict) and item.get("name") and item.get("api_key")
    ]
    if not providers:
        raise ValueError(f"no usable Kimi providers in {path}")
    return providers


def _window(detail: Any) -> dict[str, Any] | None:
    if not isinstance(detail, dict):
        return None
    limit = detail.get("limit")
    remaining = detail.get("remaining")
    try:
        remaining_pct = round(float(remaining) * 100.0 / float(limit), 2) if float(limit) else None
    except (TypeError, ValueError, ZeroDivisionError):
        remaining_pct = None
    return {
        "limit": limit,
        "used": detail.get("used"),
        "remaining": remaining,
        "remaining_pct": remaining_pct,
        "reset_time": detail.get("resetTime"),
    }


def query_kimi_usage(
    provider: Provider,
    timeout: int = 30,
    *,
    user_agent: str = "cannbot-quota-monitor/1.0",
) -> dict[str, Any]:
    """Query one provider's Kimi Code quotas without returning its API key."""
    state: dict[str, Any] = {
        "name": provider.name,
        "model": provider.model,
        "base_url": provider.base_url,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "supported": "api.kimi.com/coding" in provider.base_url.lower(),
    }
    if not state["supported"]:
        state.update({"success": False, "error": "unsupported Kimi Code base URL"})
        return state

    request = urllib.request.Request(
        USAGE_URL,
        headers={
            "Authorization": f"Bearer {provider.api_key}",
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            state["http_status"] = response.status
            raw = response.read(50_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        state["http_status"] = exc.code
        raw = exc.read(10_000).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - preserve a sanitized operational error
        state.update({"success": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
        return state

    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        state.update({"success": False, "error": raw[:500]})
        return state
    state["success"] = 200 <= int(state.get("http_status", 0)) < 300
    if not state["success"]:
        state["error"] = body.get("error", body) if isinstance(body, dict) else body
        return state
    if not isinstance(body, dict):
        state.update({"success": False, "error": "usage response is not an object"})
        return state

    state["weekly"] = _window(body.get("usage"))
    limits = body.get("limits")
    first = limits[0] if isinstance(limits, list) and limits and isinstance(limits[0], dict) else {}
    state["five_hour"] = _window(first.get("detail"))
    state["total_quota"] = _window(body.get("totalQuota"))
    parallel = body.get("parallel")
    state["parallel_limit"] = parallel.get("limit") if isinstance(parallel, dict) else None
    return state


def _with_local_reset(window: Any, zone: ZoneInfo) -> Any:
    if not isinstance(window, dict):
        return window
    result = dict(window)
    raw = window.get("reset_time")
    if raw:
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            result["reset_time_local"] = parsed.astimezone(zone).isoformat()
        except ValueError:
            result["reset_time_local"] = raw
    else:
        result["reset_time_local"] = None
    return result


def query_pool(path: Path, timeout: int = 30, timezone_name: str = "Asia/Shanghai") -> dict[str, Any]:
    zone = ZoneInfo(timezone_name)
    rows: list[dict[str, Any]] = []
    for priority, provider in enumerate(load_providers(path), 1):
        row = query_kimi_usage(provider, timeout)
        row["priority"] = priority
        for field in ("five_hour", "weekly", "total_quota"):
            row[field] = _with_local_reset(row.get(field), zone)
        rows.append(row)
    return {
        "queried_at": datetime.now(zone).isoformat(),
        "timezone": timezone_name,
        "key_config": str(path),
        "providers": rows,
    }


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-config", type=Path, default=DEFAULT_KEY_CONFIG)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--timezone", default="Asia/Shanghai")
    parser.add_argument("--compact", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    payload = query_pool(args.key_config, args.timeout, args.timezone)
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2))
    return 0 if all(row.get("success") for row in payload["providers"]) else 2


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

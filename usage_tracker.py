"""Persistent usage counter for EconoLens AI.

The app records one usage event per Streamlit browser session.
For Streamlit Community Cloud, configure Supabase credentials in
Streamlit secrets to make the total persistent across redeployments.
A local JSON fallback is provided for local development only.

Author: Rishabh Shah
Advisor: Dr. Qingyang Xiao
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


APP_COUNTER_RPC = "increment_econolens_usage"
LOCAL_COUNTER_PATH = Path(__file__).resolve().parent / ".streamlit" / "usage_counter.json"
_LOCK = threading.Lock()


def _credential(name: str, secrets: Any | None = None) -> str:
    """Read a secret from Streamlit secrets when available, then env vars."""
    if secrets is not None:
        try:
            value = secrets.get(name)
            if value:
                return str(value).strip()
        except Exception:
            pass
    return os.getenv(name, "").strip()


def _supabase_credentials(secrets: Any | None = None) -> tuple[str, str]:
    url = _credential("SUPABASE_URL", secrets).rstrip("/")
    key = _credential("SUPABASE_KEY", secrets)
    return url, key


def _parse_rpc_total(payload: Any) -> int:
    """Normalize the scalar/object/list response returned by Supabase RPC."""
    if isinstance(payload, bool):
        return int(payload)
    if isinstance(payload, int):
        return payload
    if isinstance(payload, float):
        return int(payload)
    if isinstance(payload, list) and payload:
        return _parse_rpc_total(payload[0])
    if isinstance(payload, dict):
        for key in ("increment_econolens_usage", "total_uses", "count", "value"):
            if key in payload:
                return _parse_rpc_total(payload[key])
    if isinstance(payload, str):
        return int(payload.strip())
    raise ValueError(f"Unexpected usage-counter response: {payload!r}")


def _increment_supabase(url: str, key: str, timeout: float = 6.0) -> int:
    endpoint = f"{url}/rest/v1/rpc/{APP_COUNTER_RPC}"
    response = requests.post(
        endpoint,
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={},
        timeout=timeout,
    )
    response.raise_for_status()
    return _parse_rpc_total(response.json())


def _read_local() -> dict[str, Any]:
    try:
        return json.loads(LOCAL_COUNTER_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"total_uses": 0, "last_used_at": None}


def _increment_local() -> int:
    LOCAL_COUNTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        data = _read_local()
        total = int(data.get("total_uses", 0)) + 1
        data = {
            "total_uses": total,
            "last_used_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            LOCAL_COUNTER_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            # A read-only cloud/container is allowed to fall through to a
            # session-only counter rather than breaking the application.
            pass
        return total


def record_visit(secrets: Any | None = None) -> tuple[int, str]:
    """Record one visit and return ``(total_uses, backend_name)``.

    The caller should invoke this once per Streamlit session, not on every
    widget rerun. Supabase is preferred for persistent production storage;
    local JSON is a development fallback.
    """
    url, key = _supabase_credentials(secrets)
    if url and key:
        try:
            return _increment_supabase(url, key), "Supabase"
        except Exception:
            # Do not block the app if an external counter is temporarily down.
            pass
    return _increment_local(), "Local fallback"

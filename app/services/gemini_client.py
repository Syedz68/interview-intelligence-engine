"""
gemini_client.py
----------------
Shared Gemini API client with key rotation to avoid 429 rate-limit errors.

Usage
-----
In your .env, set up to 3 keys:

    GEMINI_API_KEY=AIza...key1...
    GEMINI_API_KEY_2=AIza...key2...
    GEMINI_API_KEY_3=AIza...key3...

Only GEMINI_API_KEY is required. Keys 2 and 3 are optional extras.

Each service (english_fluency, answer_relevance, behavioral_report) gets its
own "slot" in the rotation so they prefer different keys, spreading the load.
If a key returns 429, the client automatically retries with the next key.
"""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any

import requests

logger = logging.getLogger(__name__)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)

# ── Key pool ──────────────────────────────────────────────────────────────────

def _load_keys() -> list[str]:
    """Load all configured Gemini API keys from environment."""
    keys: list[str] = []
    for env_var in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        k = os.getenv(env_var, "").strip()
        if k:
            keys.append(k)
    return keys


class _KeyRotator:
    """Thread-safe round-robin key rotator with per-slot offset."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._lock = Lock()
        self._index: int = 0  # global next-key index

    def _refresh(self) -> None:
        """Re-read keys from env (supports hot-reload if env changes)."""
        fresh = _load_keys()
        if fresh != self._keys:
            logger.info("Gemini key pool updated: %d key(s) available", len(fresh))
            self._keys = fresh

    def get_keys_starting_from(self, slot: int) -> list[str]:
        """
        Return all keys in rotation order starting at `slot % len(keys)`.
        First key in the list is preferred for this service slot.
        """
        self._refresh()
        if not self._keys:
            return []
        n = len(self._keys)
        start = slot % n
        return [self._keys[(start + i) % n] for i in range(n)]


_rotator = _KeyRotator()

# Service slots — each service prefers a different starting key
SLOT_ENGLISH_FLUENCY = 0
SLOT_ANSWER_RELEVANCE = 1
SLOT_BEHAVIORAL_REPORT = 2


# ── HTTP call with retry ───────────────────────────────────────────────────────

def call_gemini(
    prompt: str,
    *,
    slot: int = 0,
    temperature: float = 0.4,
    max_tokens: int = 2048,
    timeout: int = 60,
    max_retries: int | None = None,  # defaults to number of keys
) -> str:
    """
    Call Gemini generateContent with automatic key rotation on 429.

    Parameters
    ----------
    prompt      : full prompt string
    slot        : service slot (0=fluency, 1=relevance, 2=report) to pick
                  starting key
    temperature : generation temperature
    max_tokens  : max output tokens
    timeout     : per-request timeout in seconds
    max_retries : how many keys to try before giving up (default = all keys)

    Returns
    -------
    Raw text response from Gemini.

    Raises
    ------
    RuntimeError if all keys are exhausted or another unrecoverable error occurs.
    """
    keys = _rotator.get_keys_starting_from(slot)
    if not keys:
        raise RuntimeError(
            "No GEMINI_API_KEY configured. Set GEMINI_API_KEY in .env"
        )

    if max_retries is None:
        max_retries = len(keys)

    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }

    last_error: Exception | None = None

    for attempt, key in enumerate(keys[:max_retries]):
        masked = key[:8] + "…" + key[-4:]
        try:
            logger.debug(
                "Gemini call attempt %d/%d using key %s",
                attempt + 1, max_retries, masked
            )
            resp = requests.post(
                f"{GEMINI_API_URL}?key={key}",
                json=payload,
                timeout=timeout,
            )

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning(
                    "Gemini 429 on key %s (attempt %d/%d). "
                    "Waiting %ds then trying next key …",
                    masked, attempt + 1, max_retries, retry_after
                )
                time.sleep(min(retry_after, 10))  # cap at 10s
                last_error = RuntimeError(f"429 from key {masked}")
                continue

            if resp.status_code == 401 or resp.status_code == 403:
                logger.warning(
                    "Gemini auth error %d on key %s — skipping this key",
                    resp.status_code, masked
                )
                last_error = RuntimeError(
                    f"Auth error {resp.status_code} from key {masked}"
                )
                continue

            resp.raise_for_status()

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(
                    f"Gemini returned no candidates: {data.get('promptFeedback', data)}"
                )

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise RuntimeError("Gemini candidate has no parts")

            return parts[0].get("text", "")

        except (requests.Timeout, requests.ConnectionError) as net_exc:
            logger.warning(
                "Network error on attempt %d/%d: %s",
                attempt + 1, max_retries, net_exc
            )
            last_error = net_exc
            time.sleep(2)
            continue

        except RuntimeError:
            raise  # Already logged above; re-raise immediately

        except Exception as exc:
            logger.exception("Unexpected Gemini error on attempt %d: %s", attempt + 1, exc)
            last_error = exc
            break

    raise RuntimeError(
        f"All {max_retries} Gemini key attempt(s) failed. Last error: {last_error}"
    )

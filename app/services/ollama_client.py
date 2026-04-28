"""
ollama_client.py
----------------
Shared Ollama AI client supporting both local and cloud-hosted Ollama instances.

Configuration (.env)
--------------------
# Required
OLLAMA_BASE_URL=http://localhost:11434   # local  OR  https://your-cloud-ollama.com
OLLAMA_MODEL=llama3.1                   # any model pulled in Ollama

# Optional tuning
OLLAMA_TIMEOUT=120                      # seconds per request (default 120)
OLLAMA_MAX_RETRIES=3                    # retry attempts on transient errors

Local setup
-----------
  1. Install Ollama: https://ollama.com/download
  2. Pull a model:   ollama pull llama3.1
  3. Start server:   ollama serve  (or it auto-starts on most platforms)

Cloud / remote Ollama
---------------------
  Point OLLAMA_BASE_URL at any Ollama-compatible endpoint.
  If your server requires an API key, set OLLAMA_API_KEY and it will be sent
  as a Bearer token in the Authorization header.

Slot constants (for service-level logging)
------------------------------------------
  SLOT_ENGLISH_FLUENCY   = 0
  SLOT_ANSWER_RELEVANCE  = 1
  SLOT_BEHAVIORAL_REPORT = 2
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Slot labels (mirrors gemini_client convention) ────────────────────────────
SLOT_ENGLISH_FLUENCY   = 0
SLOT_ANSWER_RELEVANCE  = 1
SLOT_BEHAVIORAL_REPORT = 2

_SLOT_NAMES = {
    SLOT_ENGLISH_FLUENCY:   "english_fluency",
    SLOT_ANSWER_RELEVANCE:  "answer_relevance",
    SLOT_BEHAVIORAL_REPORT: "behavioral_report",
}


# ── Config helpers ────────────────────────────────────────────────────────────

def _base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

def _model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1")

def _timeout() -> int:
    return int(os.getenv("OLLAMA_TIMEOUT", "120"))

def _max_retries() -> int:
    return int(os.getenv("OLLAMA_MAX_RETRIES", "3"))

def _api_key() -> str | None:
    """Optional Bearer token for cloud/authenticated Ollama deployments."""
    return os.getenv("OLLAMA_API_KEY", "").strip() or None


def is_configured() -> bool:
    """Return True if OLLAMA_BASE_URL and OLLAMA_MODEL are set (or defaults work)."""
    # Always true for local; for cloud the caller should verify connectivity.
    return bool(_base_url() and _model())


# ── Core call ─────────────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    *,
    slot: int = 0,
    temperature: float = 0.4,
    max_tokens: int = 2048,
    timeout: int | None = None,
    max_retries: int | None = None,
) -> str:
    """
    Send a prompt to Ollama and return the model's text response.

    Parameters
    ----------
    prompt      : Full prompt string (system + user combined).
    slot        : Service slot for log labelling (0=fluency, 1=relevance, 2=report).
    temperature : Sampling temperature (0.0 = deterministic).
    max_tokens  : Max tokens the model may generate.
    timeout     : Per-request timeout in seconds (default: OLLAMA_TIMEOUT env).
    max_retries : Retry attempts on transient network errors (default: OLLAMA_MAX_RETRIES env).

    Returns
    -------
    Raw text response from the model.

    Raises
    ------
    RuntimeError if all retries are exhausted or an unrecoverable error occurs.
    """
    base      = _base_url()
    model     = _model()
    endpoint  = f"{base}/api/generate"
    tout      = timeout or _timeout()
    retries   = max_retries if max_retries is not None else _max_retries()
    slot_name = _SLOT_NAMES.get(slot, f"slot_{slot}")
    api_key   = _api_key()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature":   temperature,
            "num_predict":   max_tokens,
        },
    }

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.debug(
                "[%s] Ollama call attempt %d/%d → %s  model=%s",
                slot_name, attempt, retries, endpoint, model,
            )
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=tout)

            # Surface helpful errors before raise_for_status
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Ollama model '{model}' not found. "
                    f"Run:  ollama pull {model}"
                )
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Ollama auth error {resp.status_code}. "
                    "Check OLLAMA_API_KEY if your server requires authentication."
                )

            resp.raise_for_status()

            data = resp.json()
            text = data.get("response", "").strip()

            if not text:
                raise RuntimeError(
                    f"Ollama returned an empty response. Full payload: {data}"
                )

            logger.debug("[%s] Ollama call succeeded (%d chars)", slot_name, len(text))
            return text

        except (requests.Timeout, requests.ConnectionError) as net_exc:
            wait = 2 * attempt  # simple backoff: 2s, 4s, 6s …
            logger.warning(
                "[%s] Network error on attempt %d/%d: %s — retrying in %ds",
                slot_name, attempt, retries, net_exc, wait,
            )
            last_error = net_exc
            if attempt < retries:
                time.sleep(wait)
            continue

        except RuntimeError:
            raise  # already descriptive; propagate immediately

        except Exception as exc:
            logger.exception("[%s] Unexpected error on attempt %d: %s", slot_name, attempt, exc)
            last_error = exc
            break

    raise RuntimeError(
        f"[{slot_name}] All {retries} Ollama attempt(s) failed. "
        f"Last error: {last_error}"
    )


def check_connection() -> dict[str, Any]:
    """
    Lightweight connectivity check — useful for startup health checks.

    Returns a dict with keys: ok (bool), url, model, error (str | None).
    """
    base  = _base_url()
    model = _model()
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        available = [m["name"] for m in resp.json().get("models", [])]
        model_ready = any(m.startswith(model.split(":")[0]) for m in available)
        return {
            "ok":          model_ready,
            "url":         base,
            "model":       model,
            "available_models": available,
            "error":       None if model_ready else (
                f"Model '{model}' not pulled. Run: ollama pull {model}"
            ),
        }
    except Exception as exc:
        return {"ok": False, "url": base, "model": model, "error": str(exc)}

"""
speech_analysis_service.py
--------------------------
Derives communication-quality metrics from the Whisper STT output
(transcript text + timed segments).

No additional ML model is needed – all metrics are computed from the
segment timestamps and raw text.

Outputs
-------
{
    "speech_rate_wpm": float,        # words per minute
    "pause_count": int,              # number of pauses > threshold
    "avg_pause_duration": float,     # seconds
    "long_pause_count": int,         # pauses > LONG_PAUSE_THRESHOLD
    "filler_word_count": int,        # "um", "uh", "like", etc.
    "filler_ratio": float,           # fillers / total_words
    "clarity_score": float,          # 0-1
    "communication_score": float     # 0-1  (used in final scoring)
}
"""

from __future__ import annotations

import re
from typing import Any

# ── tunables ──────────────────────────────────────────────────────────────────
PAUSE_THRESHOLD = 1.0       # seconds gap between segments → counted as pause
LONG_PAUSE_THRESHOLD = 3.0  # seconds → "long" pause (hurts score more)

IDEAL_WPM_MIN = 120
IDEAL_WPM_MAX = 160

FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "literally",
    "actually", "right", "so", "okay", "hmm", "err",
}

# Penalty per long pause on the communication score (capped at 0.3 total)
LONG_PAUSE_PENALTY = 0.05


def analyze_speech(transcript: str,
                   segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Parameters
    ----------
    transcript:
        Full transcript string (from Whisper result["text"]).
    segments:
        List of segment dicts::

            {"start": float, "end": float, "text": str, "confidence": float}

    Returns
    -------
    Speech quality metrics dict (see module docstring).
    """
    if not segments:
        return _empty_result()

    # ── Speech rate ───────────────────────────────────────────────────────────
    words = _tokenise(transcript)
    total_words = len(words)

    total_duration = segments[-1]["end"] - segments[0]["start"]
    speech_duration = sum(s["end"] - s["start"] for s in segments)
    speech_duration = max(speech_duration, 0.1)  # guard div/0

    speech_rate_wpm = round((total_words / speech_duration) * 60, 1)

    # ── Pause analysis ────────────────────────────────────────────────────────
    pauses: list[float] = []
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap >= PAUSE_THRESHOLD:
            pauses.append(gap)

    pause_count = len(pauses)
    long_pause_count = sum(1 for p in pauses if p >= LONG_PAUSE_THRESHOLD)
    avg_pause_duration = round(sum(pauses) / pause_count, 2) if pauses else 0.0

    # ── Filler words ──────────────────────────────────────────────────────────
    lower_transcript = transcript.lower()
    filler_word_count = 0
    for filler in FILLER_WORDS:
        # Word-boundary match so "like" doesn't hit "likewise"
        pattern = r"\b" + re.escape(filler) + r"\b"
        filler_word_count += len(re.findall(pattern, lower_transcript))

    filler_ratio = round(filler_word_count / max(total_words, 1), 4)

    # ── Clarity score ─────────────────────────────────────────────────────────
    # Components:
    #   1. Speech rate penalty (far from ideal band → lower score)
    #   2. Filler penalty
    #   3. Confidence average (Whisper avg_logprob mapped to 0-1)
    rate_score = _rate_score(speech_rate_wpm)
    filler_penalty = min(0.4, filler_ratio * 2)  # cap at -0.4
    avg_confidence = _avg_confidence(segments)

    clarity_score = round(
        (rate_score * 0.4)
        + (avg_confidence * 0.4)
        + ((1.0 - filler_penalty) * 0.2),
        4,
    )
    clarity_score = max(0.0, min(1.0, clarity_score))

    # ── Communication score (0-1) ─────────────────────────────────────────────
    long_pause_total_penalty = min(0.3, long_pause_count * LONG_PAUSE_PENALTY)
    communication_score = round(
        clarity_score - long_pause_total_penalty,
        4,
    )
    communication_score = max(0.0, min(1.0, communication_score))

    return {
        "speech_rate_wpm": speech_rate_wpm,
        "pause_count": pause_count,
        "avg_pause_duration": avg_pause_duration,
        "long_pause_count": long_pause_count,
        "filler_word_count": filler_word_count,
        "filler_ratio": filler_ratio,
        "clarity_score": clarity_score,
        "communication_score": communication_score,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Simple whitespace tokeniser that strips punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


def _rate_score(wpm: float) -> float:
    """Map words-per-minute to a 0-1 score. Peak at [120, 160] WPM."""
    if IDEAL_WPM_MIN <= wpm <= IDEAL_WPM_MAX:
        return 1.0
    if wpm < IDEAL_WPM_MIN:
        # Too slow: linear decay toward 0 at 40 WPM
        return max(0.0, (wpm - 40) / (IDEAL_WPM_MIN - 40))
    # Too fast: linear decay toward 0 at 240 WPM
    return max(0.0, 1.0 - (wpm - IDEAL_WPM_MAX) / (240 - IDEAL_WPM_MAX))


def _avg_confidence(segments: list[dict[str, Any]]) -> float:
    """
    Convert Whisper avg_logprob (typically -0.5 to -0.0) to 0-1 scale.
    logprob of 0 → confidence 1; logprob of -1 → confidence 0.
    """
    logprobs = [s.get("confidence", -0.3) for s in segments if "confidence" in s]
    if not logprobs:
        return 0.7  # neutral fallback

    avg_lp = sum(logprobs) / len(logprobs)
    # Clamp logprob to [-1, 0]
    clamped = max(-1.0, min(0.0, avg_lp))
    return round(1.0 + clamped, 4)  # maps -1→0, 0→1


def _empty_result() -> dict[str, Any]:
    return {
        "speech_rate_wpm": 0.0,
        "pause_count": 0,
        "avg_pause_duration": 0.0,
        "long_pause_count": 0,
        "filler_word_count": 0,
        "filler_ratio": 0.0,
        "clarity_score": 0.0,
        "communication_score": 0.0,
    }
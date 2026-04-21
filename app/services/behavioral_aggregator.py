"""
behavioral_aggregator.py
------------------------
Takes the raw outputs from all upstream services and aggregates them
into the four scored dimensions + a weighted final score.

Dimension weights (configurable via env):
    Communication   40 %
    Engagement      30 %
    Confidence      20 %
    Professionalism 10 %

Inputs (all optional – missing data degrades gracefully):
    - gaze_metrics      from gaze_service
    - motion_metrics    from motion_service
    - emotion_metrics   from emotion_service
    - speech_metrics    from speech_analysis_service

Output
------
{
    "communication": float,       # 1-5
    "engagement": float,          # 1-5
    "confidence": float,          # 1-5
    "professionalism": float,     # 1-5
    "final_score": float,         # 1-5
    "breakdown": {
        "eye_contact_ratio": float,
        "head_stability": float,
        "movement_frequency": float,
        "emotional_stability": float,
        "dominant_emotion": str,
        "speech_clarity": float,
        "speech_rate_wpm": float,
        "filler_ratio": float,
        "pause_count": int,
        "long_pause_count": int
    }
}
"""

from __future__ import annotations

import os
from typing import Any

# ── Dimension weights ─────────────────────────────────────────────────────────
W_COMMUNICATION = float(os.getenv("W_COMMUNICATION", 0.40))
W_ENGAGEMENT = float(os.getenv("W_ENGAGEMENT", 0.30))
W_CONFIDENCE = float(os.getenv("W_CONFIDENCE", 0.20))
W_PROFESSIONALISM = float(os.getenv("W_PROFESSIONALISM", 0.10))


def aggregate_behavioral_scores(
    gaze_metrics: dict[str, Any] | None = None,
    motion_metrics: dict[str, Any] | None = None,
    emotion_metrics: dict[str, Any] | None = None,
    speech_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute all four dimension scores and a final weighted score.

    All dimension sub-scores are first computed on [0, 1] and then
    mapped to [1, 5] for the output.

    Parameters
    ----------
    gaze_metrics : output of gaze_service (eye_contact_ratio, head_stability, …)
    motion_metrics : output of motion_service
    emotion_metrics : output of emotion_service
    speech_metrics : output of speech_analysis_service

    Returns
    -------
    Scored output dict (see module docstring).
    """
    gaze = gaze_metrics or {}
    motion = motion_metrics or {}
    emotion = emotion_metrics or {}
    speech = speech_metrics or {}

    # ── Raw signals (0-1) ─────────────────────────────────────────────────────

    eye_contact_ratio: float = gaze.get("eye_contact_ratio", 0.5)
    head_stability: float = gaze.get("head_stability", 0.5)

    movement_frequency: float = motion.get("movement_frequency", 0.5)
    stability_score: float = motion.get("stability_score", 0.5)

    emotional_stability: float = emotion.get("emotional_stability", 0.5)
    professionalism_raw: float = emotion.get("professionalism_score", 0.5)
    dominant_emotion: str = emotion.get("dominant_emotion", "neutral")

    communication_raw: float = speech.get("communication_score", 0.5)
    clarity: float = speech.get("clarity_score", 0.5)
    filler_ratio: float = speech.get("filler_ratio", 0.0)
    speech_rate_wpm: float = speech.get("speech_rate_wpm", 130.0)
    pause_count: int = speech.get("pause_count", 0)
    long_pause_count: int = speech.get("long_pause_count", 0)

    # ── Dimension scores [0, 1] ───────────────────────────────────────────────

    # Communication: weighted mix of Whisper-derived speech quality.
    communication_01 = _clamp(
        communication_raw * 0.6
        + clarity * 0.4
    )

    # Engagement: eye contact + attention + low distracting movement.
    # High movement_frequency can indicate restlessness → penalty.
    engagement_01 = _clamp(
        eye_contact_ratio * 0.50
        + head_stability * 0.30
        + (1.0 - movement_frequency) * 0.20
    )

    # Confidence: head pose stability + low excessive motion + emotional stability.
    confidence_01 = _clamp(
        stability_score * 0.40
        + head_stability * 0.35
        + emotional_stability * 0.25
    )

    # Professionalism: emotion-based (neutral/happy ratio) + expression stability.
    professionalism_01 = _clamp(
        professionalism_raw * 0.60
        + emotional_stability * 0.40
    )

    # ── Map 0-1 → 1-5 ─────────────────────────────────────────────────────────
    communication = _scale(communication_01)
    engagement = _scale(engagement_01)
    confidence = _scale(confidence_01)
    professionalism = _scale(professionalism_01)

    # ── Final weighted score ──────────────────────────────────────────────────
    final_score = round(
        communication * W_COMMUNICATION
        + engagement * W_ENGAGEMENT
        + confidence * W_CONFIDENCE
        + professionalism * W_PROFESSIONALISM,
        2,
    )

    return {
        "communication": round(communication, 2),
        "engagement": round(engagement, 2),
        "confidence": round(confidence, 2),
        "professionalism": round(professionalism, 2),
        "final_score": final_score,
        "breakdown": {
            "eye_contact_ratio": round(eye_contact_ratio, 4),
            "head_stability": round(head_stability, 4),
            "movement_frequency": round(movement_frequency, 4),
            "emotional_stability": round(emotional_stability, 4),
            "dominant_emotion": dominant_emotion,
            "speech_clarity": round(clarity, 4),
            "speech_rate_wpm": round(speech_rate_wpm, 1),
            "filler_ratio": round(filler_ratio, 4),
            "pause_count": pause_count,
            "long_pause_count": long_pause_count,
        },
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _scale(value_01: float) -> float:
    """Map [0, 1] linearly to [1, 5]."""
    return round(1.0 + value_01 * 4.0, 2)
"""
emotion_service.py
------------------
Detects facial emotions for each extracted frame using DeepFace and
aggregates them into a distribution + dominant-emotion summary.

DeepFace is run lazily (import inside function) so the service stays
importable even when DeepFace is not yet installed in dev environments.

Outputs (per-video aggregate)
------------------------------
{
    "emotion_distribution": {
        "happy": 0.35,
        "neutral": 0.45,
        "sad": 0.05,
        "angry": 0.02,
        "surprised": 0.08,
        "fear": 0.03,
        "disgust": 0.02
    },
    "dominant_emotion": "neutral",
    "emotional_stability": float,   # 0-1; high = consistent expression
    "professionalism_score": float, # 0-1; based on neutral/happy ratio
    "frames_analysed": int,
    "frames_with_face": int
}
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

# Emotions considered "professionally appropriate" for weighting.
PROFESSIONAL_EMOTIONS = {"neutral", "happy"}

# Minimum frames required to produce a reliable aggregate.
MIN_FRAMES = 3

# DeepFace backend – "opencv" is the fastest; swap to "retinaface" for
# higher accuracy in production.
DETECTOR_BACKEND = os.getenv("DEEPFACE_BACKEND", "opencv")


def analyze_emotions(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Run DeepFace emotion analysis on a list of frame dicts.

    Parameters
    ----------
    frames:
        List of ``{"frame_path": str, "timestamp": float}`` dicts as
        produced by ``frame_extraction_service.extract_frames``.

    Returns
    -------
    Aggregated emotion metrics dict (see module docstring).
    """
    try:
        from deepface import DeepFace  # lazy import
    except ImportError as exc:
        raise RuntimeError(
            "DeepFace is not installed. Run: pip install deepface"
        ) from exc

    emotion_counts: Counter = Counter()
    frames_with_face = 0
    all_dominant: list[str] = []

    for frame in frames:
        frame_path = frame.get("frame_path")
        if not frame_path or not os.path.exists(frame_path):
            continue

        try:
            results = DeepFace.analyze(
                img_path=frame_path,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND,
                silent=True,
            )

            # DeepFace returns a list when multiple faces are found.
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            dominant = result.get("dominant_emotion", "neutral")
            emotions: dict[str, float] = result.get("emotion", {})

            all_dominant.append(dominant)
            frames_with_face += 1

            # Accumulate raw scores (will normalise later).
            for emotion, score in emotions.items():
                emotion_counts[emotion.lower()] += score

        except Exception:
            # Frame may have no detectable face — silently skip.
            continue

    if frames_with_face < MIN_FRAMES:
        return _empty_result(len(frames), frames_with_face)

    # ── Normalise distribution ────────────────────────────────────────────────
    total_score = sum(emotion_counts.values()) or 1.0
    distribution = {
        emotion: round(score / total_score, 4)
        for emotion, score in emotion_counts.items()
    }

    # ── Dominant emotion (by frequency across frames) ─────────────────────────
    dominant_counter: Counter = Counter(all_dominant)
    dominant_emotion = dominant_counter.most_common(1)[0][0] if dominant_counter else "neutral"

    # ── Emotional stability ───────────────────────────────────────────────────
    # High stability = one emotion dominates. Measured as the proportion of
    # frames sharing the most common emotion.
    stability = round(dominant_counter.most_common(1)[0][1] / frames_with_face, 4)

    # ── Professionalism score ─────────────────────────────────────────────────
    prof_ratio = sum(
        distribution.get(e, 0.0) for e in PROFESSIONAL_EMOTIONS
    )
    professionalism_score = round(min(1.0, prof_ratio), 4)

    return {
        "emotion_distribution": distribution,
        "dominant_emotion": dominant_emotion,
        "emotional_stability": stability,
        "professionalism_score": professionalism_score,
        "frames_analysed": len(frames),
        "frames_with_face": frames_with_face,
    }


def _empty_result(total: int, with_face: int) -> dict[str, Any]:
    return {
        "emotion_distribution": {},
        "dominant_emotion": "unknown",
        "emotional_stability": 0.0,
        "professionalism_score": 0.5,  # neutral fallback
        "frames_analysed": total,
        "frames_with_face": with_face,
    }
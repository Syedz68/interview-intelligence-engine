"""
motion_service.py
-----------------
Analyzes motion patterns across video frames using optical-flow-style
landmark delta computation (no extra CV dependency beyond what is already
in the project).

Outputs
-------
{
    "movement_frequency": float,   # normalised 0-1 (1 = constant movement)
    "stability_score": float,      # normalised 0-1 (1 = perfectly stable)
    "avg_displacement": float,     # mean landmark shift per frame (px units)
    "motion_events": int,          # frames where movement exceeds threshold
    "total_frames_analysed": int
}
"""

from __future__ import annotations

import math
from typing import Any

# ── tunables ──────────────────────────────────────────────────────────────────
# Landmark indices used as a compact "skeleton" for motion detection.
# Using a subset (nose, eye corners, mouth corners, chin) is faster and robust
# enough compared to diffing all 478 MediaPipe face landmarks.
MOTION_LANDMARK_INDICES = [1, 33, 263, 61, 291, 199, 10, 152]

# Displacement threshold (in normalised landmark coords 0-1).
# Values above this are counted as a "motion event".
MOTION_THRESHOLD = 0.012


def _landmark_centroid(landmarks: list[dict[str, float]],
                       indices: list[int]) -> tuple[float, float]:
    """Return (mean_x, mean_y) for the given landmark indices."""
    xs = [landmarks[i]["x"] for i in indices if i < len(landmarks)]
    ys = [landmarks[i]["y"] for i in indices if i < len(landmarks)]
    if not xs:
        return 0.0, 0.0
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def analyze_motion(processed_frames: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute motion metrics from the list of processed frames produced by
    ``face_landmark_service.extract_face_landmarks``.

    Parameters
    ----------
    processed_frames:
        Each element must follow the shape::

            {
                "timestamp": float,
                "face_detected": bool,
                "landmarks": [{"x": ..., "y": ..., "z": ...}, ...]
            }

    Returns
    -------
    dict with keys: movement_frequency, stability_score,
                    avg_displacement, motion_events, total_frames_analysed
    """
    # Only consider frames where a face was actually detected.
    valid = [f for f in processed_frames if f.get("face_detected") and f.get("landmarks")]

    total = len(valid)
    if total < 2:
        return {
            "movement_frequency": 0.0,
            "stability_score": 1.0,
            "avg_displacement": 0.0,
            "motion_events": 0,
            "total_frames_analysed": total,
        }

    displacements: list[float] = []
    motion_events = 0

    prev_centroid = _landmark_centroid(valid[0]["landmarks"], MOTION_LANDMARK_INDICES)

    for frame in valid[1:]:
        curr_centroid = _landmark_centroid(frame["landmarks"], MOTION_LANDMARK_INDICES)
        disp = _euclidean(prev_centroid, curr_centroid)
        displacements.append(disp)

        if disp > MOTION_THRESHOLD:
            motion_events += 1

        prev_centroid = curr_centroid

    avg_disp = sum(displacements) / len(displacements)

    # movement_frequency: proportion of frame-pairs with notable motion
    movement_frequency = round(motion_events / len(displacements), 4)

    # stability_score: inverse of normalised average displacement.
    # avg_disp is in [0, ~0.2] for typical interview footage; clamp at 0.1.
    stability_score = round(max(0.0, 1.0 - (avg_disp / 0.1)), 4)
    stability_score = min(1.0, stability_score)

    return {
        "movement_frequency": movement_frequency,
        "stability_score": stability_score,
        "avg_displacement": round(avg_disp, 6),
        "motion_events": motion_events,
        "total_frames_analysed": total,
    }
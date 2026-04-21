"""
gaze_service.py
---------------
Analyses eye-contact and head-pose stability across all processed frames.

Fixes the typo bug in the original (`landmarks[RIGHT_EYE\"][\"y\"]`) and
adds `eye_contact_ratio` + `head_stability` to the returned dict, which
are consumed by the behavioral aggregator.

Outputs (aggregate, per-video)
------------------------------
{
    "eye_contact_ratio": float,      # proportion of frames with eye contact
    "head_stability": float,         # 0-1 (1 = perfectly stable)
    "avg_yaw": float,                # mean yaw angle (degrees)
    "avg_pitch": float,
    "avg_roll": float,
    "yaw_std": float,                # standard deviation of yaw
    "pitch_std": float,
    "frames_analysed": int,
    "frames_with_face": int
}

Per-frame helper `analyze_frame` is preserved for internal use.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import cv2
import numpy as np

# ── Landmark index constants ──────────────────────────────────────────────────
NOSE_TIP = 1
LEFT_EYE = 33
RIGHT_EYE = 263
CHIN = 199
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# Eye-contact: horizontal offset between nose and eye-centre (normalised coords).
EYE_CONTACT_THRESHOLD = 0.02

# Head stability: std-dev of yaw angle in degrees below which → "stable"
STABILITY_STD_THRESHOLD = 8.0   # degrees


# ── Per-frame helpers ─────────────────────────────────────────────────────────

def estimate_head_pose(landmarks: list[dict[str, float]],
                       image_shape: tuple[int, int]) -> tuple[float, float, float]:
    """Return (yaw, pitch, roll) in degrees."""
    h, w = image_shape

    def pt(idx: int) -> tuple[float, float]:
        lm = landmarks[idx]
        return lm["x"] * w, lm["y"] * h

    image_points = np.array([
        pt(NOSE_TIP),
        pt(CHIN),
        pt(LEFT_EYE),
        pt(RIGHT_EYE),
        pt(LEFT_MOUTH),
        pt(RIGHT_MOUTH),
    ], dtype="double")

    model_points = np.array([
        (0.0,    0.0,    0.0),
        (0.0,  -330.0,  -65.0),
        (-225.0, 170.0, -135.0),
        (225.0,  170.0, -135.0),
        (-150.0,-150.0, -125.0),
        (150.0, -150.0, -125.0),
    ])

    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rmat)[0]
    yaw, pitch, roll = angles[1], angles[0], angles[2]
    return float(yaw), float(pitch), float(roll)


def detect_eye_contact(landmarks: list[dict[str, float]]) -> bool:
    """
    Heuristic: if the horizontal distance between the nose tip and the
    midpoint of the two eye corners is below a threshold, the person is
    looking roughly at the camera.
    """
    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]
    nose = landmarks[NOSE_TIP]

    eye_center_x = (left_eye["x"] + right_eye["x"]) / 2
    offset = abs(nose["x"] - eye_center_x)
    return offset < EYE_CONTACT_THRESHOLD


def analyze_frame(frame: dict[str, Any],
                  landmarks: list[dict[str, float]] | None,
                  image_shape: tuple[int, int] | None = None) -> dict[str, Any]:
    """Analyse a single frame. image_shape = (height, width)."""
    if not landmarks:
        return {"eye_contact": False, "head_pose": None}

    eye_contact = detect_eye_contact(landmarks)

    pose: dict[str, float] | None = None
    if image_shape:
        yaw, pitch, roll = estimate_head_pose(landmarks, image_shape)
        pose = {"yaw": round(yaw, 2), "pitch": round(pitch, 2), "roll": round(roll, 2)}

    return {"eye_contact": eye_contact, "head_pose": pose}


# ── Aggregate over all frames ─────────────────────────────────────────────────

def analyze_gaze(processed_frames: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute gaze / head-pose aggregate metrics from the list produced by
    ``face_landmark_service.extract_face_landmarks``.

    Parameters
    ----------
    processed_frames:
        Each element: ``{"timestamp", "face_detected", "landmarks", "frame_path"}``

    Returns
    -------
    Aggregate gaze metrics dict.
    """
    valid = [f for f in processed_frames if f.get("face_detected") and f.get("landmarks")]
    total = len(processed_frames)
    n_valid = len(valid)

    if n_valid == 0:
        return _empty_result(total)

    eye_contact_frames = 0
    yaws, pitches, rolls = [], [], []

    for frame in valid:
        landmarks = frame["landmarks"]

        if detect_eye_contact(landmarks):
            eye_contact_frames += 1

        # Load image shape from disk only if needed for pose.
        frame_path = frame.get("frame_path")
        if frame_path:
            try:
                import cv2 as _cv2  # already imported above, just for clarity
                img = _cv2.imread(frame_path)
                if img is not None:
                    h, w = img.shape[:2]
                    yaw, pitch, roll = estimate_head_pose(landmarks, (h, w))
                    yaws.append(yaw)
                    pitches.append(pitch)
                    rolls.append(roll)
            except Exception:
                pass

    # Force all angle values to plain Python float before any arithmetic.
    # cv2.RQDecomp3x3 returns numpy.float64; statistics.mean/stdev preserves
    # that type, which causes pydantic_core serialization errors downstream.
    yaws   = [float(v) for v in yaws]
    pitches = [float(v) for v in pitches]
    rolls  = [float(v) for v in rolls]

    eye_contact_ratio = round(eye_contact_frames / n_valid, 4)

    avg_yaw   = round(statistics.mean(yaws),   2) if yaws   else 0.0
    avg_pitch = round(statistics.mean(pitches), 2) if pitches else 0.0
    avg_roll  = round(statistics.mean(rolls),  2) if rolls  else 0.0
    yaw_std   = round(statistics.stdev(yaws),  2) if len(yaws)   > 1 else 0.0
    pitch_std = round(statistics.stdev(pitches), 2) if len(pitches) > 1 else 0.0

    # Head stability: low variance in yaw (left/right head turning) is most
    # indicative of attentive, stable posture.
    head_stability = round(
        max(0.0, 1.0 - yaw_std / STABILITY_STD_THRESHOLD),
        4,
    )
    head_stability = min(1.0, head_stability)

    return {
        "eye_contact_ratio": float(eye_contact_ratio),
        "head_stability":    float(head_stability),
        "avg_yaw":           float(avg_yaw),
        "avg_pitch":         float(avg_pitch),
        "avg_roll":          float(avg_roll),
        "yaw_std":           float(yaw_std),
        "pitch_std":         float(pitch_std),
        "frames_analysed":   int(total),
        "frames_with_face":  int(n_valid),
    }


def _empty_result(total: int) -> dict[str, Any]:
    return {
        "eye_contact_ratio": 0.5,
        "head_stability": 0.5,
        "avg_yaw": 0.0,
        "avg_pitch": 0.0,
        "avg_roll": 0.0,
        "yaw_std": 0.0,
        "pitch_std": 0.0,
        "frames_analysed": total,
        "frames_with_face": 0,
    }
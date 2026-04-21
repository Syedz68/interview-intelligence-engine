"""
response.py
-----------
Pydantic response schemas for the behavioral analysis pipeline.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    eye_contact_ratio: float = Field(..., ge=0, le=1)
    head_stability: float = Field(..., ge=0, le=1)
    movement_frequency: float = Field(..., ge=0, le=1)
    emotional_stability: float = Field(..., ge=0, le=1)
    dominant_emotion: str
    speech_clarity: float = Field(..., ge=0, le=1)
    speech_rate_wpm: float = Field(..., ge=0)
    filler_ratio: float = Field(..., ge=0, le=1)
    pause_count: int = Field(..., ge=0)
    long_pause_count: int = Field(..., ge=0)


class BehavioralAnalysisResult(BaseModel):
    """Top-level response from POST /intelligence/analyze-interview."""

    communication: float = Field(..., ge=1, le=5, description="Speech clarity & delivery (1-5)")
    engagement: float = Field(..., ge=1, le=5, description="Eye contact & attentiveness (1-5)")
    confidence: float = Field(..., ge=1, le=5, description="Posture & stability (1-5)")
    professionalism: float = Field(..., ge=1, le=5, description="Facial expression control (1-5)")
    final_score: float = Field(..., ge=1, le=5, description="Weighted composite score (1-5)")

    breakdown: ScoreBreakdown
    transcript: str
    segments: list[dict[str, Any]]

    raw_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Full per-service raw metrics for debugging / audit trail"
    )
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Wall-clock seconds per pipeline stage"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "communication": 4.0,
                "engagement": 3.5,
                "confidence": 4.0,
                "professionalism": 3.0,
                "final_score": 3.75,
                "breakdown": {
                    "eye_contact_ratio": 0.72,
                    "head_stability": 0.85,
                    "movement_frequency": 0.15,
                    "emotional_stability": 0.78,
                    "dominant_emotion": "neutral",
                    "speech_clarity": 0.82,
                    "speech_rate_wpm": 138.0,
                    "filler_ratio": 0.04,
                    "pause_count": 3,
                    "long_pause_count": 0,
                },
                "transcript": "Hi, my name is Alex...",
                "segments": [],
            }
        }
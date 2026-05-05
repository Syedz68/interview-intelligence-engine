"""
response.py  (v3)
-----------------
Pydantic response schemas for the full Interview Intelligence pipeline.
"""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── Original (backward compat) ────────────────────────────────────────────────

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
    communication: float = Field(..., ge=1, le=5)
    engagement: float = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=1, le=5)
    professionalism: float = Field(..., ge=1, le=5)
    final_score: float = Field(..., ge=1, le=5)
    breakdown: ScoreBreakdown
    transcript: str
    segments: list[dict[str, Any]]
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)


# ── English Fluency ───────────────────────────────────────────────────────────

class GrammarError(BaseModel):
    original: str
    correction: str
    rule: str


class VocabularySuggestion(BaseModel):
    used_word: str
    better_alternative: str
    reason: str


class EnglishFluencyReport(BaseModel):
    overall_band: str = "N/A"
    overall_score: float = 0.0
    grammar_score: float = 0.0
    vocabulary_score: float = 0.0
    fluency_score: float = 0.0
    coherence_score: float = 0.0
    pronunciation_proxy: float = 0.0
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    grammar_errors: list[GrammarError] = Field(default_factory=list)
    vocabulary_suggestions: list[VocabularySuggestion] = Field(default_factory=list)
    sample_feedback: str = ""
    cefr_justification: str = ""
    ai_powered: bool = False


# ── Answer Relevance ──────────────────────────────────────────────────────────

class QAPair(BaseModel):
    question_index: int = 0
    question: str = ""
    answer: str = ""
    relevance_score: float = 0.0
    correctness_score: float = 0.0
    completeness_score: float = 0.0
    overall_score: float = 0.0
    verdict: str = "Unscored"
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    feedback: str = ""


class QAAggregate(BaseModel):
    avg_relevance: float = 0.0
    avg_correctness: float = 0.0
    avg_completeness: float = 0.0
    overall_qa_score: float = 0.0
    questions_answered: int = 0
    questions_total: int = 0
    strong_answers: int = 0
    weak_answers: int = 0
    summary: str = ""


class AnswerRelevanceReport(BaseModel):
    qa_pairs: list[QAPair] = Field(default_factory=list)
    aggregate: QAAggregate = Field(default_factory=QAAggregate)
    ai_powered: bool = False


# ── Behavioral Narrative Report ───────────────────────────────────────────────

class BehavioralDimension(BaseModel):
    score: float = 3.0
    band: str = "Adequate"
    narrative: str = ""
    evidence: list[str] = Field(default_factory=list)


class BehavioralNarrativeReport(BaseModel):
    executive_summary: str = ""
    behavioral_dimensions: dict[str, BehavioralDimension] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    development_areas: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    hiring_recommendation: str = ""
    recommendation_rationale: str = ""
    interview_followup_questions: list[str] = Field(default_factory=list)
    onboarding_suggestions: list[str] = Field(default_factory=list)
    overall_behavioral_score: float = 0.0
    report_generated: bool = False
    ai_powered: bool = False


# ── Diarization ───────────────────────────────────────────────────────────────

class DiarizationSummary(BaseModel):
    available: bool = False
    speaker_map: dict[str, str] = Field(default_factory=dict)
    turns: list[dict[str, Any]] = Field(default_factory=list)


# ── Master result ─────────────────────────────────────────────────────────────

class FullIntelligenceResult(BaseModel):
    """
    Top-level response from POST /new-intelligence/analyze-interview-v3.
    """
    # Behavioral scores (1-5)
    communication: float = Field(..., ge=1, le=5)
    engagement: float = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=1, le=5)
    professionalism: float = Field(..., ge=1, le=5)
    final_score: float = Field(..., ge=1, le=5)
    breakdown: ScoreBreakdown

    # Transcription
    transcript: str = ""
    candidate_transcript: str = ""
    segments: list[dict[str, Any]] = Field(default_factory=list)
    candidate_segments: list[dict[str, Any]] = Field(default_factory=list)

    # Diarization
    diarization: DiarizationSummary = Field(default_factory=DiarizationSummary)

    # AI analysis
    english_fluency: EnglishFluencyReport = Field(default_factory=EnglishFluencyReport)
    answer_relevance: AnswerRelevanceReport = Field(default_factory=AnswerRelevanceReport)
    behavioral_report: BehavioralNarrativeReport = Field(default_factory=BehavioralNarrativeReport)

    # Debug
    raw_metrics: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)

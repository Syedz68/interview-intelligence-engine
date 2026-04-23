"""
english_fluency_service.py
--------------------------
AI-powered English language assessment for interview candidates.

Uses Claude (claude-sonnet-4-20250514) to analyse the candidate's transcript
and produce a detailed fluency + accuracy report.

Dimensions assessed
-------------------
1. Grammar accuracy       — subject-verb agreement, tense consistency, etc.
2. Vocabulary richness    — range, precision, register appropriateness
3. Fluency                — natural flow, coherence, connective use
4. Pronunciation proxies  — detected from Whisper confidence + filler patterns
5. Coherence & cohesion   — logical structuring of ideas
6. Overall band score     — CEFR-aligned (A1 – C2) with numeric equivalent

Output schema
-------------
{
    "overall_band":         str,        # e.g. "B2"
    "overall_score":        float,      # 0-100
    "grammar_score":        float,      # 0-100
    "vocabulary_score":     float,      # 0-100
    "fluency_score":        float,      # 0-100
    "coherence_score":      float,      # 0-100
    "pronunciation_proxy":  float,      # 0-100  (inferred, not direct)
    "strengths":            list[str],
    "weaknesses":           list[str],
    "grammar_errors":       list[{
                                "original": str,
                                "correction": str,
                                "rule": str
                            }],
    "vocabulary_suggestions": list[{
                                "used_word": str,
                                "better_alternative": str,
                                "reason": str
                            }],
    "sample_feedback":      str,        # 2-3 sentence narrative summary
    "cefr_justification":   str,        # why this band was assigned
    "ai_powered":           bool        # False if API call failed
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.services.gemini_client import call_gemini, SLOT_ENGLISH_FLUENCY

logger = logging.getLogger(__name__)

MAX_TOKENS = 2000

# Truncate very long transcripts to keep prompt cost manageable
MAX_TRANSCRIPT_CHARS = 4000


def analyze_english_fluency(
    candidate_transcript: str,
    speech_metrics: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Analyse the English language quality of a candidate's interview transcript.

    Parameters
    ----------
    candidate_transcript : The candidate's speech text (ideally diarized,
                           but full transcript works too).
    speech_metrics       : Optional dict from speech_analysis_service for
                           richer context (wpm, filler_ratio, etc.).
    api_key              : Anthropic API key (falls back to ANTHROPIC_API_KEY env).

    Returns
    -------
    Fluency report dict (see module docstring).
    """
    if not candidate_transcript or not candidate_transcript.strip():
        logger.warning("Empty transcript passed to english_fluency_service")
        return _empty_result()

    if not os.getenv("GEMINI_API_KEY", "").strip():
        logger.warning("GEMINI_API_KEY not set – returning heuristic fluency report")
        return _heuristic_result(candidate_transcript, speech_metrics)

    prompt = _build_prompt(candidate_transcript, speech_metrics)

    try:
        raw = call_gemini(prompt, slot=SLOT_ENGLISH_FLUENCY, temperature=0.3, max_tokens=MAX_TOKENS)
        result = _parse_response(raw)
        result["ai_powered"] = True
        return result
    except Exception as exc:
        logger.exception("English fluency AI call failed: %s – falling back to heuristic", exc)
        return _heuristic_result(candidate_transcript, speech_metrics)


# ── prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(
    transcript: str,
    speech_metrics: dict[str, Any] | None,
) -> str:
    truncated = transcript[:MAX_TRANSCRIPT_CHARS]
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        truncated += "\n[...transcript truncated for analysis...]"

    metrics_section = ""
    if speech_metrics:
        metrics_section = f"""
Additional speech metrics from audio analysis:
- Words per minute: {speech_metrics.get('speech_rate_wpm', 'N/A')}
- Filler word ratio: {speech_metrics.get('filler_ratio', 'N/A')}
- Filler word count: {speech_metrics.get('filler_word_count', 'N/A')}
- Pause count: {speech_metrics.get('pause_count', 'N/A')}
- Long pause count (>3s): {speech_metrics.get('long_pause_count', 'N/A')}
"""

    return f"""You are an expert English language assessor evaluating a job interview candidate's spoken English.
Analyse the transcript below and provide a detailed, fair, and actionable English language assessment.
{metrics_section}
CANDIDATE TRANSCRIPT:
\"\"\"
{truncated}
\"\"\"

Provide your assessment as a valid JSON object with EXACTLY this structure (no extra keys, no markdown fences):
{{
  "overall_band": "<CEFR band: A1|A2|B1|B2|C1|C2>",
  "overall_score": <0-100 float>,
  "grammar_score": <0-100 float>,
  "vocabulary_score": <0-100 float>,
  "fluency_score": <0-100 float>,
  "coherence_score": <0-100 float>,
  "pronunciation_proxy": <0-100 float, infer from filler patterns and hesitation markers>,
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "grammar_errors": [
    {{"original": "<exact phrase from transcript>", "correction": "<corrected form>", "rule": "<grammar rule violated>"}}
  ],
  "vocabulary_suggestions": [
    {{"used_word": "<word candidate used>", "better_alternative": "<more precise/professional word>", "reason": "<why it's better>"}}
  ],
  "sample_feedback": "<2-3 sentence overall narrative feedback suitable to share with the candidate>",
  "cefr_justification": "<1-2 sentences explaining why this CEFR band was chosen>"
}}

Rules:
- grammar_errors: list up to 5 real errors found; empty list if none
- vocabulary_suggestions: list up to 5 concrete improvements; empty list if vocabulary is strong
- Be honest but constructive; this feedback will be shown to the candidate
- Score 0-100 where 100 = native-level professional English
- If the transcript is very short (<50 words), note this limitation in sample_feedback
- Respond with ONLY the JSON object, nothing else"""


# ── API call ──────────────────────────────────────────────────────────────────

# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict[str, Any]:
    # Strip any accidental markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    return json.loads(cleaned)


# ── fallbacks ─────────────────────────────────────────────────────────────────

def _heuristic_result(
    transcript: str,
    speech_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Lightweight rule-based approximation used when the API is unavailable.
    Not as accurate as Claude but still useful.
    """
    words = transcript.split()
    word_count = len(words)
    metrics = speech_metrics or {}

    filler_ratio = metrics.get("filler_ratio", 0.05)
    wpm = metrics.get("speech_rate_wpm", 130)

    # Very rough heuristics
    fluency_score = max(0.0, min(100.0, 100 - filler_ratio * 200))
    wpm_score = 100 - abs(wpm - 140) * 0.5
    wpm_score = max(0.0, min(100.0, wpm_score))

    overall = round((fluency_score * 0.4 + wpm_score * 0.3 + 65 * 0.3), 1)

    return {
        "overall_band": _score_to_cefr(overall),
        "overall_score": overall,
        "grammar_score": 65.0,
        "vocabulary_score": 65.0,
        "fluency_score": round(fluency_score, 1),
        "coherence_score": 65.0,
        "pronunciation_proxy": round(wpm_score, 1),
        "strengths": ["Transcript available for analysis"],
        "weaknesses": ["Full AI analysis unavailable – set ANTHROPIC_API_KEY for detailed feedback"],
        "grammar_errors": [],
        "vocabulary_suggestions": [],
        "sample_feedback": (
            f"Candidate produced {word_count} words at approximately {wpm:.0f} WPM. "
            "Full AI-powered English assessment requires ANTHROPIC_API_KEY to be configured."
        ),
        "cefr_justification": "Heuristic estimate only – configure AI for accurate CEFR assessment.",
        "ai_powered": False,
    }


def _empty_result() -> dict[str, Any]:
    return {
        "overall_band": "N/A",
        "overall_score": 0.0,
        "grammar_score": 0.0,
        "vocabulary_score": 0.0,
        "fluency_score": 0.0,
        "coherence_score": 0.0,
        "pronunciation_proxy": 0.0,
        "strengths": [],
        "weaknesses": ["No transcript available"],
        "grammar_errors": [],
        "vocabulary_suggestions": [],
        "sample_feedback": "No transcript was provided for analysis.",
        "cefr_justification": "N/A – no transcript.",
        "ai_powered": False,
    }


def _score_to_cefr(score: float) -> str:
    if score >= 90:
        return "C2"
    if score >= 80:
        return "C1"
    if score >= 65:
        return "B2"
    if score >= 50:
        return "B1"
    if score >= 35:
        return "A2"
    return "A1"

"""
english_fluency_service.py
--------------------------
AI-powered English language assessment for interview candidates.

Uses Ollama (local or cloud) to analyse the candidate's transcript and
produce a detailed fluency + accuracy report.

Dimensions assessed
-------------------
1. Grammar accuracy       — subject-verb agreement, tense consistency, etc.
2. Vocabulary richness    — range, precision, register appropriateness
3. Fluency                — natural flow, coherence, connective use
4. Pronunciation proxies  — inferred from Whisper confidence + filler patterns
5. Coherence & cohesion   — logical structuring of ideas
6. Overall band score     — CEFR-aligned (A1-C2) with numeric equivalent

Output schema
-------------
{
    "overall_band":           str,        # e.g. "B2"
    "overall_score":          float,      # 0-100
    "grammar_score":          float,      # 0-100
    "vocabulary_score":       float,      # 0-100
    "fluency_score":          float,      # 0-100
    "coherence_score":        float,      # 0-100
    "pronunciation_proxy":    float,      # 0-100 (inferred)
    "strengths":              list[str],
    "weaknesses":             list[str],
    "grammar_errors":         list[{"original": str, "correction": str, "rule": str}],
    "vocabulary_suggestions": list[{"used_word": str, "better_alternative": str, "reason": str}],
    "sample_feedback":        str,
    "cefr_justification":     str,
    "ai_powered":             bool
}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.ollama_client import call_ollama, is_configured, SLOT_ENGLISH_FLUENCY

logger = logging.getLogger(__name__)

MAX_TOKENS           = 1500
MAX_TRANSCRIPT_CHARS = 3000   # keep prompt lean for smaller models


def analyze_english_fluency(
    candidate_transcript: str,
    speech_metrics: dict[str, Any] | None = None,
    *,
    api_key: str | None = None,   # unused — kept for API compatibility
) -> dict[str, Any]:
    if not candidate_transcript or not candidate_transcript.strip():
        logger.warning("Empty transcript passed to english_fluency_service")
        return _empty_result()

    if not is_configured():
        logger.warning("Ollama not configured – returning heuristic fluency report")
        return _heuristic_result(candidate_transcript, speech_metrics)

    prompt = _build_prompt(candidate_transcript, speech_metrics)

    try:
        raw    = call_ollama(prompt, slot=SLOT_ENGLISH_FLUENCY, temperature=0.3, max_tokens=MAX_TOKENS)
        result = _parse_response(raw)
        result["ai_powered"] = True
        return result
    except Exception as exc:
        logger.exception("English fluency AI call failed: %s – falling back to heuristic", exc)
        return _heuristic_result(candidate_transcript, speech_metrics)


# ── prompt ─────────────────────────────────────────────────────────────────────

def _build_prompt(transcript: str, speech_metrics: dict[str, Any] | None) -> str:
    truncated = transcript[:MAX_TRANSCRIPT_CHARS]
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        truncated += "\n[transcript truncated]"

    metrics_lines: list[str] = []
    if speech_metrics:
        pairs = [
            ("WPM",         speech_metrics.get("speech_rate_wpm")),
            ("Filler ratio",speech_metrics.get("filler_ratio")),
            ("Filler count",speech_metrics.get("filler_word_count")),
            ("Pauses",      speech_metrics.get("pause_count")),
            ("Long pauses", speech_metrics.get("long_pause_count")),
        ]
        metrics_lines = [f"- {k}: {v}" for k, v in pairs if v is not None]

    metrics_block = ("\nSpeech metrics:\n" + "\n".join(metrics_lines) + "\n") if metrics_lines else ""

    return f"""You are an expert English language assessor. Evaluate the job interview transcript below.
{metrics_block}
TRANSCRIPT:
\"\"\"
{truncated}
\"\"\"

Respond with ONLY a valid JSON object — no markdown, no extra text:
{{
  "overall_band": "<A1|A2|B1|B2|C1|C2>",
  "overall_score": <0-100>,
  "grammar_score": <0-100>,
  "vocabulary_score": <0-100>,
  "fluency_score": <0-100>,
  "coherence_score": <0-100>,
  "pronunciation_proxy": <0-100>,
  "strengths": ["<up to 3 strengths>"],
  "weaknesses": ["<up to 2 weaknesses>"],
  "grammar_errors": [
    {{"original": "<phrase>", "correction": "<fix>", "rule": "<rule>"}}
  ],
  "vocabulary_suggestions": [
    {{"used_word": "<word>", "better_alternative": "<word>", "reason": "<why>"}}
  ],
  "sample_feedback": "<2-3 sentence constructive summary>",
  "cefr_justification": "<1-2 sentences explaining the band>"
}}

Rules: grammar_errors up to 5 (empty list if none); vocabulary_suggestions up to 5; 100=native professional level; output ONLY the JSON."""

# ── parsing ────────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    return json.loads(cleaned)

# ── fallbacks ──────────────────────────────────────────────────────────────────

def _heuristic_result(transcript: str, speech_metrics: dict[str, Any] | None) -> dict[str, Any]:
    words      = transcript.split()
    word_count = len(words)
    metrics    = speech_metrics or {}
    filler_ratio = metrics.get("filler_ratio", 0.05)
    wpm          = metrics.get("speech_rate_wpm", 130)
    fluency_score = max(0.0, min(100.0, 100 - filler_ratio * 200))
    wpm_score     = max(0.0, min(100.0, 100 - abs(wpm - 140) * 0.5))
    overall       = round(fluency_score * 0.4 + wpm_score * 0.3 + 65 * 0.3, 1)
    return {
        "overall_band":           _score_to_cefr(overall),
        "overall_score":          overall,
        "grammar_score":          65.0,
        "vocabulary_score":       65.0,
        "fluency_score":          round(fluency_score, 1),
        "coherence_score":        65.0,
        "pronunciation_proxy":    round(wpm_score, 1),
        "strengths":              ["Transcript available for analysis"],
        "weaknesses":             ["Full AI analysis unavailable — configure Ollama for detailed feedback"],
        "grammar_errors":         [],
        "vocabulary_suggestions": [],
        "sample_feedback":        (
            f"Candidate produced {word_count} words at approximately {wpm:.0f} WPM. "
            "Full AI-powered assessment requires Ollama to be configured."
        ),
        "cefr_justification":     "Heuristic estimate only — configure Ollama for accurate CEFR assessment.",
        "ai_powered":             False,
    }

def _empty_result() -> dict[str, Any]:
    return {
        "overall_band":           "N/A",
        "overall_score":          0.0,
        "grammar_score":          0.0,
        "vocabulary_score":       0.0,
        "fluency_score":          0.0,
        "coherence_score":        0.0,
        "pronunciation_proxy":    0.0,
        "strengths":              [],
        "weaknesses":             ["No transcript available"],
        "grammar_errors":         [],
        "vocabulary_suggestions": [],
        "sample_feedback":        "No transcript was provided for analysis.",
        "cefr_justification":     "N/A – no transcript.",
        "ai_powered":             False,
    }

def _score_to_cefr(score: float) -> str:
    if score >= 90: return "C2"
    if score >= 80: return "C1"
    if score >= 65: return "B2"
    if score >= 50: return "B1"
    if score >= 35: return "A2"
    return "A1"

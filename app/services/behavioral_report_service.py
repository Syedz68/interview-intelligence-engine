"""
behavioral_report_service.py
-----------------------------
AI-powered behavioral analysis report generator.

Takes all structured scores (from video CV pipeline + speech analysis +
fluency + Q&A) and uses Claude to produce a rich, human-readable behavioral
assessment report with actionable hiring recommendations.

Output schema
-------------
{
    "executive_summary":    str,     # 3-4 sentence high-level summary for hiring manager
    "behavioral_dimensions": {
        "communication":     {"score": float, "band": str, "narrative": str, "evidence": list[str]},
        "engagement":        {"score": float, "band": str, "narrative": str, "evidence": list[str]},
        "confidence":        {"score": float, "band": str, "narrative": str, "evidence": list[str]},
        "professionalism":   {"score": float, "band": str, "narrative": str, "evidence": list[str]},
    },
    "strengths":             list[str],   # top 3-5 behavioral strengths
    "development_areas":     list[str],   # top 2-4 areas for growth
    "red_flags":             list[str],   # concerning signals (may be empty)
    "hiring_recommendation": str,         # "Strong Hire"|"Hire"|"Maybe"|"No Hire"
    "recommendation_rationale": str,      # 2-3 sentence justification
    "interview_followup_questions": list[str],  # 3 targeted follow-up questions
    "onboarding_suggestions": list[str],  # if hired, what support might help
    "overall_behavioral_score": float,   # 0-100
    "report_generated":      bool,
    "ai_powered":            bool
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.services.gemini_client import call_gemini, SLOT_BEHAVIORAL_REPORT

logger = logging.getLogger(__name__)

MAX_TOKENS = 2500

# Score band labels (maps 1-5 scale from behavioral aggregator)
SCORE_BANDS = {
    (4.5, 5.0): "Outstanding",
    (3.5, 4.5): "Strong",
    (2.5, 3.5): "Adequate",
    (1.5, 2.5): "Developing",
    (1.0, 1.5): "Needs Improvement",
}


def generate_behavioral_report(
    behavioral_scores: dict[str, Any],
    speech_metrics: dict[str, Any] | None = None,
    fluency_report: dict[str, Any] | None = None,
    qa_report: dict[str, Any] | None = None,
    *,
    job_role: str | None = None,
    candidate_name: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Generate a comprehensive behavioral report using Claude AI.

    Parameters
    ----------
    behavioral_scores : output from behavioral_aggregator (communication 1-5, etc.)
    speech_metrics    : output from speech_analysis_service
    fluency_report    : output from english_fluency_service
    qa_report         : output from answer_relevance_service
    job_role          : target role for context-aware recommendations
    candidate_name    : optional candidate name for personalised report
    api_key           : Anthropic API key (env fallback)

    Returns
    -------
    Behavioral report dict (see module docstring).
    """
    resolved_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not resolved_key:
        logger.warning("GEMINI_API_KEY not set – generating structured-data-only report")
        return _structured_fallback(behavioral_scores, speech_metrics, fluency_report, qa_report)

    try:
        prompt = _build_prompt(
            behavioral_scores, speech_metrics, fluency_report, qa_report,
            job_role=job_role, candidate_name=candidate_name,
        )
        raw = _call_gemini(prompt)
        result = _parse_response(raw)
        result["report_generated"] = True
        result["ai_powered"] = True
        return result
    except Exception as exc:
        logger.exception("Behavioral report generation failed: %s", exc)
        return _structured_fallback(behavioral_scores, speech_metrics, fluency_report, qa_report)


# ── prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(
    behavioral_scores: dict[str, Any],
    speech_metrics: dict[str, Any] | None,
    fluency_report: dict[str, Any] | None,
    qa_report: dict[str, Any] | None,
    job_role: str | None,
    candidate_name: str | None,
) -> str:
    speech = speech_metrics or {}
    fluency = fluency_report or {}
    qa = qa_report or {}
    scores = behavioral_scores or {}
    breakdown = scores.get("breakdown", {})

    role_context = f"Target role: {job_role}" if job_role else "Role: Not specified"
    name_context = f"Candidate: {candidate_name}" if candidate_name else "Candidate: Anonymous"

    qa_summary = qa.get("aggregate", {}).get("summary", "No Q&A data available.")
    qa_score = qa.get("aggregate", {}).get("overall_qa_score", "N/A")
    fluency_band = fluency.get("overall_band", "N/A")
    fluency_score = fluency.get("overall_score", "N/A")
    fluency_feedback = fluency.get("sample_feedback", "")
    fluency_strengths = fluency.get("strengths", [])
    fluency_weaknesses = fluency.get("weaknesses", [])

    prompt = f"""You are a senior HR psychologist and talent assessment expert writing a professional behavioral interview report.

Context:
- {name_context}
- {role_context}

=== QUANTITATIVE SCORES ===

BEHAVIORAL DIMENSIONS (scale 1-5):
- Communication:    {scores.get('communication', 'N/A')} / 5
- Engagement:       {scores.get('engagement', 'N/A')} / 5
- Confidence:       {scores.get('confidence', 'N/A')} / 5
- Professionalism:  {scores.get('professionalism', 'N/A')} / 5
- Composite Score:  {scores.get('final_score', 'N/A')} / 5

VISUAL BEHAVIORAL SIGNALS:
- Eye contact ratio:    {breakdown.get('eye_contact_ratio', 'N/A')} (1.0 = consistent)
- Head stability:       {breakdown.get('head_stability', 'N/A')} (1.0 = stable)
- Movement frequency:   {breakdown.get('movement_frequency', 'N/A')} (lower = calmer)
- Emotional stability:  {breakdown.get('emotional_stability', 'N/A')}
- Dominant emotion:     {breakdown.get('dominant_emotion', 'neutral')}

SPEECH METRICS:
- Words per minute:     {speech.get('speech_rate_wpm', 'N/A')}
- Filler words:         {speech.get('filler_word_count', 'N/A')} ({speech.get('filler_ratio', 'N/A')} ratio)
- Pauses:               {speech.get('pause_count', 'N/A')} total, {speech.get('long_pause_count', 'N/A')} long (>3s)
- Clarity score:        {speech.get('clarity_score', 'N/A')}

ENGLISH FLUENCY:
- CEFR Band:            {fluency_band}
- Fluency Score:        {fluency_score} / 100
- AI Feedback:          {fluency_feedback}
- Strengths:            {', '.join(fluency_strengths) if fluency_strengths else 'N/A'}
- Development areas:    {', '.join(fluency_weaknesses) if fluency_weaknesses else 'N/A'}

ANSWER QUALITY:
- Q&A Overall Score:    {qa_score} / 100
- Q&A Summary:          {qa_summary}

=== YOUR TASK ===

Write a comprehensive behavioral report. Return a valid JSON object with EXACTLY this structure (no markdown fences):
{{
  "executive_summary": "<3-4 sentence high-level summary for hiring manager>",
  "behavioral_dimensions": {{
    "communication": {{
      "score": {scores.get('communication', 3.0)},
      "band": "<Outstanding|Strong|Adequate|Developing|Needs Improvement>",
      "narrative": "<2-3 sentence evidence-based narrative>",
      "evidence": ["<specific observable evidence 1>", "<evidence 2>"]
    }},
    "engagement": {{
      "score": {scores.get('engagement', 3.0)},
      "band": "<band>",
      "narrative": "<narrative>",
      "evidence": ["<evidence 1>", "<evidence 2>"]
    }},
    "confidence": {{
      "score": {scores.get('confidence', 3.0)},
      "band": "<band>",
      "narrative": "<narrative>",
      "evidence": ["<evidence 1>", "<evidence 2>"]
    }},
    "professionalism": {{
      "score": {scores.get('professionalism', 3.0)},
      "band": "<band>",
      "narrative": "<narrative>",
      "evidence": ["<evidence 1>", "<evidence 2>"]
    }}
  }},
  "strengths": ["<behavioral strength 1>", "<strength 2>", "<strength 3>"],
  "development_areas": ["<area 1>", "<area 2>"],
  "red_flags": ["<any concerning signal, or empty list if none>"],
  "hiring_recommendation": "<Strong Hire|Hire|Maybe|No Hire>",
  "recommendation_rationale": "<2-3 sentence justification for the recommendation>",
  "interview_followup_questions": [
    "<targeted follow-up question 1>",
    "<targeted follow-up question 2>",
    "<targeted follow-up question 3>"
  ],
  "onboarding_suggestions": ["<suggestion if hired 1>", "<suggestion 2>"],
  "overall_behavioral_score": <0-100 float synthesising all dimensions>
}}

Rules:
- Be evidence-based: tie every claim to a specific metric
- Be constructive and professional: this is a formal HR document
- Hiring recommendation must align with the overall_behavioral_score:
    Strong Hire >= 80, Hire >= 65, Maybe >= 45, No Hire < 45
- Respond with ONLY the JSON object
- Output MUST be strictly valid JSON. Do not include trailing commas."""

    return prompt


# ── API call ──────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    return call_gemini(
        prompt,
        slot=SLOT_BEHAVIORAL_REPORT,
        temperature=0.4,   # slightly higher for richer writing
        max_tokens=MAX_TOKENS,
        timeout=90,
    )


# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Invalid JSON from Gemini:\n%s", cleaned)
        raise


# ── fallback ──────────────────────────────────────────────────────────────────

def _structured_fallback(
    behavioral_scores: dict[str, Any],
    speech_metrics: dict[str, Any] | None,
    fluency_report: dict[str, Any] | None,
    qa_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a minimally structured report from raw numbers, no AI narrative."""
    scores = behavioral_scores or {}
    breakdown = scores.get("breakdown", {})
    final = scores.get("final_score", 3.0)

    # Map 1-5 to 0-100
    overall_100 = round((final - 1) / 4 * 100, 1)

    def band(score_1_5: float) -> str:
        for (lo, hi), label in SCORE_BANDS.items():
            if lo <= score_1_5 <= hi:
                return label
        return "Adequate"

    def dim(key: str) -> dict[str, Any]:
        s = scores.get(key, 3.0)
        return {
            "score": s,
            "band": band(s),
            "narrative": f"{key.capitalize()} score: {s}/5. Configure ANTHROPIC_API_KEY for detailed narrative.",
            "evidence": [],
        }

    rec_map = [
        (80, "Strong Hire"),
        (65, "Hire"),
        (45, "Maybe"),
        (0,  "No Hire"),
    ]
    recommendation = next(r for threshold, r in rec_map if overall_100 >= threshold)

    return {
        "executive_summary": (
            f"Candidate scored {final}/5 overall. "
            "Configure ANTHROPIC_API_KEY for a full AI-powered behavioral narrative."
        ),
        "behavioral_dimensions": {
            "communication":  dim("communication"),
            "engagement":     dim("engagement"),
            "confidence":     dim("confidence"),
            "professionalism": dim("professionalism"),
        },
        "strengths": [],
        "development_areas": [],
        "red_flags": [],
        "hiring_recommendation": recommendation,
        "recommendation_rationale": (
            f"Based on composite behavioral score of {final}/5 "
            f"({overall_100}/100). Full AI rationale unavailable."
        ),
        "interview_followup_questions": [],
        "onboarding_suggestions": [],
        "overall_behavioral_score": overall_100,
        "report_generated": True,
        "ai_powered": False,
    }

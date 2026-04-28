"""
behavioral_report_service.py
-----------------------------
AI-powered behavioral analysis report generator.

Takes all structured scores (video CV pipeline + speech analysis +
fluency + Q&A) and uses Ollama to produce a rich, human-readable
behavioral assessment report with actionable hiring recommendations.

Output schema
-------------
{
    "executive_summary":         str,
    "behavioral_dimensions": {
        "communication":  {"score": float, "band": str, "narrative": str, "evidence": list[str]},
        "engagement":     {"score": float, "band": str, "narrative": str, "evidence": list[str]},
        "confidence":     {"score": float, "band": str, "narrative": str, "evidence": list[str]},
        "professionalism":{"score": float, "band": str, "narrative": str, "evidence": list[str]},
    },
    "strengths":                  list[str],
    "development_areas":          list[str],
    "red_flags":                  list[str],
    "hiring_recommendation":      str,   # Strong Hire|Hire|Maybe|No Hire
    "recommendation_rationale":   str,
    "interview_followup_questions": list[str],
    "onboarding_suggestions":     list[str],
    "overall_behavioral_score":   float,
    "report_generated":           bool,
    "ai_powered":                 bool
}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.ollama_client import call_ollama, is_configured, SLOT_BEHAVIORAL_REPORT

logger = logging.getLogger(__name__)

MAX_TOKENS = 2000

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
    api_key: str | None = None,   # kept for API compat, unused
) -> dict[str, Any]:
    if not is_configured():
        logger.warning("Ollama not configured – generating structured-data-only report")
        return _structured_fallback(behavioral_scores, speech_metrics, fluency_report, qa_report)

    try:
        prompt = _build_prompt(
            behavioral_scores, speech_metrics, fluency_report, qa_report,
            job_role=job_role, candidate_name=candidate_name,
        )
        raw    = call_ollama(prompt, slot=SLOT_BEHAVIORAL_REPORT, temperature=0.4,
                             max_tokens=MAX_TOKENS, timeout=90)
        result = _parse_response(raw)
        result["report_generated"] = True
        result["ai_powered"]       = True
        return result
    except Exception as exc:
        logger.exception("Behavioral report generation failed: %s", exc)
        return _structured_fallback(behavioral_scores, speech_metrics, fluency_report, qa_report)


# ── prompt ─────────────────────────────────────────────────────────────────────

def _build_prompt(
    behavioral_scores: dict[str, Any],
    speech_metrics: dict[str, Any] | None,
    fluency_report: dict[str, Any] | None,
    qa_report: dict[str, Any] | None,
    job_role: str | None,
    candidate_name: str | None,
) -> str:
    scores    = behavioral_scores or {}
    breakdown = scores.get("breakdown", {})
    speech    = speech_metrics or {}
    fluency   = fluency_report or {}
    qa        = qa_report or {}

    # Gather only non-None values to keep the prompt tight
    def val(d: dict, key: str, default: str = "N/A") -> str:
        v = d.get(key)
        return str(v) if v is not None else default

    ctx = []
    if candidate_name: ctx.append(f"Candidate: {candidate_name}")
    if job_role:        ctx.append(f"Role: {job_role}")
    context_block = "\n".join(ctx) or "No context provided."

    fluency_strengths  = ", ".join(fluency.get("strengths", [])) or "N/A"
    fluency_weaknesses = ", ".join(fluency.get("weaknesses", [])) or "N/A"

    return f"""You are a senior HR psychologist writing a professional behavioral interview report.

{context_block}

SCORES (1-5 scale unless noted):
Behavioral: communication={val(scores,'communication')} engagement={val(scores,'engagement')} confidence={val(scores,'confidence')} professionalism={val(scores,'professionalism')} composite={val(scores,'final_score')}
Visual signals: eye_contact={val(breakdown,'eye_contact_ratio')} head_stability={val(breakdown,'head_stability')} movement_freq={val(breakdown,'movement_frequency')} dominant_emotion={val(breakdown,'dominant_emotion','neutral')}
Speech: wpm={val(speech,'speech_rate_wpm')} fillers={val(speech,'filler_word_count')}({val(speech,'filler_ratio')}) pauses={val(speech,'pause_count')} long_pauses={val(speech,'long_pause_count')} clarity={val(speech,'clarity_score')}
English: CEFR={val(fluency,'overall_band')} score={val(fluency,'overall_score')}/100 strengths={fluency_strengths} weaknesses={fluency_weaknesses}
Q&A: overall={val(qa.get('aggregate',{}),'overall_qa_score')}/100 summary={qa.get('aggregate',{}).get('summary','N/A')}

Write a comprehensive behavioral report. Return ONLY valid JSON:
{{
  "executive_summary": "<3-4 sentence summary for hiring manager>",
  "behavioral_dimensions": {{
    "communication":  {{"score": {scores.get('communication', 3.0)}, "band": "<Outstanding|Strong|Adequate|Developing|Needs Improvement>", "narrative": "<2-3 sentences>", "evidence": ["<evidence 1>", "<evidence 2>"]}},
    "engagement":     {{"score": {scores.get('engagement', 3.0)}, "band": "<band>", "narrative": "<narrative>", "evidence": ["<evidence>"]}},
    "confidence":     {{"score": {scores.get('confidence', 3.0)}, "band": "<band>", "narrative": "<narrative>", "evidence": ["<evidence>"]}},
    "professionalism":{{"score": {scores.get('professionalism', 3.0)}, "band": "<band>", "narrative": "<narrative>", "evidence": ["<evidence>"]}}
  }},
  "strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "development_areas": ["<area 1>", "<area 2>"],
  "red_flags": [],
  "hiring_recommendation": "<Strong Hire|Hire|Maybe|No Hire>",
  "recommendation_rationale": "<2-3 sentence justification>",
  "interview_followup_questions": ["<question 1>", "<question 2>", "<question 3>"],
  "onboarding_suggestions": ["<suggestion 1>", "<suggestion 2>"],
  "overall_behavioral_score": <0-100 float>
}}

Rules: tie every claim to a metric; Strong Hire>=80, Hire>=65, Maybe>=45, No Hire<45; output ONLY the JSON."""


# ── parsing ────────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Invalid JSON from Ollama:\n%s", cleaned)
        raise


# ── fallback ───────────────────────────────────────────────────────────────────

def _structured_fallback(
    behavioral_scores: dict[str, Any],
    speech_metrics: dict[str, Any] | None,
    fluency_report: dict[str, Any] | None,
    qa_report: dict[str, Any] | None,
) -> dict[str, Any]:
    scores       = behavioral_scores or {}
    final        = scores.get("final_score", 3.0)
    overall_100  = round((final - 1) / 4 * 100, 1)

    def band(s: float) -> str:
        for (lo, hi), label in SCORE_BANDS.items():
            if lo <= s <= hi:
                return label
        return "Adequate"

    def dim(key: str) -> dict[str, Any]:
        s = scores.get(key, 3.0)
        return {
            "score":     s,
            "band":      band(s),
            "narrative": f"{key.capitalize()} score: {s}/5. Configure Ollama for detailed narrative.",
            "evidence":  [],
        }

    rec_map = [(80, "Strong Hire"), (65, "Hire"), (45, "Maybe"), (0, "No Hire")]
    recommendation = next(r for threshold, r in rec_map if overall_100 >= threshold)

    return {
        "executive_summary":              (
            f"Candidate scored {final}/5 overall. "
            "Configure Ollama for a full AI-powered behavioral narrative."
        ),
        "behavioral_dimensions": {
            "communication":  dim("communication"),
            "engagement":     dim("engagement"),
            "confidence":     dim("confidence"),
            "professionalism":dim("professionalism"),
        },
        "strengths":                      [],
        "development_areas":              [],
        "red_flags":                      [],
        "hiring_recommendation":          recommendation,
        "recommendation_rationale":       (
            f"Based on composite behavioral score of {final}/5 ({overall_100}/100). "
            "Full AI rationale unavailable."
        ),
        "interview_followup_questions":   [],
        "onboarding_suggestions":         [],
        "overall_behavioral_score":       overall_100,
        "report_generated":               True,
        "ai_powered":                     False,
    }

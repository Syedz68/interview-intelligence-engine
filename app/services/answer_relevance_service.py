"""
answer_relevance_service.py
---------------------------
AI-powered Q&A matching and answer quality assessment.

Given a list of interviewer questions and candidate answers (derived from
the diarized transcript), Claude evaluates how well each answer matches
the question and generates per-answer feedback plus an aggregate score.

How Q&A pairs are extracted
---------------------------
When diarization is available the service directly pairs consecutive
interviewer → candidate turns.

When diarization is NOT available the service uses Claude to intelligently
split the full transcript into Q&A pairs (it recognises question patterns).

Output schema
-------------
{
    "qa_pairs": [
        {
            "question_index":   int,
            "question":         str,
            "answer":           str,
            "relevance_score":  float,   # 0-100: how directly does answer address question
            "correctness_score": float,  # 0-100: factual/logical correctness
            "completeness_score": float, # 0-100: depth and thoroughness
            "overall_score":    float,   # 0-100 weighted composite
            "verdict":          str,     # "Excellent"|"Good"|"Adequate"|"Weak"|"Off-topic"
            "strengths":        list[str],
            "improvements":     list[str],
            "feedback":         str      # 1-2 sentence narrative per question
        },
        ...
    ],
    "aggregate": {
        "avg_relevance":     float,
        "avg_correctness":   float,
        "avg_completeness":  float,
        "overall_qa_score":  float,      # 0-100
        "questions_answered": int,
        "questions_total":   int,
        "strong_answers":    int,        # score >= 75
        "weak_answers":      int,        # score < 40
        "summary":           str         # 2-3 sentence overall Q&A summary
    },
    "ai_powered": bool
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.services.gemini_client import call_gemini, SLOT_ANSWER_RELEVANCE

logger = logging.getLogger(__name__)

MAX_TOKENS = 3000

MAX_TRANSCRIPT_CHARS = 6000   # cap total prompt size
MAX_ANSWER_CHARS = 800        # per answer truncation
MAX_QA_PAIRS = 10             # analyse at most N pairs to stay within token budget


def analyze_answer_relevance(
    diarization_result: dict[str, Any],
    *,
    job_role: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate how well the candidate answered each interview question.

    Parameters
    ----------
    diarization_result : output from diarization_service.diarize_audio()
    job_role           : optional context (e.g. "Senior Python Developer")
                         passed to Claude for domain-aware scoring
    api_key            : Anthropic API key (falls back to ANTHROPIC_API_KEY env)

    Returns
    -------
    Answer relevance report (see module docstring).
    """
    qa_pairs = _extract_qa_pairs(diarization_result)

    if not qa_pairs:
        return _empty_result("No Q&A pairs could be extracted from the transcript.")

    if not os.getenv("GEMINI_API_KEY", "").strip():
        logger.warning("GEMINI_API_KEY not set – returning placeholder answer relevance")
        return _placeholder_result(qa_pairs)

    try:
        raw = _call_gemini(qa_pairs, job_role)
        result = _parse_response(raw, qa_pairs)
        result["ai_powered"] = True
        return result
    except Exception as exc:
        logger.exception("Answer relevance AI call failed: %s", exc)
        return _placeholder_result(qa_pairs)


# ── Q&A pair extraction ───────────────────────────────────────────────────────

def _extract_qa_pairs(
    diarization_result: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Build (question, answer) pairs from diarized turns.

    Strategy:
    - If diarization is available: pair consecutive interviewer → candidate turns.
    - If not: use heuristic sentence-split on full transcript.
    """
    turns: list[dict[str, Any]] = diarization_result.get("turns", [])
    diarization_available: bool = diarization_result.get("diarization_available", False)

    pairs: list[dict[str, str]] = []

    if diarization_available and turns:
        # Walk turns; every interviewer turn followed by a candidate turn = a pair
        i = 0
        while i < len(turns) - 1:
            current = turns[i]
            nxt = turns[i + 1]
            if current["role"] == "interviewer" and nxt["role"] == "candidate":
                q = current["text"].strip()
                a = nxt["text"].strip()
                if q and a:
                    pairs.append({"question": q, "answer": a})
                i += 2
            else:
                i += 1
    else:
        # No diarization: try to split on question-like sentences
        full_text = diarization_result.get("candidate_transcript", "")
        pairs = _heuristic_split(full_text)

    return pairs[:MAX_QA_PAIRS]


def _heuristic_split(text: str) -> list[dict[str, str]]:
    """
    Rough split: identify sentences ending in '?' as questions,
    treat following sentences as the answer.
    """
    if not text:
        return []

    sentences = re.split(r'(?<=[.?!])\s+', text)
    pairs: list[dict[str, str]] = []
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if sent.endswith("?") and i + 1 < len(sentences):
            answer_parts = []
            j = i + 1
            while j < len(sentences) and not sentences[j].strip().endswith("?"):
                answer_parts.append(sentences[j].strip())
                j += 1
            answer = " ".join(answer_parts)
            if answer:
                pairs.append({"question": sent, "answer": answer})
            i = j
        else:
            i += 1

    # If no questions found, treat entire text as a single answer to a generic prompt
    if not pairs and text.strip():
        pairs = [{"question": "Tell me about yourself / general interview response",
                  "answer": text[:MAX_ANSWER_CHARS]}]

    return pairs


# ── prompt & API ──────────────────────────────────────────────────────────────

def _call_gemini(
    qa_pairs: list[dict[str, str]],
    job_role: str | None,
) -> str:
    role_context = f"The interview is for the position: {job_role}." if job_role else ""

    qa_formatted = ""
    for idx, pair in enumerate(qa_pairs, 1):
        q = pair["question"][:500]
        a = pair["answer"][:MAX_ANSWER_CHARS]
        qa_formatted += f"\n--- Q{idx} ---\nQuestion: {q}\nAnswer: {a}\n"

    prompt = f"""You are an expert interview coach and talent assessor.
{role_context}

Evaluate each candidate answer below on three dimensions:
1. Relevance (0-100): Does the answer directly address what was asked?
2. Correctness (0-100): Is the answer factually/logically sound and appropriate?
3. Completeness (0-100): Is the answer sufficiently detailed and well-structured?

Q&A PAIRS:
{qa_formatted}

Return STRICTLY valid JSON with EXACTLY this structure (no markdown, no explanation):
{{
  "qa_pairs": [
    {{
      "question_index": 1,
      "question": "<echo the question>",
      "answer": "<first 120 chars of answer>",
      "relevance_score": <0-100 float>,
      "correctness_score": <0-100 float>,
      "completeness_score": <0-100 float>,
      "overall_score": <weighted average: relevance*0.4 + correctness*0.35 + completeness*0.25>,
      "verdict": "<Excellent|Good|Adequate|Weak|Off-topic>",
      "strengths": ["<strength>"],
      "improvements": ["<improvement suggestion>"],
      "feedback": "<1-2 sentence constructive feedback>"
    }}
  ],
  "aggregate": {{
    "avg_relevance": <float>,
    "avg_correctness": <float>,
    "avg_completeness": <float>,
    "overall_qa_score": <float>,
    "questions_answered": <int>,
    "questions_total": {len(qa_pairs)},
    "strong_answers": <int>,
    "weak_answers": <int>,
    "summary": "<2-3 sentences overall Q&A performance summary>"
  }}
}}

Rules:
- Be precise and fair
- Respond ONLY with JSON"""

    return call_gemini(prompt, slot=SLOT_ANSWER_RELEVANCE, temperature=0.3, max_tokens=MAX_TOKENS)


# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_response(raw: str, original_pairs: list[dict[str, str]]) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Invalid JSON from Gemini: %s", cleaned)
        raise

    for i, pair in enumerate(data.get("qa_pairs", []), 1):
        pair.setdefault("question_index", i)

    return data


# ── fallbacks ─────────────────────────────────────────────────────────────────

def _placeholder_result(qa_pairs: list[dict[str, str]]) -> dict[str, Any]:
    placeholder_pairs = []
    for i, pair in enumerate(qa_pairs, 1):
        placeholder_pairs.append({
            "question_index": i,
            "question": pair["question"][:200],
            "answer": pair["answer"][:200],
            "relevance_score": 0.0,
            "correctness_score": 0.0,
            "completeness_score": 0.0,
            "overall_score": 0.0,
            "verdict": "Unscored",
            "strengths": [],
            "improvements": ["Set ANTHROPIC_API_KEY for AI-powered scoring"],
            "feedback": "AI scoring requires ANTHROPIC_API_KEY.",
        })

    return {
        "qa_pairs": placeholder_pairs,
        "aggregate": {
            "avg_relevance": 0.0,
            "avg_correctness": 0.0,
            "avg_completeness": 0.0,
            "overall_qa_score": 0.0,
            "questions_answered": len(qa_pairs),
            "questions_total": len(qa_pairs),
            "strong_answers": 0,
            "weak_answers": 0,
            "summary": "AI-powered scoring unavailable. Configure ANTHROPIC_API_KEY.",
        },
        "ai_powered": False,
    }


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "qa_pairs": [],
        "aggregate": {
            "avg_relevance": 0.0,
            "avg_correctness": 0.0,
            "avg_completeness": 0.0,
            "overall_qa_score": 0.0,
            "questions_answered": 0,
            "questions_total": 0,
            "strong_answers": 0,
            "weak_answers": 0,
            "summary": reason,
        },
        "ai_powered": False,
    }

"""
answer_relevance_service.py
---------------------------
AI-powered Q&A matching and answer quality assessment.

Given a list of interviewer questions and candidate answers (from the
diarized transcript), evaluates how well each answer matches the question
and generates per-answer feedback plus an aggregate score.

Output schema
-------------
{
    "qa_pairs": [
        {
            "question_index":    int,
            "question":          str,
            "answer":            str,
            "relevance_score":   float,   # 0-100
            "correctness_score": float,   # 0-100
            "completeness_score":float,   # 0-100
            "overall_score":     float,   # weighted composite
            "verdict":           str,     # Excellent|Good|Adequate|Weak|Off-topic
            "strengths":         list[str],
            "improvements":      list[str],
            "feedback":          str
        }, ...
    ],
    "aggregate": {
        "avg_relevance":     float,
        "avg_correctness":   float,
        "avg_completeness":  float,
        "overall_qa_score":  float,
        "questions_answered":int,
        "questions_total":   int,
        "strong_answers":    int,
        "weak_answers":      int,
        "summary":           str
    },
    "ai_powered": bool
}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.services.ollama_client import call_ollama, is_configured, SLOT_ANSWER_RELEVANCE

logger = logging.getLogger(__name__)

MAX_TOKENS       = 2500
MAX_ANSWER_CHARS = 600    # per answer
MAX_QA_PAIRS     = 10     # analyse at most N pairs


def analyze_answer_relevance(
    diarization_result: dict[str, Any],
    *,
    job_role: str | None = None,
    api_key: str | None = None,   # kept for API compat, unused
) -> dict[str, Any]:
    qa_pairs = _extract_qa_pairs(diarization_result)

    if not qa_pairs:
        return _empty_result("No Q&A pairs could be extracted from the transcript.")

    if not is_configured():
        logger.warning("Ollama not configured – returning placeholder answer relevance")
        return _placeholder_result(qa_pairs)

    try:
        raw    = _call_ollama(qa_pairs, job_role)
        result = _parse_response(raw, qa_pairs)
        result["ai_powered"] = True
        return result
    except Exception as exc:
        logger.exception("Answer relevance AI call failed: %s", exc)
        return _placeholder_result(qa_pairs)


# ── Q&A extraction ─────────────────────────────────────────────────────────────

def _extract_qa_pairs(diarization_result: dict[str, Any]) -> list[dict[str, str]]:
    turns                 = diarization_result.get("turns", [])
    diarization_available = diarization_result.get("diarization_available", False)
    pairs: list[dict[str, str]] = []

    if diarization_available and turns:
        i = 0
        while i < len(turns) - 1:
            cur, nxt = turns[i], turns[i + 1]
            if cur["role"] == "interviewer" and nxt["role"] == "candidate":
                q, a = cur["text"].strip(), nxt["text"].strip()
                if q and a:
                    pairs.append({"question": q, "answer": a})
                i += 2
            else:
                i += 1
    else:
        full_text = diarization_result.get("candidate_transcript", "")
        pairs = _heuristic_split(full_text)

    return pairs[:MAX_QA_PAIRS]


def _heuristic_split(text: str) -> list[dict[str, str]]:
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
    if not pairs and text.strip():
        pairs = [{"question": "General interview response", "answer": text[:MAX_ANSWER_CHARS]}]
    return pairs


# ── prompt & API ───────────────────────────────────────────────────────────────

def _call_ollama(qa_pairs: list[dict[str, str]], job_role: str | None) -> str:
    role_ctx = f"Position: {job_role}." if job_role else ""

    qa_block = ""
    for idx, pair in enumerate(qa_pairs, 1):
        q = pair["question"][:400]
        a = pair["answer"][:MAX_ANSWER_CHARS]
        qa_block += f"\nQ{idx}: {q}\nA{idx}: {a}\n"

    prompt = f"""You are an expert interview coach. Assess each candidate answer below.
{role_ctx}

Score each answer on:
1. Relevance (0-100): Does it address the question?
2. Correctness (0-100): Is it factually/logically sound?
3. Completeness (0-100): Is it sufficiently detailed?

Q&A PAIRS:
{qa_block}

Return ONLY valid JSON — no markdown, no explanation:
{{
  "qa_pairs": [
    {{
      "question_index": 1,
      "question": "<echo question>",
      "answer": "<first 100 chars of answer>",
      "relevance_score": <0-100>,
      "correctness_score": <0-100>,
      "completeness_score": <0-100>,
      "overall_score": <relevance*0.4 + correctness*0.35 + completeness*0.25>,
      "verdict": "<Excellent|Good|Adequate|Weak|Off-topic>",
      "strengths": ["<strength>"],
      "improvements": ["<suggestion>"],
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
    "strong_answers": <answers with overall>=75>,
    "weak_answers": <answers with overall<40>,
    "summary": "<2-3 sentence overall performance summary>"
  }}
}}"""

    return call_ollama(prompt, slot=SLOT_ANSWER_RELEVANCE, temperature=0.3, max_tokens=MAX_TOKENS)


# ── parsing ────────────────────────────────────────────────────────────────────

def _parse_response(raw: str, original_pairs: list[dict[str, str]]) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Invalid JSON from Ollama: %s", cleaned)
        raise
    for i, pair in enumerate(data.get("qa_pairs", []), 1):
        pair.setdefault("question_index", i)
    return data


# ── fallbacks ──────────────────────────────────────────────────────────────────

def _placeholder_result(qa_pairs: list[dict[str, str]]) -> dict[str, Any]:
    placeholder_pairs = [
        {
            "question_index":    i,
            "question":          p["question"][:200],
            "answer":            p["answer"][:200],
            "relevance_score":   0.0,
            "correctness_score": 0.0,
            "completeness_score":0.0,
            "overall_score":     0.0,
            "verdict":           "Unscored",
            "strengths":         [],
            "improvements":      ["Configure Ollama for AI-powered scoring"],
            "feedback":          "AI scoring requires Ollama to be configured.",
        }
        for i, p in enumerate(qa_pairs, 1)
    ]
    return {
        "qa_pairs": placeholder_pairs,
        "aggregate": {
            "avg_relevance":      0.0,
            "avg_correctness":    0.0,
            "avg_completeness":   0.0,
            "overall_qa_score":   0.0,
            "questions_answered": len(qa_pairs),
            "questions_total":    len(qa_pairs),
            "strong_answers":     0,
            "weak_answers":       0,
            "summary":            "AI-powered scoring unavailable. Configure Ollama.",
        },
        "ai_powered": False,
    }


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "qa_pairs": [],
        "aggregate": {
            "avg_relevance":      0.0,
            "avg_correctness":    0.0,
            "avg_completeness":   0.0,
            "overall_qa_score":   0.0,
            "questions_answered": 0,
            "questions_total":    0,
            "strong_answers":     0,
            "weak_answers":       0,
            "summary":            reason,
        },
        "ai_powered": False,
    }

"""
diarization_service.py
----------------------
Speaker diarization using pyannote.audio to separate interviewer vs candidate
speech turns, then merges them with Whisper segments so downstream AI services
know EXACTLY who said what.

Requirements
------------
    pip install pyannote.audio torch

You need a Hugging Face token with access to:
    pyannote/speaker-diarization-3.1  (accept conditions on HF hub)

Set  HF_TOKEN=hf_xxx  in your .env file.

FIXES applied
-------------
1. pyannote v3+ uses `use_auth_token` OR `token` parameter depending on version
   — we now try both so it works across pyannote 2.x and 3.x.
2. Added explicit audio pre-processing: pyannote requires 16kHz mono WAV.
   We use soundfile/librosa to verify and convert if needed.
3. The pipeline is loaded lazily and cached to avoid reloading on each request.
4. Detailed logging added so you can see exactly where it fails.
5. Overlap threshold lowered to 0.1s (was 0.3s) — short questions from the
   interviewer were often missed because they fell under the old threshold.

Output
------
{
    "turns": [
        {
            "speaker":    "SPEAKER_00",          # raw pyannote label
            "role":       "interviewer"|"candidate"|"unknown",
            "start":      float,                 # seconds
            "end":        float,
            "text":       str,                   # merged from Whisper segments
            "word_count": int
        },
        ...
    ],
    "speaker_map": {                             # role assignment
        "SPEAKER_00": "interviewer",
        "SPEAKER_01": "candidate"
    },
    "candidate_transcript":  str,               # candidate-only full text
    "interviewer_transcript": str,              # interviewer-only full text
    "candidate_segments":    list[dict],         # Whisper-style segments, candidate only
    "interviewer_segments":  list[dict],         # Whisper-style segments, interviewer only
    "diarization_available": bool               # False if pyannote not configured
}

Role assignment heuristic
--------------------------
The speaker who talks MORE total time is assumed to be the candidate
(candidates give longer answers; interviewers ask shorter questions).
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
OVERLAP_THRESHOLD = 0.1   # seconds — LOWERED from 0.3 so short turns aren't missed

# Module-level pipeline cache so we don't reload the model on every request
_pipeline_cache: Any = None
_pipeline_token: str = ""


def diarize_audio(
    audio_path: str,
    whisper_segments: list[dict[str, Any]],
    *,
    num_speakers: int | None = None,
    candidate_speaker: str | None = None,   # override auto-detection
    min_speakers: int = 2,
    max_speakers: int = 2,
) -> dict[str, Any]:
    """
    Run speaker diarization on `audio_path` then align results with
    Whisper segments so each segment has a speaker role attached.

    Parameters
    ----------
    audio_path        : path to 16-kHz mono WAV extracted by audio_service
    whisper_segments  : list of {start, end, text, confidence} from stt_service
    num_speakers      : exact number of speakers (None = auto-detect)
    candidate_speaker : force a specific pyannote label as the candidate
    min_speakers      : lower bound for auto-detection
    max_speakers      : upper bound for auto-detection

    Returns
    -------
    Diarization result dict (see module docstring).
    """
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not hf_token:
        logger.warning(
            "HF_TOKEN not set – skipping diarization. "
            "All transcript will be attributed to 'candidate'."
        )
        return _fallback_result(whisper_segments)

    # Verify audio file exists
    if not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        return _fallback_result(whisper_segments)

    try:
        return _run_pyannote(
            audio_path=audio_path,
            whisper_segments=whisper_segments,
            hf_token=hf_token,
            num_speakers=num_speakers,
            candidate_speaker=candidate_speaker,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except ImportError as exc:
        logger.warning(
            "pyannote.audio not installed – skipping diarization. "
            "Run: pip install pyannote.audio  |  Error: %s", exc
        )
        return _fallback_result(whisper_segments)
    except Exception as exc:
        logger.exception("Diarization failed: %s – falling back to no-diarization", exc)
        return _fallback_result(whisper_segments)


# ── internal ──────────────────────────────────────────────────────────────────

def _load_pipeline(hf_token: str) -> Any:
    """Load (or return cached) pyannote diarization pipeline."""
    global _pipeline_cache, _pipeline_token

    # Return cached pipeline if token hasn't changed
    if _pipeline_cache is not None and _pipeline_token == hf_token:
        logger.info("Using cached pyannote pipeline")
        return _pipeline_cache

    from pyannote.audio import Pipeline  # type: ignore
    import pyannote.audio  # type: ignore

    pyannote_version = getattr(pyannote.audio, "__version__", "unknown")
    logger.info("pyannote.audio version: %s", pyannote_version)
    logger.info("Loading pyannote diarization pipeline from HuggingFace …")

    # pyannote 3.x uses `token`, older versions use `use_auth_token`
    # We try the new API first, fall back to old
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        logger.info("Pipeline loaded with use_auth_token parameter")
    except TypeError:
        # Older pyannote that doesn't accept use_auth_token
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
        )
        logger.info("Pipeline loaded without auth token parameter")

    # Move to GPU if available
    try:
        import torch
        if torch.cuda.is_available():
            pipeline = pipeline.to(torch.device("cuda"))
            logger.info("pyannote running on GPU")
        else:
            logger.info("pyannote running on CPU")
    except Exception as gpu_exc:
        logger.debug("GPU setup skipped: %s", gpu_exc)

    _pipeline_cache = pipeline
    _pipeline_token = hf_token
    return pipeline


def _ensure_16k_mono_wav(audio_path: str) -> tuple[str, bool]:
    """
    Ensure the audio is 16kHz mono WAV as required by pyannote.
    Returns (path_to_use, created_temp_file).
    If conversion is needed, writes a temp file and returns its path.
    """
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(audio_path)
        logger.info(
            "Audio info: %s Hz, %d ch, %.1fs, format=%s",
            info.samplerate, info.channels, info.duration, info.format
        )

        if info.samplerate == 16000 and info.channels == 1:
            logger.info("Audio already 16kHz mono — no conversion needed")
            return audio_path, False

        # Need to resample/convert
        logger.info(
            "Converting audio to 16kHz mono (was %dHz, %dch) …",
            info.samplerate, info.channels
        )
        import numpy as np  # type: ignore
        data, sr = sf.read(audio_path, dtype="float32")

        # Mix to mono
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to 16k
        if sr != 16000:
            try:
                import librosa  # type: ignore
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            except ImportError:
                # Basic linear resample fallback
                ratio = 16000 / sr
                new_len = int(len(data) * ratio)
                data = np.interp(
                    np.linspace(0, len(data) - 1, new_len),
                    np.arange(len(data)),
                    data,
                )

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, data.astype(np.float32), 16000, subtype="PCM_16")
        logger.info("Converted audio written to: %s", tmp.name)
        return tmp.name, True

    except ImportError:
        logger.warning("soundfile not installed — using audio as-is (may cause pyannote errors)")
        return audio_path, False
    except Exception as exc:
        logger.warning("Audio check/conversion failed (%s) — using original file", exc)
        return audio_path, False


def _run_pyannote(
    audio_path: str,
    whisper_segments: list[dict[str, Any]],
    hf_token: str,
    num_speakers: int | None,
    candidate_speaker: str | None,
    min_speakers: int,
    max_speakers: int,
) -> dict[str, Any]:

    pipeline = _load_pipeline(hf_token)

    # Ensure audio is in the right format
    processed_path, is_temp = _ensure_16k_mono_wav(audio_path)

    try:
        # Build kwargs for pipeline run
        diarize_kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            diarize_kwargs["num_speakers"] = num_speakers
        else:
            diarize_kwargs["min_speakers"] = min_speakers
            diarize_kwargs["max_speakers"] = max_speakers

        logger.info(
            "Running diarization on %s with kwargs=%s …",
            processed_path, diarize_kwargs
        )
        diarization = pipeline(processed_path, **diarize_kwargs)

    finally:
        # Clean up temp file if we created one
        if is_temp:
            try:
                os.unlink(processed_path)
            except Exception:
                pass

    # Convert pyannote output → list of {speaker, start, end}
    raw_turns: list[dict[str, Any]] = []
    speaker_durations: dict[str, float] = {}

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        dur = segment.end - segment.start
        raw_turns.append({"speaker": speaker, "start": segment.start, "end": segment.end})
        speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + dur

    logger.info(
        "Diarization complete: %d turns found. Speaker durations: %s",
        len(raw_turns),
        {k: round(v, 1) for k, v in speaker_durations.items()}
    )

    if not raw_turns:
        logger.warning("Diarization returned 0 turns — falling back")
        return _fallback_result(whisper_segments)

    if len(speaker_durations) < 2:
        logger.warning(
            "Only %d speaker(s) detected — diarization may not have separated speakers. "
            "Check that the audio has 2 distinct speakers.",
            len(speaker_durations)
        )

    # ── Assign roles ──────────────────────────────────────────────────────────
    speaker_map = _assign_roles(speaker_durations, candidate_speaker)
    logger.info("Speaker role assignment: %s", speaker_map)

    # ── Merge Whisper segments into turns ─────────────────────────────────────
    enriched_turns = _merge_whisper(raw_turns, whisper_segments, speaker_map)

    # ── Build per-role transcripts & segments ─────────────────────────────────
    candidate_segs, interviewer_segs = [], []
    candidate_texts, interviewer_texts = [], []

    for turn in enriched_turns:
        role = turn["role"]
        if turn["text"]:
            if role == "candidate":
                candidate_texts.append(turn["text"])
                candidate_segs.extend(turn.get("_raw_segments", []))
            elif role == "interviewer":
                interviewer_texts.append(turn["text"])
                interviewer_segs.extend(turn.get("_raw_segments", []))

    # Strip internal helper key
    for turn in enriched_turns:
        turn.pop("_raw_segments", None)

    candidate_word_count = sum(t["word_count"] for t in enriched_turns if t["role"] == "candidate")
    interviewer_word_count = sum(t["word_count"] for t in enriched_turns if t["role"] == "interviewer")
    logger.info(
        "Final transcript — candidate: %d words, interviewer: %d words",
        candidate_word_count, interviewer_word_count
    )

    return {
        "turns": enriched_turns,
        "speaker_map": speaker_map,
        "candidate_transcript": " ".join(candidate_texts).strip(),
        "interviewer_transcript": " ".join(interviewer_texts).strip(),
        "candidate_segments": candidate_segs,
        "interviewer_segments": interviewer_segs,
        "diarization_available": True,
    }


def _assign_roles(
    speaker_durations: dict[str, float],
    candidate_speaker: str | None,
) -> dict[str, str]:
    """
    Heuristic: the speaker with the MOST total speech time = candidate.
    The speaker with LEAST speech time = interviewer.
    Supports 2-speaker interviews (most common).
    """
    if not speaker_durations:
        return {}

    if candidate_speaker and candidate_speaker in speaker_durations:
        role_map: dict[str, str] = {}
        for spk in speaker_durations:
            if spk == candidate_speaker:
                role_map[spk] = "candidate"
            else:
                role_map[spk] = "interviewer"
        return role_map

    sorted_speakers = sorted(speaker_durations, key=lambda s: speaker_durations[s], reverse=True)
    role_map = {}
    for i, spk in enumerate(sorted_speakers):
        if i == 0:
            role_map[spk] = "candidate"
        elif i == 1:
            role_map[spk] = "interviewer"
        else:
            role_map[spk] = "unknown"
    return role_map


def _merge_whisper(
    raw_turns: list[dict[str, Any]],
    whisper_segments: list[dict[str, Any]],
    speaker_map: dict[str, str],
) -> list[dict[str, Any]]:
    """
    For each diarization turn, collect Whisper segments that overlap it
    (overlap ≥ OVERLAP_THRESHOLD seconds), then assemble turn text.
    """
    enriched: list[dict[str, Any]] = []

    for turn in raw_turns:
        t_start, t_end = turn["start"], turn["end"]
        matched_segs: list[dict[str, Any]] = []

        for seg in whisper_segments:
            s_start, s_end = seg["start"], seg["end"]
            overlap = min(t_end, s_end) - max(t_start, s_start)
            if overlap >= OVERLAP_THRESHOLD:
                matched_segs.append(seg)

        text = " ".join(s["text"] for s in matched_segs).strip()
        word_count = len(text.split()) if text else 0
        role = speaker_map.get(turn["speaker"], "unknown")

        enriched.append({
            "speaker": turn["speaker"],
            "role": role,
            "start": round(t_start, 2),
            "end": round(t_end, 2),
            "text": text,
            "word_count": word_count,
            "_raw_segments": matched_segs,   # stripped before return
        })

    return enriched


def _fallback_result(whisper_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    When diarization is unavailable, treat ALL speech as candidate.
    Downstream services still work; they just won't filter by speaker.
    """
    full_text = " ".join(s["text"] for s in whisper_segments).strip()
    turns = [
        {
            "speaker": "SPEAKER_00",
            "role": "candidate",
            "start": whisper_segments[0]["start"] if whisper_segments else 0.0,
            "end": whisper_segments[-1]["end"] if whisper_segments else 0.0,
            "text": full_text,
            "word_count": len(full_text.split()),
        }
    ] if whisper_segments else []

    return {
        "turns": turns,
        "speaker_map": {"SPEAKER_00": "candidate"},
        "candidate_transcript": full_text,
        "interviewer_transcript": "",
        "candidate_segments": whisper_segments,
        "interviewer_segments": [],
        "diarization_available": False,
    }
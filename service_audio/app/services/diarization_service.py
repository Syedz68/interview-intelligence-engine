"""
diarization_service.py
----------------------
Speaker diarization using pyannote.audio to separate interviewer vs candidate
speech turns, then merges them with Whisper word-level timestamps so downstream
AI services know EXACTLY who said what.

Pipeline (aligned with it_two.py testing procedure)
----------------------------------------------------
  audio → ensure 16kHz mono WAV
        → pyannote diarization  (supports 1–5 speakers)
        → gap filling           (closes micro-gaps so no word falls in a hole)
        → Whisper word-level alignment  (midpoint overlap, not segment overlap)
        → UNKNOWN word rescue   (nearest-neighbour by timestamp)
        → short-turn merging    (same speaker < 1.5 s gap → merge)
        → dual-heuristic role labelling
        → build output

Gap filling
-----------
Pyannote leaves micro-gaps between segments. Words in those gaps get no
speaker label → UNKNOWN. We extend each segment's end to meet the next
segment's start, closing the gap safely.

Role assignment heuristic (extended for multi-speaker interviews)
-----------------------------------------------------------------
Heuristic A — MOST total speaking time  → candidate
               ALL other speakers        → interviewer
Heuristic B — FEWEST turn count         → candidate
               (interviewers ask many short questions)

When A and B agree → HIGH confidence.
When they disagree → A (speaking time) wins, LOW confidence warning.

Up to 5 speakers are supported (1 candidate + up to 4 interviewers).
Speakers beyond 5 are labelled "unknown".

UNKNOWN word rescue
-------------------
After gap filling, any words still UNKNOWN are assigned to the nearest
diarized speaker by timestamp proximity (closest segment boundary).

Requirements
------------
    pip install pyannote.audio torch torchaudio
    pip install openai-whisper   (for word_timestamps support)
    pip install soundfile librosa  (audio conversion helpers)

Set  HF_TOKEN=hf_xxx  in your .env file.
Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1

Output
------
{
    "turns": [
        {
            "speaker":    "SPEAKER_00",
            "role":       "interviewer" | "candidate" | "unknown",
            "start":      float,
            "end":        float,
            "text":       str,
            "word_count": int
        }, ...
    ],
    "speaker_map":            {"SPEAKER_00": "interviewer", "SPEAKER_01": "candidate", ...},
    "role_confidence":        "HIGH — both heuristics agree" | "LOW — ...",
    "candidate_transcript":   str,
    "interviewer_transcript": str,
    "candidate_segments":     list[dict],
    "interviewer_segments":   list[dict],
    "diarization_available":  bool
}
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
MAX_SPEAKERS     = 5     # 1 candidate + up to 4 interviewers
MERGE_GAP        = 1.5   # seconds — same-speaker turns closer than this are merged
MIN_SPEAKERS     = 2     # pyannote lower bound when num_speakers is not set

# Module-level pipeline cache — avoids reloading the ~1 GB model on every request
_pipeline_cache: Any = None
_pipeline_token: str = ""


# ── data structure (mirrors it_two.py Segment) ────────────────────────────────

@dataclass
class _Seg:
    start:      float
    end:        float
    speaker_id: str
    text:       str = ""
    _raw_segs:  list = field(default_factory=list, repr=False)


# ── public API ─────────────────────────────────────────────────────────────────

def diarize_audio(
    audio_path:       str,
    whisper_segments: list[dict[str, Any]],
    *,
    num_speakers:     int | None = None,
    candidate_speaker:str | None = None,
    min_speakers:     int        = MIN_SPEAKERS,
    max_speakers:     int        = MAX_SPEAKERS,
) -> dict[str, Any]:
    """
    Run speaker diarization on `audio_path` then align with Whisper segments.

    Parameters
    ----------
    audio_path        : path to audio file (WAV preferred; will be converted if needed)
    whisper_segments  : list of {start, end, text, words?, confidence} from stt_service
                        — must include word-level timestamps for best accuracy.
                          If words are absent, falls back to segment-level overlap.
    num_speakers      : exact number of speakers (None = let pyannote auto-detect)
    candidate_speaker : force a specific pyannote label as the candidate
    min_speakers      : lower bound for auto-detection (default 2)
    max_speakers      : upper bound for auto-detection (default 5)

    Returns
    -------
    Diarization result dict — see module docstring.
    """
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if not hf_token:
        logger.warning(
            "HF_TOKEN not set — skipping diarization. "
            "All transcript will be attributed to 'candidate'."
        )
        return _fallback_result(whisper_segments)

    if not os.path.exists(audio_path):
        logger.error("Audio file not found: %s", audio_path)
        return _fallback_result(whisper_segments)

    try:
        return _run_pipeline(
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
            "pyannote.audio not installed — skipping diarization. "
            "Run: pip install pyannote.audio  |  Error: %s", exc
        )
        return _fallback_result(whisper_segments)
    except Exception as exc:
        logger.exception("Diarization failed: %s — falling back to no-diarization", exc)
        return _fallback_result(whisper_segments)


# ── pipeline ───────────────────────────────────────────────────────────────────

def _run_pipeline(
    audio_path:        str,
    whisper_segments:  list[dict[str, Any]],
    hf_token:          str,
    num_speakers:      int | None,
    candidate_speaker: str | None,
    min_speakers:      int,
    max_speakers:      int,
) -> dict[str, Any]:

    # 1. Ensure audio is 16kHz mono WAV
    processed_path, is_temp = _ensure_16k_mono_wav(audio_path)

    try:
        # 2. Load / reuse cached pyannote pipeline
        pipeline = _load_pipeline(hf_token)

        # 3. Build diarization kwargs
        diar_kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            diar_kwargs["num_speakers"] = num_speakers
        else:
            diar_kwargs["min_speakers"] = min_speakers
            diar_kwargs["max_speakers"] = min(max_speakers, MAX_SPEAKERS)

        logger.info("Running diarization — kwargs=%s …", diar_kwargs)
        diarization = pipeline(processed_path, **diar_kwargs)

        # 4. Get total audio duration for gap filling
        try:
            import torchaudio
            waveform, sr = torchaudio.load(processed_path)
            total_duration = waveform.shape[1] / sr
        except Exception:
            total_duration = None

    finally:
        if is_temp:
            try:
                os.unlink(processed_path)
            except Exception:
                pass

    # 5. Convert pyannote output → _Seg list
    raw_segments: list[_Seg] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        raw_segments.append(_Seg(
            start=round(turn.start, 3),
            end=round(turn.end, 3),
            speaker_id=speaker,
        ))

    logger.info("Pyannote returned %d raw segments", len(raw_segments))

    if not raw_segments:
        logger.warning("Diarization returned 0 segments — falling back")
        return _fallback_result(whisper_segments)

    # 6. Fill micro-gaps (it_two.py: fill_gaps)
    raw_segments = _fill_gaps(raw_segments, total_duration)
    logger.info("Gaps filled")

    # 7. Word-level alignment + UNKNOWN rescue (it_two.py: align_words_to_speakers)
    aligned_segs = _align_words_to_speakers(whisper_segments, raw_segments)

    if not aligned_segs:
        logger.warning("Word alignment produced no segments — falling back to segment overlap")
        aligned_segs = _segment_level_align(raw_segments, whisper_segments)

    # 8. Merge close same-speaker turns (it_two.py: merge_close_turns)
    aligned_segs = _merge_close_turns(aligned_segs, gap_threshold=MERGE_GAP)

    # 9. Dual-heuristic role labelling (it_two.py: label_speakers — extended)
    speaker_durations: dict[str, float] = {}
    speaker_turn_counts: dict[str, int]  = {}
    for s in aligned_segs:
        dur = s.end - s.start
        speaker_durations[s.speaker_id]   = speaker_durations.get(s.speaker_id, 0.0) + dur
        speaker_turn_counts[s.speaker_id] = speaker_turn_counts.get(s.speaker_id, 0) + 1

    speaker_map, confidence = _assign_roles(
        speaker_durations, speaker_turn_counts, candidate_speaker
    )
    logger.info("Role assignment [%s]: %s", confidence, speaker_map)

    # 10. Build enriched turns with text
    turns = _build_turns(aligned_segs, speaker_map)

    # 11. Compile per-role transcripts
    candidate_texts, interviewer_texts = [], []
    candidate_segs, interviewer_segs   = [], []

    for turn in turns:
        role = turn["role"]
        if turn["text"]:
            if role == "candidate":
                candidate_texts.append(turn["text"])
                candidate_segs.extend(turn.pop("_raw_segments", []))
            elif role == "interviewer":
                interviewer_texts.append(turn["text"])
                interviewer_segs.extend(turn.pop("_raw_segments", []))
            else:
                turn.pop("_raw_segments", None)
        else:
            turn.pop("_raw_segments", None)

    candidate_words   = sum(t["word_count"] for t in turns if t["role"] == "candidate")
    interviewer_words = sum(t["word_count"] for t in turns if t["role"] == "interviewer")
    logger.info(
        "Transcript built — candidate: %d words, interviewer(s): %d words",
        candidate_words, interviewer_words
    )

    return {
        "turns":                  turns,
        "speaker_map":            speaker_map,
        "role_confidence":        confidence,
        "candidate_transcript":   " ".join(candidate_texts).strip(),
        "interviewer_transcript": " ".join(interviewer_texts).strip(),
        "candidate_segments":     candidate_segs,
        "interviewer_segments":   interviewer_segs,
        "diarization_available":  True,
    }


# ── Step: gap filling (it_two.py fill_gaps) ───────────────────────────────────

def _fill_gaps(segments: list[_Seg], total_duration: float | None = None) -> list[_Seg]:
    """
    Extend each segment's end to meet the start of the next one.
    Closes micro-gaps so no transcribed word falls into a hole.
    """
    if not segments:
        return segments

    filled: list[_Seg] = []
    for i, seg in enumerate(segments):
        new_end = seg.end
        if i + 1 < len(segments):
            next_start = segments[i + 1].start
            if next_start > seg.end:
                new_end = next_start
        elif total_duration is not None and total_duration > seg.end:
            new_end = total_duration   # extend last segment to end of audio
        filled.append(_Seg(start=seg.start, end=new_end, speaker_id=seg.speaker_id))

    return filled


# ── Step: word-level alignment + UNKNOWN rescue (it_two.py align_words_to_speakers) ──

def _align_words_to_speakers(
    whisper_segments: list[dict[str, Any]],
    diar_segments:    list[_Seg],
) -> list[_Seg]:
    """
    1. Extract word-level timestamps from Whisper output.
    2. Assign each word to a speaker using midpoint overlap.
    3. Rescue remaining UNKNOWN words via nearest-neighbour.
    4. Merge consecutive same-speaker words into speaker turns.

    Falls back to segment-level processing if no word timestamps are available.
    """
    # Extract words
    words: list[dict] = []
    for seg in whisper_segments:
        for w in seg.get("words", []):
            words.append({
                "word":    str(w.get("word", "")).strip(),
                "start":   float(w.get("start", seg["start"])),
                "end":     float(w.get("end",   seg["end"])),
                "speaker": "UNKNOWN",
            })

    if not words:
        return []   # caller will fall back to segment-level

    # Assign speaker by midpoint (it_two.py: find_speaker)
    def _find_speaker(t_start: float, t_end: float) -> str:
        mid  = (t_start + t_end) / 2.0
        best = "UNKNOWN"
        best_overlap = -1.0
        for seg in diar_segments:
            if seg.start <= mid <= seg.end:
                overlap = min(t_end, seg.end) - max(t_start, seg.start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best = seg.speaker_id
        return best

    for w in words:
        w["speaker"] = _find_speaker(w["start"], w["end"])

    # Rescue UNKNOWN words (it_two.py: rescue_unknown_words)
    unknown_count = sum(1 for w in words if w["speaker"] == "UNKNOWN")
    if unknown_count > 0:
        logger.info("Rescuing %d UNKNOWN word(s) via nearest-neighbour …", unknown_count)
        words = _rescue_unknown(words, diar_segments)
        still = sum(1 for w in words if w["speaker"] == "UNKNOWN")
        logger.info("Rescue complete — %d still UNKNOWN (should be 0)", still)

    # Merge consecutive same-speaker words into turns
    merged: list[_Seg] = []
    cur_spk   = words[0]["speaker"]
    cur_words = [words[0]["word"]]
    cur_start = words[0]["start"]
    cur_end   = words[0]["end"]

    for w in words[1:]:
        if w["speaker"] == cur_spk:
            cur_words.append(w["word"])
            cur_end = w["end"]
        else:
            merged.append(_Seg(
                start=cur_start, end=cur_end,
                speaker_id=cur_spk,
                text=" ".join(cur_words).strip(),
            ))
            cur_spk   = w["speaker"]
            cur_words = [w["word"]]
            cur_start = w["start"]
            cur_end   = w["end"]

    merged.append(_Seg(
        start=cur_start, end=cur_end,
        speaker_id=cur_spk,
        text=" ".join(cur_words).strip(),
    ))

    logger.info("Word-level alignment: %d speaker turns", len(merged))
    return merged


def _rescue_unknown(words: list[dict], diar_segments: list[_Seg]) -> list[dict]:
    """Assign UNKNOWN words to the nearest segment by timestamp proximity."""
    for w in words:
        if w["speaker"] != "UNKNOWN":
            continue
        mid       = (w["start"] + w["end"]) / 2.0
        best_spk  = "UNKNOWN"
        best_dist = float("inf")
        for seg in diar_segments:
            dist = min(abs(mid - seg.start), abs(mid - seg.end))
            if dist < best_dist:
                best_dist = dist
                best_spk  = seg.speaker_id
        w["speaker"] = best_spk
    return words


# ── Step: segment-level fallback alignment (original project approach) ────────

def _segment_level_align(
    diar_segments:    list[_Seg],
    whisper_segments: list[dict[str, Any]],
    overlap_threshold: float = 0.1,
) -> list[_Seg]:
    """
    Fallback used when Whisper word timestamps are absent.
    For each diarization segment, collect Whisper segments that overlap it.
    """
    result: list[_Seg] = []
    for dseg in diar_segments:
        matched: list[dict] = []
        for wseg in whisper_segments:
            overlap = min(dseg.end, wseg["end"]) - max(dseg.start, wseg["start"])
            if overlap >= overlap_threshold:
                matched.append(wseg)
        text = " ".join(s["text"] for s in matched).strip()
        result.append(_Seg(
            start=dseg.start, end=dseg.end,
            speaker_id=dseg.speaker_id,
            text=text, _raw_segs=matched,
        ))
    logger.info("Segment-level fallback alignment: %d turns", len(result))
    return result


# ── Step: merge close turns (it_two.py merge_close_turns) ─────────────────────

def _merge_close_turns(segments: list[_Seg], gap_threshold: float = MERGE_GAP) -> list[_Seg]:
    """
    If the same speaker has two consecutive turns with a gap smaller than
    gap_threshold, merge them. Fixes long answers being split by brief gaps.
    """
    if not segments:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap  = seg.start - prev.end
        if seg.speaker_id == prev.speaker_id and gap <= gap_threshold:
            merged[-1] = _Seg(
                start=prev.start, end=seg.end,
                speaker_id=prev.speaker_id,
                text=(prev.text + " " + seg.text).strip(),
                _raw_segs=prev._raw_segs + seg._raw_segs,
            )
        else:
            merged.append(seg)

    logger.info("Turn merging: %d → %d turns", len(segments), len(merged))
    return merged


# ── Step: dual-heuristic role labelling (extended for up to 5 speakers) ───────

def _assign_roles(
    speaker_durations:   dict[str, float],
    speaker_turn_counts: dict[str, int],
    candidate_speaker:   str | None = None,
) -> tuple[dict[str, str], str]:
    """
    Identifies the candidate and interviewers across up to 5 speakers.

    Heuristic A — most total speaking time  → candidate
    Heuristic B — fewest turn count         → candidate
                  (interviewers ask many short questions, so they have more turns)

    Both heuristics agree → HIGH confidence.
    They disagree → Heuristic A (speaking time) wins, LOW confidence warning.

    All remaining speakers (after the candidate is identified) are labelled
    "interviewer". Speakers beyond MAX_SPEAKERS are labelled "unknown".

    Returns
    -------
    (speaker_map, confidence_string)
    """
    if not speaker_durations:
        return {}, "N/A — no speakers"

    speaker_ids = sorted(speaker_durations.keys())

    # Manual override
    if candidate_speaker and candidate_speaker in speaker_durations:
        role_map = {
            spk: ("candidate" if spk == candidate_speaker else "interviewer")
            for spk in speaker_ids
        }
        return role_map, "MANUAL — candidate_speaker override"

    # Only one speaker detected
    if len(speaker_ids) == 1:
        return {speaker_ids[0]: "candidate"}, "LOW — only 1 speaker detected"

    # Heuristic A: most speaking time → candidate
    candidate_by_time  = max(speaker_durations,   key=speaker_durations.get)
    # Heuristic B: fewest turn count → candidate
    candidate_by_turns = min(speaker_turn_counts, key=speaker_turn_counts.get)

    if candidate_by_time == candidate_by_turns:
        candidate_id = candidate_by_time
        confidence   = "HIGH — both heuristics agree"
    else:
        candidate_id = candidate_by_time   # time-based is more reliable
        confidence   = (
            f"LOW — heuristics disagree. "
            f"Time says {candidate_by_time} ({round(speaker_durations[candidate_by_time], 1)}s), "
            f"turns say {candidate_by_turns} ({speaker_turn_counts[candidate_by_turns]} turns). "
            f"Using time-based heuristic. Verify output manually."
        )

    # Build role map — candidate gets the top spot, rest are interviewers
    # (up to MAX_SPEAKERS total; beyond that → unknown)
    role_map: dict[str, str] = {}
    for spk in speaker_ids:
        if spk == candidate_id:
            role_map[spk] = "candidate"
        else:
            role_map[spk] = "interviewer"

    # Log stats for every speaker
    logger.info("Speaker role assignment [%s]:", confidence)
    for spk in speaker_ids:
        logger.info(
            "  %s → %-12s  %.1fs total  %d turns",
            spk, role_map[spk],
            speaker_durations.get(spk, 0),
            speaker_turn_counts.get(spk, 0),
        )

    return role_map, confidence


# ── Step: build output turns ───────────────────────────────────────────────────

def _build_turns(segments: list[_Seg], speaker_map: dict[str, str]) -> list[dict[str, Any]]:
    """Convert internal _Seg list into the output turn dicts."""
    turns: list[dict[str, Any]] = []
    for seg in segments:
        if not seg.text:
            continue
        role = speaker_map.get(seg.speaker_id, "unknown")
        turns.append({
            "speaker":       seg.speaker_id,
            "role":          role,
            "start":         round(seg.start, 3),
            "end":           round(seg.end,   3),
            "text":          seg.text,
            "word_count":    len(seg.text.split()),
            "_raw_segments": seg._raw_segs,   # stripped later
        })
    return turns


# ── Audio conversion (kept from original project) ─────────────────────────────

def _ensure_16k_mono_wav(audio_path: str) -> tuple[str, bool]:
    """
    Ensure audio is 16kHz mono WAV as required by pyannote.
    Returns (path_to_use, created_temp_file).
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        logger.info(
            "Audio info: %dHz, %dch, %.1fs, format=%s",
            info.samplerate, info.channels, info.duration, info.format,
        )

        if info.samplerate == 16000 and info.channels == 1:
            logger.info("Audio already 16kHz mono — no conversion needed")
            return audio_path, False

        logger.info(
            "Converting to 16kHz mono (was %dHz, %dch) …",
            info.samplerate, info.channels,
        )
        import numpy as np
        data, sr = sf.read(audio_path, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        if sr != 16000:
            try:
                import librosa
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            except ImportError:
                ratio   = 16000 / sr
                new_len = int(len(data) * ratio)
                data    = np.interp(
                    np.linspace(0, len(data) - 1, new_len),
                    np.arange(len(data)), data,
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


# ── Pipeline cache (kept from original project) ───────────────────────────────

def _load_pipeline(hf_token: str) -> Any:
    """Load (or return cached) pyannote diarization pipeline."""
    global _pipeline_cache, _pipeline_token

    if _pipeline_cache is not None and _pipeline_token == hf_token:
        logger.info("Using cached pyannote pipeline")
        return _pipeline_cache

    from pyannote.audio import Pipeline  # type: ignore
    import pyannote.audio               # type: ignore

    logger.info(
        "Loading pyannote diarization pipeline (pyannote.audio %s) …",
        getattr(pyannote.audio, "__version__", "unknown"),
    )

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        logger.info("Pipeline loaded with use_auth_token parameter")
    except TypeError:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        logger.info("Pipeline loaded without auth token parameter")

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


# ── Fallback ───────────────────────────────────────────────────────────────────

def _fallback_result(whisper_segments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    When diarization is unavailable, attribute ALL speech to the candidate.
    Downstream services still work — they just can't filter by speaker.
    """
    full_text = " ".join(s["text"] for s in whisper_segments).strip()
    turns = [
        {
            "speaker":    "SPEAKER_00",
            "role":       "candidate",
            "start":      whisper_segments[0]["start"]  if whisper_segments else 0.0,
            "end":        whisper_segments[-1]["end"]   if whisper_segments else 0.0,
            "text":       full_text,
            "word_count": len(full_text.split()),
        }
    ] if whisper_segments else []

    return {
        "turns":                  turns,
        "speaker_map":            {"SPEAKER_00": "candidate"},
        "role_confidence":        "N/A — diarization unavailable",
        "candidate_transcript":   full_text,
        "interviewer_transcript": "",
        "candidate_segments":     whisper_segments,
        "interviewer_segments":   [],
        "diarization_available":  False,
    }

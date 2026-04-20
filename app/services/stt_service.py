import whisper

model = whisper.load_model("base")
# options: tiny, base, small, medium, large


def transcribe_audio(audio_path: str) -> dict:
    result = model.transcribe(audio_path)

    full_text = result["text"].strip()

    formatted_segments = [
        {
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
            "confidence": round(seg.get("avg_logprob", 0), 2)
        }
        for seg in result["segments"]
    ]

    return {
        "text": full_text,
        "segments": formatted_segments
    }
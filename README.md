# Interview Intelligence Engine

AI-powered behavioral interview analysis split into two isolated microservices.

---

## Project Structure

```
interview-intelligence-engine/
│
├── service_audio/                 ← .venv_audio  (numpy<2)
│   ├── main.py                    FastAPI app — port 8001
│   └── app/
│       └── services/
│           ├── audio_service.py
│           ├── stt_service.py
│           ├── diarization_service.py
│           └── speech_analysis_service.py
│
├── service_web/                   ← .venv_web  (numpy≥2)
│   ├── main.py                    FastAPI gateway — port 8000
│   ├── face_landmarker.task       MediaPipe model (auto-downloaded if missing)
│   └── app/
│       ├── api/
│       │   ├── v1/routes/process.py
│       │   └── v2/routes/process.py
│       ├── core/
│       │   ├── config.py
│       │   ├── error_handler.py
│       │   ├── exception_handler.py
│       │   └── numpy_encoder.py
│       ├── schemas/
│       │   ├── request.py
│       │   └── response.py
│       ├── pipeline/
│       │   └── intelligence_pipeline.py   (calls service_audio via HTTP)
│       └── services/
│           ├── emotion_service.py
│           ├── gaze_service.py
│           ├── motion_service.py
│           ├── frame_extraction_service.py
│           ├── face_landmark_service.py
│           ├── behavioral_aggregator.py
│           ├── english_fluency_service.py
│           ├── answer_relevance_service.py
│           ├── behavioral_report_service.py
│           ├── ollama_client.py
│           └── video_service.py
│
├── start.bat / start.sh           Launch scripts for both services
├── requirements_env_a.txt         service_audio deps  (numpy<2)
├── requirements_env_b.txt         service_web deps    (numpy≥2)
└── .env.example
```

---

## Why Two Environments?

| Constraint | numpy < 2 | numpy ≥ 2 |
|---|---|---|
| pyannote.audio 3.3.x | ✅ | ❌ |
| numba 0.65.x | ✅ | ❌ |
| mediapipe 0.10.x | ❌ | ✅ |
| DeepFace / TF 2.21 | ✅ | ✅ |

`service_audio` handles Whisper + pyannote (numpy<2).  
`service_web` handles OpenCV / MediaPipe / DeepFace + the FastAPI gateway (numpy≥2).

---

## Quick Start

### 1. Set up environments

```bash
# Audio environment (numpy<2)
python -m venv .venv_audio
source .venv_audio/bin/activate        # Windows: .venv_audio\Scripts\activate
pip install "numpy==1.26.4"
pip install torch==2.2.2+cu121 torchaudio==2.2.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_env_a.txt
deactivate

# Web environment (numpy≥2)
python -m venv .venv_web
source .venv_web/bin/activate
pip install -r requirements_env_b.txt
deactivate
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and fill in:
#   HF_TOKEN        – Hugging Face token for pyannote diarization
#   OLLAMA_MODEL    – e.g. llama3.1 (run: ollama pull llama3.1)
#   SECRET_KEY      – any random string
```

### 3. Start both services

```bash
# Linux / Mac
chmod +x start.sh
./start.sh

# Windows
start.bat
```

Or start individually:

```bash
./start.sh audio   # port 8001 only
./start.sh web     # port 8000 only
```

### 4. Open the API docs

- **service_web**   → http://localhost:8000/docs  
- **service_audio** → http://localhost:8001/docs  

---

## API Endpoints

### service_web (port 8000)

| Method | Path | Description |
|---|---|---|
| POST | `/intelligence/process-interview` | v1 – STT only |
| POST | `/intelligence/process-video` | v1 – landmark extraction |
| POST | `/new-intelligence/process-interview` | v2 – STT only |
| POST | `/new-intelligence/analyze-interview` | v2 – full behavioral pipeline |
| POST | `/new-intelligence/analyze-interview-v3` | v3 – full AI pipeline (recommended) |

### service_audio (port 8001)

| Method | Path | Description |
|---|---|---|
| POST | `/extract-audio` | Extract WAV from video path |
| POST | `/transcribe` | Whisper STT |
| POST | `/diarize` | pyannote speaker diarization |
| POST | `/analyze-speech` | Speech quality metrics |
| POST | `/full-audio-pipeline` | All audio stages in one call |

---

## AI Features (require Ollama)

Install Ollama and pull a model:

```bash
# https://ollama.com/download
ollama pull llama3.1
```

| Feature | Stage | Service |
|---|---|---|
| English Fluency (CEFR) | 11 | service_web → Ollama |
| Answer Relevance Scoring | 12 | service_web → Ollama |
| Behavioral Report + Hiring Recommendation | 14 | service_web → Ollama |

All AI features degrade gracefully when Ollama is unavailable.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AUDIO_SERVICE_URL` | `http://localhost:8001` | service_audio base URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1` | Model to use |
| `HF_TOKEN` | *(required)* | Hugging Face token for pyannote |
| `W_COMMUNICATION` | `0.40` | Behavioral score weight |
| `W_ENGAGEMENT` | `0.30` | Behavioral score weight |
| `W_CONFIDENCE` | `0.20` | Behavioral score weight |
| `W_PROFESSIONALISM` | `0.10` | Behavioral score weight |

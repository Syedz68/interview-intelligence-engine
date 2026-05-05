#!/usr/bin/env bash
# ============================================================
# start.sh  –  Launch both microservices
# ============================================================
# Usage:
#   ./start.sh            # start both services
#   ./start.sh audio      # start service_audio only (port 8001)
#   ./start.sh web        # start service_web only   (port 8000)
# ============================================================

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

start_audio() {
    echo "▶  Starting service_audio on port 8001..."
    cd "$ROOT/service_audio"
    # Activate the numpy<2 venv if it exists
    if [ -d "$ROOT/.venv_audio" ]; then
        source "$ROOT/.venv_audio/bin/activate"
    fi
    uvicorn main:app --host 0.0.0.0 --port 8001 --reload &
    AUDIO_PID=$!
    echo "   service_audio PID: $AUDIO_PID"
}

start_web() {
    echo "▶  Starting service_web on port 8000..."
    cd "$ROOT/service_web"
    # Activate the numpy≥2 venv if it exists
    if [ -d "$ROOT/.venv_web" ]; then
        source "$ROOT/.venv_web/bin/activate"
    fi
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
    WEB_PID=$!
    echo "   service_web PID: $WEB_PID"
}

case "${1:-both}" in
    audio)
        start_audio
        wait $AUDIO_PID
        ;;
    web)
        start_web
        wait $WEB_PID
        ;;
    both|"")
        start_audio
        sleep 2   # give audio service a head-start (Whisper model load is slow)
        start_web
        echo ""
        echo "Both services running."
        echo "  service_web   → http://localhost:8000/docs"
        echo "  service_audio → http://localhost:8001/docs"
        echo "Press Ctrl+C to stop."
        wait
        ;;
    *)
        echo "Unknown argument: $1"
        echo "Usage: ./start.sh [audio|web|both]"
        exit 1
        ;;
esac

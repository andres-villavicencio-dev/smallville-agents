#!/bin/bash
# Pause the generative agents sim, yield VRAM for TTS, wait, then resume.
# Usage: sim_pause_for_tts.sh [pause_seconds]
#
# Called by cron 2 minutes after each podcast generation starts.

PAUSE_SECONDS=${1:-300}  # Default 5 minutes
SIM_URL="http://localhost:3000"

echo "[$(date)] Pausing sim (yield_vram=true) for ${PAUSE_SECONDS}s..."

# Pause with VRAM yield
PAUSE_RESULT=$(curl -s -X POST "${SIM_URL}/api/pause?yield_vram=true" 2>/dev/null)
if echo "$PAUSE_RESULT" | grep -q "paused"; then
    echo "[$(date)] Sim paused. VRAM freed for TTS."
else
    echo "[$(date)] WARNING: Could not pause sim (maybe not running?): $PAUSE_RESULT"
    exit 1
fi

# Wait for TTS to finish
sleep "$PAUSE_SECONDS"

# Resume
RESUME_RESULT=$(curl -s -X POST "${SIM_URL}/api/resume" 2>/dev/null)
if echo "$RESUME_RESULT" | grep -q "resumed"; then
    echo "[$(date)] Sim resumed."
else
    echo "[$(date)] WARNING: Could not resume sim: $RESUME_RESULT"
    exit 1
fi

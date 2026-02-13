#!/bin/bash

# Smallville Generative Agents Simulation
# Usage: ./run.sh [arguments]

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running. Please start it with: ollama serve"
    echo "   Then pull a model: ollama pull qwen2.5:3b"
    exit 1
fi

# Check if required model exists
MODEL=${OLLAMA_MODEL:-"qwen2.5:3b"}
if ! curl -s http://localhost:11434/api/show -d "{\"name\":\"$MODEL\"}" >/dev/null 2>&1; then
    echo "❌ Model $MODEL not found. Please pull it with: ollama pull $MODEL"
    exit 1
fi

echo "🏠 Starting Smallville Simulation..."
echo "   Model: $MODEL"
echo "   GPU Queue: $([ "$*" == *"--no-gpu-queue"* ] && echo "Disabled" || echo "Enabled")"
echo ""

# Run simulation
python main.py --committee "$@"

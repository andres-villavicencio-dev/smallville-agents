# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the Stanford paper "Generative Agents: Interactive Simulacra of Human Behavior" (arXiv:2304.03442) using local Ollama models. Simulates 25 AI agents living in "Smallville" with memory, reflection, planning, and natural conversation.

## Running the Simulation

```bash
# Install dependencies (Python 3.11+, venv at ./venv/)
pip install -r requirements.txt

# Prerequisites: Ollama must be running with models pulled
ollama serve
ollama pull qwen2.5:3b

# Run simulation (default: 10x speed, 2 days)
python main.py

# Common options
python main.py --speed 5 --days 3
python main.py --no-gpu-queue          # Skip GPU queue, call Ollama directly
python main.py --load-state saves/X.json
python main.py --committee             # Mixture-of-experts mode
python main.py --webui                 # Launch REST API + WebSocket map UI
python main.py --config                # Print config and exit

# Shell wrapper (checks Ollama health first)
./run.sh [args]

# Override models via env vars
OLLAMA_MODEL=llama3.2:3b python main.py
MODEL_PLANNING=qwen2.5:3b MODEL_CONVERSATION=llama3.2:3b python main.py

# Run tests
python -m pytest tests/ -v
```

## Architecture

```
main.py (SmallvilleSimulation)  — orchestrates tick loop, init, save/load, shutdown
  ├── agent.py (GenerativeAgent)     — cognition: plan, observe, reflect, act
  │     ├── memory.py (MemoryStream) — SQLite+FTS5 storage, TF-IDF retrieval scoring
  │     └── skillbank.py (SkillBank) — hierarchical skill library distilled from experience
  ├── environment.py (SmallvilleEnvironment) — 12 locations, agent movement tracking
  ├── conversation.py (ConversationManager)  — memory-driven dialogue between agents
  ├── llm.py (OllamaClient)         — async Ollama API + GPU queue integration
  ├── committee.py                   — optional mixture-of-experts (5 specialist models + judge)
  ├── display.py (SimulationDisplay) — Rich terminal UI with live panels
  ├── webui.py                       — REST API + WebSocket server for browser-based map UI
  ├── telegram_broadcaster.py        — optional Telegram channel broadcasting
  ├── personas.py                    — 25 agent definitions (dict-based)
  ├── prompts.py                     — LLM prompt templates
  ├── config.py                      — all tunable parameters and constants
  └── tests/                         — pytest suite (memory, agent, environment, conversation)
```

**Tick loop** (`SmallvilleSimulation.run`): Each tick advances `TICK_DURATION_SECONDS` (10s game time). Per tick: agent planning → observation → conversation initiation → display update.

**Memory retrieval** scores each memory as `α×recency + β×importance + γ×relevance` where recency uses exponential decay (0.99), importance is LLM-rated 1-10, and relevance is TF-IDF cosine similarity. Reflection triggers when cumulative importance of recent observations exceeds 150.

**SkillBank** (`skillbank.py`): Agents distill experiences (conversations, reflections, plan outcomes) into reusable skills stored in SQLite. Skills are retrieved by TF-IDF similarity and weighted by effectiveness. Inspired by SkillRL (arXiv:2602.08234).

**Task-specific model routing** (`config.py:MODELS`): Different Ollama models are assigned to planning, conversation, reflection, and importance-scoring tasks. Committee mode (`--committee`) replaces single-model calls with 5 specialist experts (social, spatial, temporal, emotional, memory) plus a judge synthesizer.

**GPU queue integration** (`llm.py`): Routes LLM calls through `../gpu-queue/queue_manager.py` with Pi 5 fallback (gemma3:1b). Disable with `--no-gpu-queue`.

## Key Data Paths

- `db/memories.db` — SQLite database for all agent memories and skills (created at runtime)
- `saves/` — JSON state snapshots (auto-save every 100 ticks)
- `simulation.log` — detailed activity log

## Customization Points

- **Add agents**: Add entries to `AGENT_PERSONAS` dict in `personas.py` (fields: name, age, occupation, personality, background, relationships, daily_routine, goals, home_location, work_location, optional special_memory)
- **Add locations**: Add entries to `SMALLVILLE_LOCATIONS` dict in `config.py`
- **Tune memory**: Adjust `MEMORY_RETRIEVAL_WEIGHTS`, `IMPORTANCE_THRESHOLD`, `RECENCY_DECAY_FACTOR` in `config.py`
- **Tune conversations**: Adjust `CONVERSATION_PROBABILITY`, `MAX_CONVERSATION_TURNS`, `CONVERSATION_RELEVANCE_THRESHOLD` in `config.py`

## Codebase Conventions

- Async throughout — all agent cognition and LLM calls use `async/await`
- No class inheritance hierarchy; agents are flat `GenerativeAgent` instances differentiated by persona data
- Prompt templates live in `prompts.py`, not inline in logic modules
- The simulation starts on 2023-02-13 08:00 (Valentine's Day scenario where Isabella Rodriguez spreads party info through natural conversation)

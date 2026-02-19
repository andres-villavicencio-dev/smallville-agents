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

## Committee Mode (Mixture of Experts)

Enabled with `--committee` or `USE_COMMITTEE=1`. Instead of one model per task, decisions flow through specialized experts sequentially (8GB VRAM constraint — one model at a time, Ollama handles swapping).

**7 experts** defined in `committee.py`:
| Expert | Default Model | Role |
|--------|--------------|------|
| Social | `smallville-social` (fine-tuned) | Relationships, social norms, interpersonal dynamics |
| Spatial | `qwen2.5:3b` | Locations, movement, where to go next |
| Temporal | `gemma3:1b` | Time-of-day, scheduling, urgency |
| Emotional | `llama3.2:3b` | Mood, personality, emotional states |
| Memory | `gemma3:4b` | Memory retrieval context and relevance |
| Dialogue | `smallville-actor` (fine-tuned) | Character voice and conversation |
| Judge | `qwen2.5:3b` | Synthesizes expert outputs into final action |

Override any expert model via env: `COMMITTEE_MODEL_SOCIAL=other:model`

**5 pipelines** route different decision types through relevant expert subsets: `decide_action`, `conversation_response`, `should_converse`, `plan_day`, `reflect`.

## Fine-Tuned Models

Two custom models trained via QLoRA on Gemma 2 2B (`unsloth/gemma-2-2b-it`):

1. **`smallville-social`** — Social reasoning expert (202 examples, 7 categories of social scenarios across all 25 agents)
2. **`smallville-actor`** — Dialogue expert (177 examples, distinctive character voices)

### Fine-Tuning Pipeline (`finetune/`)

```bash
# Generate training data programmatically
python finetune/generate_social_data.py    # → data/social_expert_training.jsonl
python finetune/generate_training_data.py  # → data/actor_training.jsonl

# Train (requires unsloth + trl, ~1 min per 50 examples on RTX 3070)
python finetune/train_social.py            # → output-social/
python finetune/train.py                   # → output/

# Export: two-step because direct GGUF OOMs on 8GB VRAM
python finetune/export_gguf.py             # save_pretrained_merged → convert_hf_to_gguf.py → Q8_0

# Deploy to Ollama
ollama create smallville-social -f finetune/output-social/Modelfile
ollama create smallville-actor -f finetune/output/Modelfile
```

**Training config**: LORA_R=64, LORA_ALPHA=128, dropout=0.05, LR=2e-4, cosine schedule, packing enabled.
**Python**: `~/.pyenv/versions/3.11.9/bin/python` (has unsloth, torch 2.5.1+cu121).

## Known Issues & Lessons

- **Reflection loops**: Never count reflections toward their own importance threshold (+ 5-min cooldown)
- **"Helpers not guests"**: Small models default to instrumental reasoning (tasks) over social reasoning — agents plan to "check wiring" at a party instead of attending for fun. Fine-tuned social expert addresses this.
- **Conversation meta-leak**: `_clean_dialogue()` regex in `conversation.py` strips LLM meta-commentary from dialogue
- **30-tick conversation cooldown** prevents agents from re-entering conversations immediately
- **Re-planning cap**: Max 3 re-plans/day to prevent thrashing, but too aggressive a cap suppresses emergence
- **Chatterbox/TTS chunks**: 300 chars max or output is garbled noise
- **Sim venv needs `pyyaml`** — missing it causes all conversation responses to fall back to "I see."

## Key Data Paths

- `db/memories.db` — SQLite database for all agent memories and skills (FTS5 for full-text search)
- `saves/latest_state.json` — auto-resume state (skips re-planning on restart)
- `saves/` — JSON state snapshots (auto-save every 100 ticks)
- `simulation.log` — detailed activity log
- `finetune/data/` — training JSONL files
- `finetune/output*/` — merged model weights and Modelfiles

## Customization Points

- **Add agents**: Add entries to `AGENT_PERSONAS` dict in `personas.py` (fields: name, age, occupation, personality, background, relationships, daily_routine, goals, home_location, work_location, optional special_memory)
- **Add locations**: Add entries to `SMALLVILLE_LOCATIONS` dict in `config.py`
- **Tune memory**: Adjust `MEMORY_RETRIEVAL_WEIGHTS`, `IMPORTANCE_THRESHOLD`, `RECENCY_DECAY_FACTOR` in `config.py`
- **Tune conversations**: Adjust `CONVERSATION_PROBABILITY`, `MAX_CONVERSATION_TURNS`, `CONVERSATION_RELEVANCE_THRESHOLD` in `config.py`
- **Tune performance**: `TICK_DURATION_SECONDS` (default 180 = 3 min game time), `CONVERSATION_CHECK_INTERVAL` (every N ticks), `AGENT_BATCH_SIZE` (concurrent processing)

## Codebase Conventions

- Async throughout — all agent cognition and LLM calls use `async/await`
- No class inheritance hierarchy; agents are flat `GenerativeAgent` instances differentiated by persona data
- Prompt templates live in `prompts.py`, not inline in logic modules
- The simulation starts on 2023-02-13 08:00 (Valentine's Day scenario where Isabella Rodriguez spreads party info through natural conversation)
- GPU queue integration in `llm.py` — routes through `../gpu-queue/queue_manager.py` with Pi 5 fallback. Disable with `--no-gpu-queue`
- Rule-based importance scoring for common observations (keywords in `config.py`) to avoid LLM calls

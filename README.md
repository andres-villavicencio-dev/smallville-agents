# Generative Agents: Reproducing "Interactive Simulacra of Human Behavior" with Small Local Models

A faithful reproduction of the Stanford paper [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — but running entirely on **local small language models** (1B–4B parameters) via Ollama, with no OpenAI dependency.

The twist: instead of one large model, we use a **Committee of Experts** architecture and **Representation Fine-tuning (RFM) neural steering** to give 25 agents distinct personalities from a single 3B model.

## Key Results

### The Valentine's Day Party Experiment

The simulation starts on Feb 13th with a single seed memory: *Isabella Rodriguez is planning a Valentine's Day party at Hobbs Cafe on Feb 14th, 5–7 PM.* The question: can small local models produce emergent social behavior — word spreading, agents replanning their day, and actually attending the party?

| Experiment | Architecture | Party Awareness | Attendees | Key Finding |
|---|---|---|---|---|
| Exp 1 | Single model (qwen2.5:3b) | 100% | N/A | Flawed — measured lunch, not party |
| Exp 2 | Single model + replan cap | 80% (20/25) | 0 | Replan cap (3/day) killed emergence |
| Exp 3 | Committee of Experts | 100% (25/25) | 6 present, 3 deliberate | "Helpers not guests" — agents planned to *fix wiring* at the party, not attend for fun |
| Exp 4 | Committee + fine-tuned experts | Improved | Marginal | Fine-tuning helped but didn't fix core issue |
| **Exp 8** | **RFM Neural Steering** ⭐ | **Organic spread** | **51% replan rate** | **Helpers-not-guests FIXED.** Agents explicitly planned to "Attend Valentine's Day party" |

### The "Helpers Not Guests" Problem

Small models default to *instrumental reasoning* — when told about a party, they plan to inspect wiring, check fire exits, or write safety protocols. Nobody plans to "attend for fun." This is a fundamental limitation of small models that we solved with neural-level personality injection via RFM concept vectors.

## Architecture

### Committee of Experts (Exp 3–4)

Instead of one large model, seven specialized small models handle different cognitive tasks:

| Expert | Model | Role |
|---|---|---|
| Social | `smallville-social` (fine-tuned Gemma 2 2B) | Should agents converse? |
| Dialogue | `smallville-actor` (fine-tuned Gemma 2 2B) | Generate conversation responses |
| Temporal | `gemma3:1b` | Time-aware planning |
| Emotional | `llama3.2:3b` | Emotional context |
| Memory | `gemma3:4b` | Memory synthesis & reflection |
| Spatial | `qwen2.5:3b` | Location-aware decisions |
| Judge | `qwen2.5:3b` | Plan evaluation |

Sequential execution fits within 8GB VRAM (one model loaded at a time).

### RFM Neural Steering (Exp 8) ⭐

A single **QWEN 3 4B** model with **27 personality concept vectors** injected at the neural level via [Representation Fine-tuning Method](https://arxiv.org/abs/2404.03592):

- **27 concepts**: social_warmth, task_focus, creativity, authority, empathy, leadership, curiosity, ambition, humor, patience, assertiveness, etc.
- Each agent gets a unique blend of concept vectors → distinct personality from the same base model
- Results: 3,534 conversations, 588 reflections, 51% replan rate in 5+ hours

### Voice Integration (KittenTTS)

Agents have voices! Each of the 25 agents is mapped to one of 8 KittenTTS Nano voices (15M params, 25MB), running on a Raspberry Pi 5 at 1.6x realtime on CPU:

- Voice endpoint: `http://<pi-ip>:8377/tts`
- Conversations are automatically synthesized to OGG audio when they end
- Gender-matched voice assignment across all 25 agents

## Core Components

```
├── main.py                  # Simulation loop & orchestration
├── agent.py                 # Agent class with memory, planning, reflection
├── committee.py             # Committee of Experts / RFM steering engine
├── conversation.py          # Conversation system with voice hooks
├── environment.py           # Smallville world (20+ locations)
├── memory.py                # Memory stream with SQLite + FTS5
├── reflection_engine.py     # Reflection & conversation evaluation
├── planning_utils.py        # Location snapping, schedule management
├── personas.py              # 25 agent definitions with relationships
├── config.py                # All configuration & location definitions
├── voice_map.py             # Agent → KittenTTS voice mapping
├── voice_integration.py     # TTS generation on conversation end
├── telegram_broadcaster.py  # Live digests to Telegram group
├── webui.py                 # Web UI on port 3000
├── finetune/                # QLoRA training scripts & Modelfiles
│   ├── train.py             # Actor (dialogue) fine-tuning
│   └── train_social.py      # Social expert fine-tuning
├── db/memories.db           # Agent memory database
└── saves/                   # Auto-saved simulation state
```

## The Smallville World

### Locations (20+)

**Residential:**
- Lin Family Home, Moreno Family Home, Moore Family Home, The Willows (apartments)
- Williams Residence, Anderson Residence, Davis Residence, Mayor Residence
- Peterson Cottage, Thompson Residence, Wilson Apartment, Rodriguez Home

**Commercial & Public:**
- Oak Hill College, Hobbs Cafe, The Rose and Crown Pub, Harvey Oak Supply Store
- Johnson Park, Town Hall, Library, Pharmacy

### 25 Agents

Each agent has a name, age, occupation, personality traits, background story, relationships, home/work locations, daily routines, and personal goals. Key characters:

- **Isabella Rodriguez** — Cafe owner, party planner, social hub
- **John Lin** — Pharmacy owner, family man
- **Mrs. Peterson** — Retired teacher, community heart
- **Eddy Lin** — Music student, creative dreamer
- **Mayor Johnson** — Town leader, always politicking
- **Dr. Williams** — Town doctor, mentor figure
- **Frank Wilson** — Maintenance worker (the one who finally attended the party *for fun*)

## Installation

### Prerequisites

```bash
# Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull gemma3:1b gemma3:4b llama3.2:3b qwen2.5:3b

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Fine-tuned Models (optional, for Committee mode)

```bash
# If GGUF files exist in finetune/output*/
ollama create smallville-social -f finetune/output-social/Modelfile
ollama create smallville-actor -f finetune/output/Modelfile
```

### KittenTTS Voice Server (optional)

```bash
# On a Pi 5 or any machine with Python 3.12
uv venv ~/kittentts-venv --python 3.12
source ~/kittentts-venv/bin/activate
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv pip install \
  https://github.com/KittenML/KittenTTS/releases/download/0.8/kittentts-0.8.0-py3-none-any.whl \
  soundfile fastapi uvicorn
python kittentts_server.py  # Runs on port 8377
```

## Running

```bash
# Standard run — Committee of Experts, 25 agents, WebUI
python main.py --committee --num-agents 25 --speed 1000 --webui --webui-port 3000

# Recommended: run in tmux (sim can run for hours)
tmux new-session -d -s sim './venv/bin/python3 main.py --committee --num-agents 25 --speed 1000 --webui --webui-port 3000'

# Monitor
tmux attach -t sim          # Terminal UI
open http://localhost:3000   # Web UI
```

### Key Options

| Flag | Default | Description |
|---|---|---|
| `--committee` | off | Use Committee of Experts (vs single model) |
| `--num-agents N` | 25 | Number of agents |
| `--speed N` | 1000 | Simulation speed multiplier |
| `--webui` | off | Enable web UI |
| `--webui-port N` | 3000 | Web UI port |

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:3b` | Default model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `SMALLVILLE_VOICES` | `1` | Enable/disable voice generation |
| `KITTENTTS_URL` | `http://192.168.1.70:8377/tts` | KittenTTS endpoint |
| `COMMITTEE_MODEL_SOCIAL` | `smallville-social` | Social expert model |
| `DIALOGUE_ACTOR_MODEL` | `smallville-actor` | Dialogue expert model |

## Memory System

### Types
- **Observations** — perceptions of environment and agents
- **Reflections** — high-level insights synthesized from observations
- **Plans** — intended actions and schedules

### Retrieval
Each memory scored by: **α×recency + β×importance + γ×relevance**
- Recency: exponential decay on last access
- Importance: 1–10 scale (LLM-rated or rule-based)
- Relevance: TF-IDF cosine similarity to query

### Reflection
Triggered when cumulative importance exceeds threshold (150). Generates salient questions, retrieves relevant memories, synthesizes insights, stores as high-importance memory.

## Fine-Tuning Pipeline

For training custom experts (QLoRA on Gemma 2 2B):

```bash
cd finetune/
# Edit training data in train.py or train_social.py
python train_social.py  # ~1 min per 50 examples on RTX 3070

# Export: merge → GGUF → Ollama
# (see finetune/ scripts for details)
```

Config: LoRA r=64, alpha=128, dropout=0.05, LR=2e-4, cosine schedule. Requires ~5GB VRAM during training.

## Telegram Broadcasting

Live simulation digests every 10 minutes to a Telegram group:

```bash
python telegram_broadcaster.py  # Requires TELEGRAM_BOT_TOKEN
```

Shows location clusters, active conversations, replans, party tracking, and reflection highlights.

## Hardware

**Tested on:**
- **GPU**: NVIDIA RTX 3070 (8GB VRAM) — runs full 25-agent committee
- **Fallback**: Raspberry Pi 5 (8GB RAM) — Ollama with gemma3:1b + KittenTTS
- **OS**: Ubuntu 24.04

**VRAM budgets:**
- Committee mode: ~2–4 GB (sequential model loading)
- RFM steering: ~1.95 GB (single QWEN 3 4B + vectors)
- KittenTTS: 0 GPU (CPU-only, runs on Pi)

## Lessons Learned

- Small models (1B–4B) default to instrumental reasoning over social reasoning — the "helpers not guests" problem
- Fine-tuning helps but doesn't fix the core issue; neural-level personality steering (RFM) does
- Reflection loops: never count reflections toward their own trigger threshold
- Location capacity limits cause natural overflow to nearby locations
- Committee of diverse small models outperforms a single larger model for multi-faceted agent behavior

## Citation

```bibtex
@article{park2023generative,
  title={Generative Agents: Interactive Simulacra of Human Behavior},
  author={Park, Joon Sung and O'Brien, Joseph C and Cai, Carrie Jun and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S},
  journal={arXiv preprint arXiv:2304.03442},
  year={2023}
}
```

## License

MIT

---

**🏠 Welcome to Smallville — where 25 AI agents live, learn, gossip, and (eventually) attend parties for fun.**

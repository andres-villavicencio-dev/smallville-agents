# AGENTS.md - Agentic Coding Guidelines

This file provides guidance for AI coding agents operating in this repository.

## Project Overview

Implementation of Stanford's "Generative Agents" paper (arXiv:2304.03442) using local Ollama models. Simulates 25 AI agents in "Smallville" with memory, reflection, planning, and natural conversation.

## Build/Lint/Test Commands

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Includes ruff and pytest
```

### Running the Simulation
```bash
python main.py                           # Default: 10x speed, 2 days
python main.py --speed 5 --days 3        # Custom speed/duration
python main.py --no-gpu-queue            # Skip GPU queue, call Ollama directly
python main.py --committee                # Mixture-of-experts mode
python main.py --webui                   # Launch REST API + WebSocket map UI
python main.py --config                   # Print config and exit

# Shell wrapper (checks Ollama health first)
./run.sh [args]

# Override models via env vars
OLLAMA_MODEL=llama3.2:3b python main.py
MODEL_PLANNING=qwen2.5:3b MODEL_CONVERSATION=llama3.2:3b python main.py
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_memory.py -v

# Run a single test function
python -m pytest tests/test_memory.py::test_add_memory -v

# Run tests matching a pattern
python -m pytest tests/ -k "conversation" -v

# Run with verbose output and show print statements
python -m pytest tests/ -v -s
```

### Linting & Formatting
```bash
# Run linter
ruff check .

# Auto-fix lint issues
ruff check . --fix

# Format code
ruff format .

# Check formatting without applying
ruff format . --check
```

---

## Code Style Guidelines

### General Principles
- **Python 3.11+** - Use modern Python features (type hints, match/case, etc.)
- **Async-first** - All agent cognition and LLM calls use `async/await`
- **Flat architecture** - No deep class hierarchies; agents are flat `GenerativeAgent` instances
- **Prompt templates in prompts.py** - Never inline LLM prompts in logic modules
- **Logging** - Use Python's `logging` module for all logging

### Imports
Order imports by category (blank line between groups):
1. Standard library
2. Third-party packages
3. Local project imports

Within each group, sort alphabetically:
```python
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import numpy as np
from rich import print

from agent import GenerativeAgent
from config import USE_COMMITTEE
from memory import MemoryStream
```

### Type Hints
Use modern type hints (Python 3.11+):
```python
# Good
def process_items(items: list[str]) -> dict[str, int]: ...

# Avoid (legacy)
from typing import List, Dict
def process_items(items: List[str]) -> Dict[str, int]: ...

# Use | for unions (Python 3.10+)
def get_value() -> str | None: ...
```

### Naming Conventions
| Element | Convention | Example |
|---------|------------|---------|
| Functions/variables | snake_case | `get_current_plan_item` |
| Classes | PascalCase | `GenerativeAgent` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_RECENT_MEMORIES` |
| Private methods | snake_case with underscore | `_initialize_special_memories` |
| Modules | snake_case | `memory_stream.py` |

### Dataclasses
Use `@dataclass` for simple data containers:
```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ConversationTurn:
    speaker: str
    message: str
    timestamp: datetime
```

### Error Handling
- Use specific exception types
- Log errors at appropriate level
- Provide context in error messages:
```python
try:
    await agent.plan_daily_schedule(current_time)
except Exception as e:
    logger.error(f"Daily planning failed for {agent.name}: {e}")
    raise
```

### Async/Await
- Never use blocking I/O in async functions (use `aiohttp`, `asyncio.to_thread`, etc.)
- Use `asyncio.gather` for concurrent operations:
```python
# Good - concurrent execution
await asyncio.gather(*[agent.think() for agent in agents])

# Avoid - sequential
for agent in agents:
    await agent.think()
```

### Database Operations
- Use parameterized queries to prevent SQL injection
- Close connections properly (use context managers):
```python
with sqlite3.connect(self.db_path) as conn:
    conn.execute("SELECT * FROM memories WHERE agent_name = ?", (name,))
```

### Constants Configuration
All tunable parameters live in `config.py`:
- Memory weights and thresholds
- Simulation timing (tick duration, speed)
- Conversation probability and limits
- Performance tuning (concurrency limits)

### Testing Guidelines
- Use pytest fixtures from `tests/conftest.py`
- Mock LLM calls with `patch_llm` fixture
- Use `AsyncMock` for async methods
- Test files: `tests/test_*.py`

Example test:
```python
@pytest.fixture
def patch_llm(mock_llm_client):
    async def fake_get_client(*args, **kwargs):
        return mock_llm_client
    with patch("agent.get_llm_client", side_effect=fake_get_client):
        yield mock_llm_client

@pytest.mark.asyncio
async def test_agent_observation(test_agent):
    obs = "Saw a cat in the park"
    memory = await test_agent.observe(obs, "Johnson Park")
    assert memory.description == obs
```

---

## Key Files & Directories

| Path | Purpose |
|------|---------|
| `main.py` | Simulation orchestration, tick loop |
| `agent.py` | Agent cognition (plan, observe, reflect) |
| `memory.py` | SQLite + TF-IDF memory stream |
| `environment.py` | 12 locations, agent movement |
| `conversation.py` | Dialogue between agents |
| `llm.py` | Ollama API client |
| `config.py` | All tunable parameters |
| `prompts.py` | LLM prompt templates |
| `db/memories.db` | SQLite memory database |
| `saves/` | Simulation state snapshots |
| `tests/` | pytest test suite |

---

## Common Development Tasks

### Adding a New Agent
Edit `personas.py` - add entry to `AGENT_PERSONAS` dict with fields:
- name, age, occupation, personality, background
- relationships, daily_routine, goals
- home_location, work_location

### Adding a New Location
Edit `config.py` - add entry to `SMALLVILLE_LOCATIONS` dict.

### Tuning Memory
Adjust in `config.py`:
- `MEMORY_RETRIEVAL_WEIGHTS` (recency, importance, relevance)
- `IMPORTANCE_THRESHOLD` (reflection trigger)
- `RECENCY_DECAY_FACTOR`

### Tuning Conversations
Adjust in `config.py`:
- `CONVERSATION_PROBABILITY`
- `MAX_CONVERSATION_TURNS`
- `CONVERSATION_RELEVANCE_THRESHOLD`

---

## Known Issues & Lessons

- **Reflection loops**: Never count reflections toward their own importance threshold
- **Conversation cooldown**: 30-tick cooldown prevents agents from re-entering conversations immediately
- **Re-planning cap**: Max 3 re-plans/day prevents thrashing but suppresses emergence if too aggressive
- **Conversation meta-leak**: Use `_clean_dialogue()` regex to strip LLM meta-commentary

---

## Architecture Notes

### Tick Loop
Each tick advances `TICK_DURATION_SECONDS` (default 180s = 3 min game time):
1. Agent planning
2. Observation generation (for agents at busy locations)
3. Conversation initiation
4. Display update

### Memory Retrieval
Scores: `α×recency + β×importance + γ×relevance`
- Recency: exponential decay (0.99)
- Importance: LLM-rated 1-10
- Relevance: TF-IDF cosine similarity
- Reflection triggers when cumulative importance exceeds 150

### Committee Mode
Enabled with `--committee` flag. Routes decisions through 7 expert models:
- Social, Spatial, Temporal, Emotional, Memory, Dialogue, Judge
- Override via env: `COMMITTEE_MODEL_SOCIAL=other:model`

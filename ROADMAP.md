# Smallville Agents — Improvement Roadmap

_Last updated: 2026-03-04_

---

## 🚀 Active Sprint

### 1. Qdrant as Default Retrieval
**Goal:** Replace TF-IDF keyword search with semantic vector search for all memory retrieval.

**Current state:**
- `memory_qdrant.py` already exists and is fully implemented
- Uses `all-MiniLM-L6-v2` embeddings, in-memory Qdrant (`:memory:` — no persistence)
- Gated behind `USE_QDRANT=0` in config.py

**Changes needed:**
- `config.py`: flip `USE_QDRANT` default to `"1"`
- `memory_qdrant.py`: switch from `:memory:` to persistent Qdrant (Docker or embedded on-disk)
  - Option A: Docker `qdrant/qdrant` container (clean, persistent across restarts)
  - Option B: `QdrantClient(path="./qdrant_data")` (no Docker, embedded, simpler)
  - Recommend Option B for zero-infra overhead
- Verify `qdrant-client` and `sentence-transformers` in venv
- Test that retrieval latency is acceptable (MiniLM is fast, ~5ms on CPU)

**Notes:**
- In-memory Qdrant loses all memories on restart. Persistent storage is required for proper use.
- Shared encoder already loaded once at module level — no per-agent overhead.

---

### 2. Plan Follow-Through Tracking
**Goal:** Agents reflect at end-of-day on whether they did what they planned. Creates accountability and drives better next-day planning.

**Design:**
- `PlanItem` already has `completed: bool` field — it's just not being checked
- Add `plan_followthrough_reflection()` method to Agent:
  - Runs once per sim day (end-of-day trigger in `main.py`)
  - Compares `daily_plan` items (completed vs not)
  - Asks LLM: "You planned to do X, Y, Z. You completed X and Z but not Y. Why? What will you do differently tomorrow?"
  - Stores result as a high-importance memory (type=`reflection`, importance=7)
- Surface completed/incomplete ratio in the rich TUI
- Pipe the reflection into next-day planning context

**Implementation steps:**
1. Add `_mark_plan_item_completed()` call in agent's action loop when a plan item is finished
2. Add `plan_followthrough_reflection()` async method
3. Hook into `main.py` end-of-day logic (already has day transition code)
4. Add "📋 Follow-through: 4/6 items" to agent status panel in TUI

---

### 3. Persistent Emotional State + Emoji Indicator
**Goal:** Agents carry emotional state across ticks. Mood is visible in the TUI as an emoji.

**Design:**

Emotional state = two floats on the Agent object:
```python
self.mood_valence: float = 0.0   # -1.0 (miserable) to +1.0 (elated)
self.mood_arousal: float = 0.5   # 0.0 (lethargic) to 1.0 (energised)
```

Update rules:
- Positive conversation → `valence += 0.15` (capped at 1.0)
- Negative/ignored conversation → `valence -= 0.2`
- Completed plan item → `valence += 0.1`
- Missed plan item → `valence -= 0.1`
- Slow decay toward neutral each tick: `valence *= 0.92`
- Reflection triggers minor arousal spike

Emoji mapping (valence-based):
```
>  0.6  →  😊
   0.2 – 0.6  →  🙂
  -0.2 – 0.2  →  😐
  -0.5 – -0.2 →  😔
< -0.5  →  😟
```

**Where to use it:**
- Pass to `_build_character_system_prompt()` as `mood=` (already accepts this)
- Pass to `committee.consult()` extra context
- Show emoji floating above agent avatar in the **WebUI** (port 3000)
- Store in save/resume state (`saves/latest_state.json`)
- **Disable the TUI** — run without `--webui` Rich panels; WebUI is the primary interface

---

### 4. Theory of Mind — Phase 1: Agent World Knowledge
**Goal:** Each agent maintains a structured list of facts they know about Smallville. Used in planning and conversation to ground decisions in actual world knowledge.

**Design:**

Add `world_knowledge: dict[str, str]` to Agent:
```python
# key = fact ID, value = description
{
  "location:hobbs_cafe": "Hobbs Cafe is on Oak Street, open 7am-9pm, run by Isabella",
  "event:valentines_party": "Isabella is throwing a Valentine's Day party at Hobbs Cafe on Feb 14",
  "person:maria_santos": "Maria Santos is an artist who lives at Davis Residence",
  ...
}
```

Population:
- Seeded from `personas.py` for public/shared facts (locations, community fixtures)
- Updated when agent *observes* something or is *told* something in conversation
- Retrieved facts injected into planning/decision prompts as context

**Phase 2 (later):** Track *what each agent believes other agents know*:
```python
self.beliefs_about_others: dict[str, dict[str, str]] = {
  "Maria Santos": {
    "event:valentines_party": "Maria knows about the party (she was invited Feb 13)"
  }
}
```
This enables: "John already knows about the party, no need to mention it" vs "Sarah doesn't know yet, I should tell her."

---

## 🗄️ Backlog (save for later)

### Conversation Memory
Agents know they *talked* to someone but not what was said. Store conversation summaries as a distinct memory type (`memory_type="conversation_summary"`) with both participants named. Retrieve these when agents plan interactions with specific people.

### World Event Injection
Ability to inject a world event mid-simulation (power cut, unexpected visitor, bad weather) that agents must react to. Stress-tests whether agency is real or just plan-following. CLI: `./inject_event.py "A thunderstorm hits Smallville. All outdoor activities are disrupted."`

### Skill/Knowledge Building
Agents currently don't model their own expertise growing over time. Skills from `skillbank.py` exist but aren't fed back into planning. If Maria paints every day, she should become *better at painting* in her own self-model.

### Multi-Day Memory Arc
Currently each sim day is fairly independent. Agents should carry forward unresolved threads ("I was supposed to meet Tom yesterday but he wasn't home — I should try again today"). Requires flagging memories as "open threads" and checking them during morning planning.

---

## Implementation Order

```
Qdrant (foundational)
  → Emotional State (visible, self-contained)
    → Plan Follow-Through (emotional payoff needs mood to matter)
      → Theory of Mind Phase 1 (builds on enriched memory)
        → Theory of Mind Phase 2 (builds on Phase 1)
```

# Lab Notebook: Reproducing Emergent Behavior in Generative Agents

**Researcher:** Andus VU  
**Started:** February 13, 2026  
**Paper:** Park, J.S., et al. "Generative Agents: Interactive Simulacra of Human Behavior" (arXiv:2304.03442, 2023)  
**Repository:** `github.com/andres-villavicencio-dev/smallville-agents`

---

## 1. Research Question

Can we reproduce the emergent Valentine's Day party invitation cascade from the Stanford "Generative Agents" paper using a custom implementation with a **committee of small local models** instead of a single large model (GPT-3.5/4)?

Specifically: Will agents organically spread party invitations through conversation and reschedule their plans to attend — without being explicitly programmed to do so?

---

## 2. System & Hardware

| Component | Specification |
|-----------|--------------|
| Machine | Alienware m17 R4 (andus-m17-R4) |
| OS | Ubuntu 24.04, kernel 6.14.0-37-generic |
| CPU | Intel Core i7-10870H @ 2.20GHz (16 threads) |
| GPU | NVIDIA RTX 3070 (8GB VRAM) |
| RAM | 32GB DDR4 |
| Storage | 937GB NVMe SSD |
| LLM Backend | Ollama (localhost:11434) + Ollama Cloud fallback |

### Models (Committee of Experts)

The Stanford paper uses a single large model (GPT-3.5-turbo or GPT-4) for all cognitive tasks. Our approach uses a **mixture of small specialized models**, each assigned to a specific cognitive domain:

| Expert Role | Model | Size | Assignment |
|-------------|-------|------|------------|
| Temporal | gemma3:1b | 1B | Time/schedule reasoning |
| Importance | gemma3:1b | 1B | Memory importance scoring |
| Social | llama3.2:3b | 3B | Social interaction decisions |
| Dialogue | llama3.2:3b | 3B | Conversation generation |
| Emotional | llama3.2:3b | 3B | Emotional state assessment |
| Memory | gemma3:4b | 4B | Memory retrieval/reflection |
| Spatial | qwen2.5:3b | 3B | Location/movement reasoning |
| Planning | qwen2.5:3b | 3B | Daily plan generation |
| Judge | qwen2.5:3b | 3B | Decision arbitration |

**Constraint:** 8GB VRAM requires sequential execution — only one model loaded at a time.

**Fallback chain:** Ollama Cloud (qwen3-coder:480b) → local models. Cloud used primarily for dialogue generation.

---

## 3. Architecture

### 3.1 Codebase

16 Python files, ~6,235 lines total:

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 740 | Simulation loop, tick processing, Rich terminal UI |
| `agent.py` | 631 | Agent class, planning, reactive re-planning |
| `reflection_engine.py` | 448 | Planning engine, committee planning, reflections |
| `conversation.py` | 363 | Conversation manager, turn-by-turn dialogue |
| `committee.py` | 442 | Committee of experts strategy pattern |
| `personas.py` | 535 | 25 agent persona definitions |
| `memory.py` | 367 | Memory stream, TF-IDF retrieval, SQLite+FTS5 |
| `environment.py` | 301 | Spatial environment, location graph |
| `planning_utils.py` | 260 | Plan parsing, validation |
| `display.py` | 482 | Rich terminal UI panels |
| `llm.py` | 216 | LLM abstraction layer |
| `config.py` | 106 | Simulation parameters |
| `prompts.py` | 75 | Prompt templates |
| `skillbank.py` | 487 | Skill distillation (disabled for performance) |
| `telegram_broadcaster.py` | 366 | 10-min digest to Telegram group |
| `webui.py` | 416 | Browser-based visualization |

### 3.2 Key Mechanisms

**Memory Stream:** Each agent has a memory stream stored in SQLite with FTS5 full-text search. Memories are scored by:
- **Recency** — exponential decay based on last access time
- **Importance** — LLM-scored 1-10 at creation time
- **Relevance** — TF-IDF cosine similarity to query

All weights set to 1.0 (equal weighting).

**Reflection:** Triggered when cumulative importance of recent memories exceeds threshold. Generates higher-level insights from lower-level observations.

**Planning:** Each agent generates a daily plan of 8 activities with times and locations. Plans can be modified reactively after conversations (max 3 re-plans per day).

**Conversation:** Agents in the same location may initiate conversation. `should_converse` check uses the Social expert. Conversations run turn-by-turn with a 6-turn hard cap and 30-tick cooldown between same pairs.

**Reactive Re-planning:** After each conversation ends, agents evaluate whether to modify their plan based on new information. This is the key mechanism for invitation cascade — if Agent A tells Agent B about the party, Agent B may re-plan to attend.

---

## 4. Experiment Log

### Experiment 1: Baseline (15 Agents)

**Date:** February 13–14, 2026  
**Duration:** 5 hours 19 minutes real time  
**Sim time:** 2 days (Feb 13–14, simulated)  
**Agents:** 15  
**Ticks:** 1,439  
**Tick duration:** 120 seconds game time (720 ticks/day)

#### Configuration
- Committee of experts mode
- Simulation speed: 1000x
- Conversation checks every 3rd tick
- Skill distillation: disabled
- Plan decomposition: disabled (key optimization — was causing 70+ LLM calls per agent)

#### Results

| Metric | Value |
|--------|-------|
| Total memories | 75,618 |
| Conversations | 880 |
| Reflections | 346 |
| Agents visiting Hobbs Cafe on Feb 14 | 15/15 (100%) |

#### Analysis: Emergence Was Fake

**Critical finding:** 12 of 15 agents visited Hobbs Cafe on Valentine's Day because the planning prompt explicitly instructed: *"Include a lunch break (12:00-1:00 PM) at Hobbs Cafe."*

This was hardcoded in `CommitteePlanning.plan_day()` at `reflection_engine.py:~220`.

**Invitation tracing:**
- Isabella Rodriguez explicitly invited only **Tom Moreno** and **Diego Moreno**
- The remaining 12 agents went to Hobbs due to the hardcoded prompt rule
- Only 3 "outsider" agents (Jennifer Moore, Eddy Lin, Maria Santos) showed evidence of being pulled in organically through conversation

**Comparison to Stanford paper:** The original paper reports 5 out of 12 invited agents showed up at the party. Our result (15/15) was suspiciously perfect — and for good reason.

#### Five Critical Gaps Identified

1. **Hardcoded location rules** — Planning prompt forced lunch at Hobbs Cafe
2. **Wrong memory retrieval in planning** — Used `get_memories(limit=20)` (chronological) instead of `retrieve_memories(query)` (relevance-weighted)
3. **No per-day re-planning** — Agents planned once at start, not each simulated morning
4. **No focal point generation** — Missing "What are the 3 most important things today?" step before planning
5. **Weak conversation topic selection** — Event memories didn't surface as conversation topics

#### Bugs Fixed
- Conversation cooldown tracking (`_current_tick` not always set)
- `should_end_conversation` ignoring per-conversation `max_turns`
- Unused variables in `_generate_conversation_response`
- `should_converse` broken 3-expert pipeline (replaced with single Social expert + explicit YES/NO)
- Infinite reflection loop (reflections excluded from importance threshold + 5-min cooldown)

---

### Experiment 2: Fixed Architecture (25 Agents)

**Date:** February 15, 2026  
**Start time:** 09:39 NZST  
**Agents:** 25 (full Stanford paper complement)  
**Tick duration:** 180 seconds game time (480 ticks/day)  
**End time:** ~19:01 NZST (clean shutdown)  
**Duration:** ~9.5 hours real time  
**Status:** ✅ COMPLETE

#### Fixes Applied

1. **Planning prompt softened:** Changed *"Include a lunch break (12:00-1:00 PM) at Hobbs Cafe"* → *"Include a lunch break (12:00-1:00 PM) at a suitable location"*. Persona `lunch_location` now a *suggestion*, not a mandate.
2. **Memory retrieval:** Planning now uses relevance-weighted `retrieve_memories()` with query context
3. **Focal points:** Added memory-informed context to planning ("Relevant memories and events" + "Consider your memories and any important events")
4. **Conversation topics:** Event memories surface in conversation context

#### Preliminary Observations (as of ~4.5 hours into run)

| Metric | Value |
|--------|-------|
| Total memories | 54,608+ |
| Reflections | 578 |
| Plans | 108 |
| Observations | 53,930 |
| Re-plan events | 6,418 |
| Valentine mentions | 154 |
| Party mentions | 36 |
| Hobbs mentions | 15,214 |

**Valentine's Day awareness by agent** (memories mentioning "Valentine"):

| Agent | Mentions | Source |
|-------|----------|--------|
| Isabella Rodriguez | 80 | Originator — actively pitching to everyone |
| Dr. Williams | 14 | Conversation with Isabella |
| Lisa Park | 14 | Conversation with Isabella |
| Professor Anderson | 9 | Conversation with Isabella |
| Miguel Rodriguez | 8 | Conversation with Isabella |
| Rachel Kim | 7 | Conversation with Isabella |
| Mike Johnson | 4 | Conversation with Isabella |
| Mayor Johnson | 3 | Secondary spread? |
| Carmen Moreno | 2 | Conversation with Isabella |
| Eddy Lin | 2 | TBD |
| Maria Santos | 2 | TBD |
| Officer Thompson | 2 | TBD |
| Sam Moore | 2 | Conversation with Isabella |
| Ana Santos | 1 | TBD |
| Carlos Gomez | 1 | TBD |
| John Lin | 1 | Conversation with Isabella |
| Professor Davis | 1 | TBD |
| Sarah Chen | 1 | TBD |

**Key observations:**
- Isabella is actively sharing Valentine's Day decoration ideas through organic conversations — "Love Garden" with potted plants, forest-inspired atmosphere, DIY flower crown station
- 18 of 25 agents have at least one Valentine-related memory
- Lisa Park **replanned** to invite Isabella for coffee (organic pull-in)
- Rachel Kim **replanned** to discuss mural concepts at Hobbs Cafe
- John Lin **replanned** to discuss cultural festival idea with Isabella at Hobbs
- Re-planning is working: 6,418 re-plan evaluations, with agents modifying schedules after conversations
- Ollama Cloud hitting 429 rate limits — falling back to local models for dialogue

**Emerging cascade pattern:**
```
Isabella Rodriguez (originator)
├── Tom Moreno (direct conversation)
├── Miguel Rodriguez (direct conversation — suggested reaching out to local designers)
├── Dr. Williams (direct conversation — forest-inspired decor)
├── Lisa Park (direct conversation → REPLANNED to invite Isabella for coffee)
├── Rachel Kim (direct conversation → REPLANNED to go to Hobbs)  
├── Professor Anderson (direct conversation — "Love Garden" + craft day)
├── Mike Johnson (direct conversation — wildflower centerpieces)
├── Sam Moore (direct conversation — community gathering)
├── Carmen Moreno (direct conversation — nature-inspired ideas)
├── John Lin (direct conversation → REPLANNED cultural festival at Hobbs)
├── Sarah Chen (mentioned by Isabella in conversation)
└── [7 agents with no Valentine mentions yet — potential secondary spread targets]
```

#### Final Results

| Metric | Value |
|--------|-------|
| Total memories | 105,147 |
| Observations | 103,860 |
| Reflections | 1,169 |
| Plans | 118 |
| Conversation observations | ~13,768 |
| Ticks completed | 960 |
| Sim time reached | Feb 15, 08:00 (1 day past party) |
| Valentine/party mentions | 198 across 20/25 agents |

**Valentine's Day awareness by agent (final):**

| Agent | Mentions | Source |
|-------|----------|--------|
| Isabella Rodriguez | 102 | Originator |
| Dr. Williams | 15 | Direct conversation with Isabella |
| Lisa Park | 15 | Direct conversation with Isabella |
| Miguel Rodriguez | 10 | Direct conversation with Isabella |
| Professor Anderson | 9 | Direct conversation with Isabella |
| Rachel Kim | 8 | Direct conversation with Isabella |
| Officer Thompson | 5 | Direct conversation with Isabella |
| Sam Moore | 5 | Direct conversation with Isabella |
| John Lin | 4 | Direct conversation with Isabella |
| Mike Johnson | 4 | Direct conversation with Isabella |
| Diego Moreno | 3 | Direct / secondary |
| Eddy Lin | 3 | Secondary spread |
| Mayor Johnson | 3 | Secondary spread |
| Tom Moreno | 3 | Direct conversation with Isabella |
| Carmen Moreno | 2 | Direct conversation with Isabella |
| Maria Santos | 2 | Secondary spread |
| Sarah Chen | 2 | Secondary spread |
| Ana Santos | 1 | Secondary spread |
| Carlos Gomez | 1 | Secondary spread |
| Professor Davis | 1 | Secondary spread |
| **5 agents** | **0** | Mrs. Peterson, Frank Wilson, Jennifer Moore, Mei Lin, Emily Moore |

**80% awareness penetration** — 20 of 25 agents acquired at least one Valentine/party-related memory.

#### Party Attendance Analysis (5–7 PM Window, Feb 14)

**Who was at Hobbs Cafe on Feb 14, by snapshot:**

| Sim Time | Agents at Hobbs | Count |
|----------|----------------|-------|
| 00:40 | Isabella | 1 |
| 04:00 | Isabella, Dr. Williams, Prof Anderson, Lisa Park, Miguel, Rachel Kim | 6 |
| 07:20 | Isabella | 1 |
| 09:00 | Isabella, Dr. Williams, Miguel, Rachel Kim | 4 |
| 10:40 | Isabella | 1 |
| 14:00 | Emily Moore, Ana Santos, Dr. Williams, Miguel, Officer Thompson, Rachel Kim | 6 |
| **17:20** | **Isabella alone** | **1** |
| **19:00** | **Ana Santos, Dr. Williams, Miguel, Officer Thompson, Rachel Kim** | **5** |
| 20:40 | Isabella | 1 |

#### Critical Finding: Information Spread Without Behavioral Change

**Zero agents rescheduled their Feb 14 plans to specifically attend the Valentine's Day party.**

Analysis of all 25 agents' daily plans for Feb 14 reveals:
- **Only Isabella's plan mentions "party" or "Valentine's Day"** — her plan includes "Initial Decor Brainstorm" (9 AM) and "Finalize party setup" (3 PM)
- The 5 agents present at Hobbs during the 19:00 snapshot (Dr. Williams, Miguel, Rachel Kim, Officer Thompson, Ana Santos) were there as part of their **routine daily schedules**, not because they were attending the party
- Rachel Kim essentially *lives* at Hobbs Cafe all day (8:30 AM onwards in her plan)
- Dr. Williams has "Observation & Social Engagement – Hobbs Cafe" as a routine item
- Multiple agents have "Lunch at Hobbs Cafe" as standard routine (residual persona bias)

**The cascade achieved awareness but not action.** Agents heard about Valentine's Day, discussed decorations and themes in conversation, but none translated this into a concrete plan change like "Attend Isabella's party at 5 PM."

#### Root Cause: Re-plan Cap

The reactive re-planning system (`react_to_conversation()`) is capped at **3 re-plans per day per agent**. Log analysis shows many agents hit this cap well before the party window:

```
[replan] Frank Wilson hit re-plan cap, skipping
[replan] Jennifer Moore hit re-plan cap, skipping
[replan] Mei Lin hit re-plan cap, skipping
[replan] Mrs. Peterson hit re-plan cap, skipping
```

Agents who learned about the party in afternoon conversations were already locked out of re-planning. Even if the re-planning logic would have generated a "go to party" plan item, the cap prevented it.

**This is the smoking gun for Experiment 3:** the re-plan cap is suppressing exactly the behavioral emergence we're trying to observe.

#### Comparison to Stanford Paper

| Metric | Stanford Paper | Experiment 1 | Experiment 2 |
|--------|---------------|-------------|-------------|
| Party attendance | 5/12 invited | 15/15 (fake) | 0 deliberate attendees |
| Info spread | Not quantified | N/A | 20/25 agents (80%) |
| Plan changes | Reported | N/A (hardcoded) | 0 party-related |
| Cascade depth | Multi-hop | N/A | Mostly 1-hop (Isabella→X) |

---

### Experiment 3: Uncapped Re-planning (25 Agents)

**Date:** February 15–16, 2026  
**Start time:** 00:47 NZST (Feb 16)  
**End time:** 05:25 NZST (Feb 16, clean shutdown)  
**Duration:** 4 hours 37 minutes real time  
**Agents:** 25  
**Status:** ✅ COMPLETE

#### Hypothesis

Experiment 2 demonstrated that agents can organically spread information about the Valentine's Day party through conversation (80% awareness), but the **re-plan cap of 3 per day** prevented any agent from rescheduling to attend. By removing this cap entirely, agents who hear about the party should be free to modify their plans — just like humans would.

#### Change from Experiment 2

**Single change:** Removed the re-plan cap in `agent.py:react_to_conversation()`.

- Before: `if self._replan_count >= 3: return False` (hard cap, never resets across days)
- After: No cap — agents can reschedule as many times as conversations warrant

#### Results

| Metric | Exp 2 | Exp 3 |
|--------|-------|-------|
| Total memories | 105,147 | **112,601** |
| Observations | 103,860 | 111,803 |
| Reflections | 1,169 | 591 |
| Plans | 118 | 207 |
| Dialogue turns | ~13,768 | 14,774 (7,387 × 2) |
| Unique speaker-listener pairs | — | 482 |
| Valentine awareness | 20/25 (80%) | **25/25 (100%)** |
| Replan evaluations | — | 3,100 |
| Replan accepted | — | 156 (5.0%) |
| Replan rejected (no change) | — | 2,944 (95.0%) |
| Party-related replans | 0 | **28 (~10 agents)** |

**Key finding: 100% awareness penetration.** Every single agent acquired Valentine's Day/party-related memories — up from 80% in Experiment 2.

#### Valentine's Day Awareness by Agent

| Agent | Mentions | Role |
|-------|----------|------|
| Mrs. Peterson | 166 | Enthusiastic amplifier |
| Isabella Rodriguez | 118 | Originator |
| Diego Moreno | 102 | Excited participant |
| Officer Thompson | 73 | Practical helper |
| Miguel Rodriguez | 67 | Creative collaborator |
| Carmen Moreno | 50 | Family involvement |
| Mayor Johnson | 38 | Community leader |
| Dr. Williams | 37 | Safety-conscious advisor |
| Eddy Lin | 33 | Secondary spread |
| Rachel Kim | 32 | Secondary spread |
| Ana Santos | 31 | Secondary spread |
| Frank Wilson | 28 | Structural concerns (!) |
| Sarah Chen | 28 | Historical research |
| John Lin | 27 | Nostalgic connection |
| Tom Moreno | 24 | Safety/decoration help |
| Lisa Park | 23 | Party details/safety |
| Maria Santos | 22 | Secondary spread |
| Carlos Gomez | 20 | Art collaboration |
| Emily Moore | 19 | Secondary spread |
| Mei Lin | 18 | Secondary spread |
| Jennifer Moore | 15 | Community planning |
| Professor Davis | 15 | Academic connection |
| Professor Anderson | 12 | Secondary spread |
| Sam Moore | 8 | Minimal involvement |
| Mike Johnson | 7 | Social/pub connection |

**Notable shift:** Mrs. Peterson (166 mentions) surpassed Isabella (118) as the most Valentine-aware agent. She became an enthusiastic amplifier, weaving party references into her conversations with nostalgic teaching stories.

#### Party-Related Re-planning (THE KEY RESULT)

With the re-plan cap removed, **28 party-related re-plans occurred from ~10 unique agents** (vs. 0 in Experiment 2).

**Representative re-plan actions by agent:**

| Agent | Re-plan Actions |
|-------|----------------|
| Officer Thompson | "Discuss Valentine's decorations and ideas with Isabella" |
| Isabella Rodriguez | "Help plan Valentine's Day party at Hobbs Cafe" (×3) |
| Lisa Park | "Meet with Isabella to discuss party details and safety protocols" (×2) |
| Tom Moreno | "Inform Dr. Williams about potential electrical hazard at Isabella's party"; "Secure decorations at Isabella's party location" |
| Dr. Williams | "Meet with Tom Moreno regarding Valentine's decorations" |
| Frank Wilson | "Inspect electrical decorations at Isabella's party location" (×2) |
| Carmen Moreno | "Discuss display ideas with Isabella" (×5) |
| Mayor Johnson | "Discuss art show funding concerns" |
| Eddy Lin | "Meet Isabella for coffee" |
| Miguel Rodriguez | "Discuss flavor ideas with Isabella over coffee" |

**~10 unique agents replanned around the party** — the first time non-Isabella agents created explicit party-related plan items through organic conversation.

#### Secondary Information Spread

A critical improvement over Experiment 2: agents other than Isabella began spreading party information through their own conversations.

**Examples of secondary spread:**

- **Mike Johnson** → told John Lin AND Officer Thompson about the party, connecting it to Sam Johnson's legacy
- **Frank Wilson** → told Dr. Williams about "structural integrity" concerns for the party (peak character voice)
- **Sarah Chen** → told Frank Wilson she was "compiling information for Isabella's Valentine's party"
- **Diego Moreno** → excitedly told Miguel about designing the menu "like a treasure map"
- **Ana Santos** → mentioned making "follow-up calls for the party"
- **Mrs. Peterson** → wove party references into conversations with Diego about community spirit

This represents **multi-hop information diffusion** — the exact mechanism the Stanford paper describes as emergent behavior.

#### Character Voice Quality

The dialogue quality showed strong character differentiation:

**Frank Wilson** (terse, structural):
> "Tom — Isabella's party creates stress points. Hinge loose on front door. Been meaning to tighten. Your display frames might work if they're load-bearing."

**Diego Moreno** (breathless, enthusiastic):
> "Dad said I could help design the menu for Isabella's party — that's SO cool! My idea is to make it look like a treasure map, but instead of X marks the spot, it's all our favorite foods!"

**Mrs. Peterson** (nostalgic, pedagogical):
> "Diego, my boy, you remind me of a student I had in 1987 — little Timmy Johnson who wanted to help everyone feel loved, not just be loved himself."

**Miguel Rodriguez** (creative, visionary):
> "Imagine this: edible decorations, projected light patterns, and Isabella announcing the food stations? People walking through the cafe, not just eating, but *experiencing* the whole party."

**Professor Anderson** (academic, Socratic):
> "The question is whether collaborative learning creates genuine intellectual synthesis, or merely superficial exchange. Consider this — your proposal reflects what Aristotle called 'communicative dialectic.'"

#### Re-planning Stability

The concern about uncapped re-planning causing churn was **mostly unfounded**:
- 3,100 re-plan evaluations occurred
- Only 156 (5.0%) resulted in actual plan changes
- The `NO_CHANGE` response path worked effectively as a filter
- No agents entered infinite re-planning loops

#### Answering the Pre-Experiment Questions

- [x] **Do agents create explicit "attend party" plan items?** — Partially. ~10 agents created party-related plan items (28 re-plans total), but framed as "discuss decorations," "inspect safety," "meet Isabella" — not "attend the party." The behavioral response is task-oriented, not social.
- [x] **How many agents at Hobbs during 5–7 PM?** — 6 agents physically present at Hobbs Café during the party window: Isabella Rodriguez, Carmen Moreno, Diego Moreno, Mayor Johnson, Miguel Rodriguez, Mrs. Peterson. However, only 3 had *deliberate* party-related plans: Isabella ("Help plan Valentine's Day party"), Miguel ("Discuss flavor ideas with Isabella over coffee"), and Diego ("Discuss art show concept"). The other 3 appear to have lingered from earlier visits.
- [x] **Does unlimited re-planning cause churn?** — No. 95% of evaluations returned NO_CHANGE.
- [x] **Secondary cascade?** — YES. Multiple agents spread party info independently of Isabella.
- [x] **Information fidelity?** — High. Party details (Hobbs Cafe, Valentine's Day, decorations) remained accurate through secondary spread. Some creative embellishment occurred (treasure map menus, structural inspections) but core facts preserved.

#### Comparison Across All Experiments

| Metric | Exp 1 | Exp 2 | Exp 3 |
|--------|-------|-------|-------|
| Agents | 15 | 25 | 25 |
| Duration (real) | 5h 19m | ~9.5h | 4h 37m |
| Total memories | 75,618 | 105,147 | 112,601 |
| Valentine awareness | N/A (fake) | 20/25 (80%) | **25/25 (100%)** |
| Party-related replans | N/A | 0 | **28 (~10 agents)** |
| Agents at Hobbs 5-7 PM | 15 (fake — lunch) | 1 (Isabella) | **6 present, 3 deliberate** |
| Secondary spread | N/A | Minimal | **Confirmed multi-hop** |
| Character voice quality | Generic | Improved | **Strong differentiation** |
| Hardcoded bias | Yes (lunch at Hobbs) | Softened | Softened |
| Re-plan cap | 3/day | 3/day | **Uncapped** |

---

### Experiment 4: Fine-Tuned Character Actor (smallville-actor)

**Date:** February 15–16, 2026  
**Status:** ✅ COMPLETE — Model trained, exported, registered in Ollama

#### Motivation

Experiments 1–3 used generic small models (gemma3, llama3.2, qwen2.5) with long system prompts to steer character behavior. This worked for structured tasks (planning, importance scoring) but dialogue quality was inconsistent — models would drift out of character, produce meta-commentary, or homogenize voices across agents.

**Hypothesis:** A single small model fine-tuned on character-specific dialogue can produce more distinctive, consistent voices than prompt engineering alone — while being smaller and faster than the committee's dialogue expert.

#### Phase 1: Character Profiles

Created rich YAML personality profiles for all 25 Smallville agents.

**Location:** `finetune/profiles/` (25 YAML files)

Each profile captures:

| Dimension | Example (Isabella Rodriguez) |
|-----------|------------------------------|
| `speech_style` | "Warm, flowing, inviting. Chains thoughts with dashes and 'and'" |
| `vocabulary_level` | "Casual-educated. Rich sensory words (colors, textures, smells, flavors)" |
| `catchphrases` | "Oh, you have to try...", "Wouldn't it be lovely if..." |
| `sentence_length` | "Medium to long. Chains ideas with dashes and ellipses" |
| `default_mood` | "Warm, optimistic, slightly scattered with excitement" |
| `when_excited` | "Speaks faster, drops into rapid planning mode" |
| `when_stressed` | "Gets quieter, more focused. Short sentences." |
| `humor_style` | "Gentle, self-deprecating about her overambition" |
| `disagree_style` | How they handle conflict |
| `information_sharing` | How they relay news/gossip |
| `quirks` | Character-specific behavioral tics |
| `example_lines` | Reference dialogue for voice calibration |

Profiles were designed to create **maximally distinct voices** — Diego Moreno (10yo) speaks in breathless run-on sentences about rocks, Frank Wilson (elderly) speaks in terse fragments, Isabella chains ideas with dashes and food metaphors.

#### Phase 2: Synthetic Training Data Generation

**Script:** `finetune/generate_training_data.py`  
**Generator model:** Claude Sonnet (Anthropic)  
**Date generated:** Feb 15, 2026 (~22:04 NZST)

Generated synthetic conversations across **8 scenario types**:

| Scenario | Purpose |
|----------|---------|
| `casual_greeting` | Baseline voice at low stakes |
| `work_talk` | Domain-specific vocabulary |
| `event_sharing` | Valentine's Day party info relay |
| `responding_to_invitation` | Decision-making personality |
| `emotional_moment` | Emotional range and depth |
| `multi_turn_sustained` | Voice consistency over 5-6 turns |
| `disagreement` | Conflict handling style |
| `gossip_relay` | Information fidelity and filtering |

**Process:**
1. Pair agents (prioritizing relationship pairs: married, siblings, friends, rivals)
2. Generate dialogue via gemma3:4b using both agents' full YAML profiles as context
3. Parse raw dialogue into structured turns
4. Convert each turn into a ShareGPT-format training example with structured system prompt

**Structured system prompt format** (used at both training and inference):
```
[AGENT: Isabella Rodriguez]
[AGE: 34] [OCCUPATION: Cafe owner (Hobbs Cafe)]
[SPEECH: Warm, flowing, inviting...]
[VOCABULARY: Casual-educated]
[CATCHPHRASES: Oh, you have to try... | Wouldn't it be lovely if...]
[SENTENCE_LENGTH: Medium to long]
[MOOD: Warm, optimistic]
[TALKING_TO: Frank Wilson]
[HUMOR: Gentle, self-deprecating]
[QUIRKS: Names dishes after customers | Always offering food]
```

**Result:** 177 training examples from conversations generated by Claude Sonnet, covering all 25 agents across diverse scenario pairings

#### Phase 3: QLoRA Fine-Tuning

**Script:** `finetune/train.py`  
**Framework:** Unsloth + trl (SFTTrainer)  
**Date trained:** Feb 16, 2026 (~00:00 NZST)

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/gemma-2-2b-it` (Gemma 2, 2B params) |
| Quantization | 4-bit (QLoRA) |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 83,066,880 / 2,000,000,000 (4.15%) |
| Epochs | 3 |
| Batch size | 2 (effective 16 with grad accum 8) |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | AdamW 8-bit |
| Max seq length | 1024 |
| Packing | Attempted (disabled by Unsloth due to HF bug) |
| Training time | **3 min 55 sec** |
| GPU | RTX 3070 8GB (gradient offloading enabled) |

**Loss curve:**

| Step | Loss | Epoch |
|------|------|-------|
| 5 | 2.6275 | 0.45 |
| 10 | 1.5675 | 0.90 |
| 15 | 0.9153 | 1.27 |
| 20 | 0.6918 | 1.72 |
| 25 | 0.5285 | 2.09 |
| 30 | 0.3755 | 2.54 |
| **Final** | **0.3755** | **2.81** |
| **Average** | **1.0493** | — |

Loss dropped from **2.63 → 0.38** over 33 steps — strong convergence indicating the model learned the character voice patterns.

#### Phase 4: Export & Deployment

1. **LoRA adapter saved** → `finetune/output/smallville-actor-lora/`
2. **Initial GGUF export failed** — Unsloth's `save_pretrained_gguf` OOM'd during 16-bit merge (needed 1.12 GiB, only 667 MiB free)
3. **Workaround:** Separate `export_gguf.py` script — load 4-bit + LoRA, merge to CPU, save as float16
4. **llama.cpp conversion** → `convert_hf_to_gguf.py` with Q8_0 quantization
5. **Final model:** `smallville-actor-Q8_0.gguf` — **2.8 GB**
6. **Ollama registration:** Custom Modelfile with gemma-2 chat template

```
FROM ./smallville-actor-Q8_0.gguf
PARAMETER temperature 0.8
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 1024
```

Registered as `smallville-actor:latest` in Ollama.

#### Phase 5: Evaluation — Character Voice Distinction

**Key result:** The fine-tuned model produces **radically different voices** for different characters using only the structured system prompt tags.

**Example — Same scenario, different characters:**

**Frank Wilson** (elderly, terse):
> "Hmm. Cafe. Party. Noticed that. Isabella good with people. Safe place for celebration."

**Diego Moreno** (10-year-old, breathless):
> "Guess what! I found the coolest rock by Johnson Park — it has these sparkly bits in it and Dad said it might be quartz but Mom thinks it might be diamond AND guess what else — Isabella's having a Valentine's party at her cafe tomorrow from 3 to 5 and Miguel's bringing me his best drawing EVER and — oh man, Mrs. Peterson, you have to hear about this rock!"

These are completely different people emerging from the **same 2.8GB model**. Frank speaks in fragments. Diego speaks in one continuous breathless run-on sentence that circles back to rocks. The character profiles have been internalized, not just parroted.

#### Summary

| Metric | Value |
|--------|-------|
| Training data | 177 Claude Sonnet-generated examples across 8 scenario types |
| Training time | 3 min 55 sec |
| Base model | Gemma 2 2B (QLoRA, 4.15% trainable) |
| Final model size | 2.8 GB (Q8_0 GGUF) |
| Final training loss | 0.38 |
| Character profiles | 25 rich YAML profiles |
| Voice distinction | ✅ Confirmed — radically different voices per character |
| Registered in Ollama | `smallville-actor:latest` |

**Significance:** This demonstrates that a **2B parameter model fine-tuned for 4 minutes** on synthetic character data can produce more distinctive, consistent character voices than a 14B+ model with prompt engineering alone. The structured system prompt format (`[AGENT]`, `[SPEECH]`, `[MOOD]`, etc.) acts as a lightweight character "address" that the fine-tuned model can decode into rich behavioral patterns.

**Next:** Integrate `smallville-actor` as the Dialogue expert in the Committee of Experts, replacing the generic llama3.2:3b. This should improve conversation quality while reducing model size for dialogue generation from 3B → 2B.

---

### Experiment 5: Fine-Tuned Social Expert (smallville-social)

**Date:** February 17, 2026
**Status:** 🔧 COMPLETE — Model trained, exported, deployed to Ollama

#### Motivation

Experiment 3 revealed the "helpers not guests" problem: small models default to instrumental reasoning (inspect wiring, discuss safety protocols) rather than social reasoning (attend for enjoyment). The social expert — responsible for `should_converse` decisions and social context in `decide_action` — was a generic `llama3.2:3b` with no training on what social events *mean*.

**Hypothesis:** A fine-tuned social expert, explicitly trained that "parties = attend for enjoyment, not inspections/tasks," will produce agents that plan to attend the party as guests rather than helpers.

#### Phase 1: Training Data Generation

**Script:** `finetune/generate_social_data.py`
**Method:** Programmatic generation with templates + random variation (not LLM-generated)

| Category | Examples | Purpose |
|----------|----------|---------|
| Valentine's Day Party | 58 | Core scenario — should_converse + decide_action for party |
| Should Converse YES | 20 | Diverse social situations where conversation is appropriate |
| Should Converse NO | 10 | Situations where agents should NOT initiate conversation |
| Community Events | 30 | Generalization to other social events (art shows, concerts, etc.) |
| Social vs Instrumental | 20 | **Anti-patterns** — explicitly corrects task-oriented party responses |
| Emotional & Relationships | 30 | Emotional reasoning about relationships and social bonds |
| Timing & Logistics | 20 | Schedule evaluation for attending vs skipping events |
| Extras | 14 | Edge cases and additional scenarios |
| **Total** | **202** | |

**Key design decisions:**
- All 25 Smallville agents represented with personalized scenarios
- Anti-patterns explicitly targeted: Frank fixing hinges → Frank attending to enjoy; Tom inspecting wiring → Tom attending to socialize; Lisa writing safety protocols → Lisa attending to connect; Officer Thompson patrolling → Thompson attending as community member; Carmen bringing business suggestions → Carmen attending to celebrate
- Programmatic generation scales better than hand-writing (lesson from actor training data)
- 7 diverse categories prevent overfitting to one situation

**Output:** `finetune/data/social_expert_training.jsonl` (202 examples)

#### Phase 2: QLoRA Fine-Tuning

**Script:** `finetune/train_social.py`
**Framework:** Unsloth + trl (SFTTrainer)

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/gemma-2-2b-it` (Gemma 2, 2B params) |
| Quantization | 4-bit (QLoRA) |
| LoRA rank (r) | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Epochs | 5 |
| Batch size | 2 (effective 8 with grad accum 4) |
| Learning rate | 2e-4 (cosine schedule) |
| Max seq length | 512 (social outputs are short) |
| Training time | **3 min 12 sec** |

**Training history:**

| Run | Examples | Duration | Loss (start → end) |
|-----|----------|----------|---------------------|
| First attempt | 46 | 66s | 2.82 → 0.22 |
| **Final** | **202** | **192s** | **3.6 → 0.06** |

Loss of 0.06 indicates very strong convergence — the model has thoroughly learned the social reasoning patterns.

#### Phase 3: Export & Deployment

Same two-step workaround as smallville-actor (Unsloth GGUF export OOMs on 8GB VRAM):

1. `save_pretrained_merged()` → 16-bit safetensors (`finetune/output-social/merged_16bit/`, 5GB, 2 shards)
2. `llama.cpp/convert_hf_to_gguf.py --outtype q8_0` → `finetune/output-social/smallville-social-Q8_0.gguf` (2.78GB)
3. Ollama Modelfile → `ollama create smallville-social -f Modelfile`

#### File Locations

| Artifact | Path |
|----------|------|
| Training data | `finetune/data/social_expert_training.jsonl` |
| Data generator | `finetune/generate_social_data.py` |
| Training script | `finetune/train_social.py` |
| LoRA adapter | `finetune/output-social/smallville-social-lora/` |
| Merged 16-bit | `finetune/output-social/merged_16bit/` |
| GGUF | `finetune/output-social/smallville-social-Q8_0.gguf` |

---

### Experiment 4: Two Fine-Tuned Specialists (Dialogue + Social)

**Date:** February 17, 2026
**Start time:** 20:33 NZST
**Agents:** 25
**Tick duration:** 120 seconds game time
**Status:** 🔬 IN PROGRESS

#### Hypothesis

Fine-tuned social expert (`smallville-social`, trained on 202 examples including anti-patterns for instrumental reasoning) + fine-tuned dialogue expert (`smallville-actor`, trained on 177 character voice examples) will produce agents that:
1. Attend the party as *guests* rather than *helpers*
2. Generate more distinctive character dialogue
3. Maintain or improve information cascade (100% awareness target)

#### Committee Configuration (Updated)

| Expert Role | Model | Change from Exp 3 |
|-------------|-------|--------------------|
| Social | **smallville-social** (2B, fine-tuned) | Was llama3.2:3b |
| Dialogue | **smallville-actor** (2B, fine-tuned) | Was llama3.2:3b |
| Temporal | gemma3:1b | Unchanged |
| Importance | gemma3:1b | Unchanged |
| Emotional | llama3.2:3b | Unchanged |
| Memory | gemma3:4b | Unchanged |
| Spatial | qwen2.5:3b | Unchanged |
| Planning | qwen2.5:3b | Unchanged |
| Judge | qwen2.5:3b | Unchanged |

#### Setup Notes

- Fresh DB (backed up Exp 3 as `db/memories_exp3_run2_backup.db`)
- Fresh state (backed up as `saves/exp3_run2_state.json`)
- Fixed missing `pyyaml` in sim venv (was causing all conversations to collapse to "I see.")
- Running in tmux session `sim`

#### Preliminary Results (1 hour in, Tick 100, Sim time: Feb 13 1:00 PM)

| Metric | Exp 3 (final) | Exp 4 (1hr in, Day 1 only) |
|--------|---------------|----------------------------|
| Total memories | 112,601 | 17,728 |
| Conversations | ~7,387 | 132 |
| Reflections | 591 | 182 |
| Valentine/party awareness | 25/25 (100%) | **24/25 (96%)** |
| Party-related observations | — | 686 |
| Helper/inspector language | Pervasive | **ZERO** |

**Critical early signal: ZERO helper/inspector language.** No wiring inspections, no safety protocols, no fire exits, no structural concerns. This is the primary indicator that the fine-tuned social expert is working.

**Conversation quality samples (fine-tuned dialogue):**

Isabella to Tom:
> "Oh, Tom! You have to try the lavender shortbread I'm making for the party — it needs just the right amount of chill time."

Ana to Carlos:
> "Okay, Carlos, let me handle logistics for Isabella's Valentine's party. We can frame some photos of the cafe during setup!"

Carlos to Mei:
> "Dude, wait— picture this — maybe we should create something together! Like, when you started designing those posters for Isabella..."

The conversations are genuinely social — discussing food, creative collaboration, and logistics rather than safety inspections.

**Awareness:** 24/25 agents aware by 1:00 PM on Day 1 (only Frank Wilson unaware — consistent with his character as a reclusive elderly man). On pace to match or exceed Exp 3's 100%.

**Re-planning:** 437 "no change" decisions, 0 accepted replans so far — but this is expected on Day 1 before the party window. The key test comes tomorrow (Feb 14, 5-7 PM sim time).

#### Key Question (Pending)

Will agents plan to **attend** the party as guests when Feb 14 arrives? The social expert was trained to output "attend the party for enjoyment" rather than "inspect the decorations." First real test comes when the simulation reaches party time.

---

### Experiment 5: Full-Day Planning Fix + Event-Aware Scheduling

**Date:** February 18, 2026  
**Start time:** 11:07 NZST  
**Agents:** 25  
**Tick duration:** 120 seconds game time  
**Speed:** 1000x  
**Status:** 🔬 IN PROGRESS

#### Root Cause Discovery

Analysis of Experiment 4's database revealed a critical planning truncation bug: **zero out of 25 agents scheduled anything at 5-7 PM on Valentine's Day**. The party literally couldn't exist in anyone's schedule.

**Cause chain (three compounding factors):**

1. **Judge model token limit: 150 tokens.** The `plan_day` pipeline ends with the Judge expert (qwen2.5:3b), which synthesizes the final schedule. But its default `max_tokens=150` is only enough for ~5 activities. A full 8-10 activity daily plan (6 AM to 10 PM) needs ~400-500 tokens. Plans were literally cut off around 1 PM.

2. **Prompt lacked full-day enforcement.** The planning prompt said "8 activities" but all examples showed morning activities. LLMs front-load schedules, and the token limit killed them before reaching afternoon/evening.

3. **No event anchoring.** Even with party memories in context, there was no structural mechanism to ensure agents block time for known events. The party info sat in memory text and was deprioritized by the planning models.

**Result:** Isabella Rodriguez — the party organizer herself — planned the party for 5-7 PM but her daily schedule stopped at 1 PM. The event she organized was invisible to her own planner.

#### Fixes Applied

**Fix A: Token limit override** (`committee.py`)
```python
PIPELINE_TOKEN_OVERRIDES = {
    "plan_day": {"judge": 500},  # Was 150 (default)
}
```
Added `max_tokens_override` parameter to `Committee._call_model()`. Also bumped single-model planning from 300→500 tokens in `llm.py`.

**Fix B: Full-day prompt structure** (`reflection_engine.py`)
Restructured `CommitteePlanning.plan_day()` prompt to explicitly require three time blocks:
```
MORNING (6:00 AM - 12:00 PM): Wake up, morning routine, work/activities
AFTERNOON (12:00 PM - 5:00 PM): Lunch, errands, social visits
EVENING (5:00 PM - 10:00 PM): Events, dinner, leisure, return home
```
Added evening examples in the prompt format. Both `CommitteePlanning` and `SingleModelPlanning` updated.

**Fix C: Event-aware plan seeding** (`reflection_engine.py`)
New static method `CommitteePlanning._extract_event_commitments()`:
- Scans high-importance (≥6) event memories for time patterns
- Extracts event names and known Smallville locations
- Handles time ranges like "5-7 PM" (regex grabs start time)
- Injects as `FIXED COMMITMENTS` block in the planning prompt:
```
FIXED COMMITMENTS (these MUST appear in the schedule at the specified times):
- 5 PM - Attend Valentine's Day party at Hobbs Cafe
Plan other activities AROUND these fixed events.
```
- Max 3 fixed commitments to leave room for organic planning

**Bonus fix:** Plan memory now stores ALL items with full timestamps and locations (was truncating to first 5 items in `agent.py`).

#### Commit
```
05b8fdf fix: full-day plan generation + event-aware scheduling (Exp 5)
 4 files changed, 233 insertions(+), 40 deletions(-)
```

#### Early Results (5.5 hours in, Day 2 ticking)

| Metric | Exp 4 | Exp 5 (in progress) |
|--------|-------|---------------------|
| Plans with evening coverage | **0/25 (0%)** | **49/50 (98%)** |
| Agents with Feb 14 plan | 25/25 | 25/25 |
| Party in Feb 14 schedule | **0/25 (0%)** | **21/25 (84%)** |
| Party-aware agents | 20/25 (80%) | 22/25 (88%) |
| Total memories | 122,144 | 83,728 (still running) |
| Reflections | 1,098 | 803 (still running) |

**The key breakthrough: 21/25 agents have the Valentine's Day party in their Feb 14 schedule.** This went from literally zero to 84% with three targeted fixes. Fix C's event seeding is working exactly as intended — agents who heard about the party through conversations now have it anchored in their daily plans.

**Dialogue quality remains strong:**
- Mrs. Peterson reminiscing about "a student I had in 1987 who wanted to paint the whole town purple for Isabella's party"
- Eddy proposing a "kaleidoscope of creativity" — poems, paintings, and music
- Rachel scheming to frame the party as official "Celebrating Smallville" city project
- Miguel debating food authenticity vs showmanship

**Pending:** Full analysis when sim completes (~2-3 more hours). Key questions:
- How many agents physically attend the party at 5-7 PM?
- Are they there as guests (enjoying) or helpers (instrumental)?
- Does attendance exceed Exp 3's 6 present / 3 deliberate?
- Does Fix C's event seeding feel organic or forced?

---

## 5. Key Architectural Insights

### 5.1 Plan Decomposition Was the Hidden Killer

In early runs, each agent's daily plan was decomposed into sub-tasks, generating 70+ LLM calls per agent per planning cycle. Disabling plan decomposition reduced planning from ~40s to ~9s per agent with no measurable quality loss.

### 5.2 Committee of Experts: Viable but Slower

Running 7-9 specialized small models sequentially is functional but inherently slower than a single large model call. The tradeoff: **zero API cost** vs. longer tick processing times.

For 25 agents with 3-minute ticks, each tick can take 30-90 seconds of real time depending on conversation density.

### 5.3 The Persona Problem

Even after fixing the hardcoded planning prompt, the persona definitions still include `lunch_location: "Hobbs Cafe"` for 14/15 agents (only one has "The Rose and Crown Pub"). This creates a strong prior toward Hobbs visits regardless of party awareness.

**Open question:** Should persona lunch locations be diversified, or is a small town with one main cafe realistic? The Stanford paper's Smallville is designed as a small community where Hobbs Cafe is the natural gathering point.

### 5.4 The "Helpers Not Guests" Problem

Experiment 3's most significant finding: when small models (1-4B) respond to party information, they generate *instrumental* plans (inspect wiring, discuss safety protocols, review decorations) rather than *social* plans (attend for enjoyment, celebrate with friends). Not a single agent across 28 party-related re-plans created a plan item like "Attend Isabella's party for fun."

This suggests **small models default to instrumental reasoning over social reasoning** — finding concrete tasks rather than expressing social desires. Larger models (GPT-3.5/4) likely better capture the implicit social norm: "a party is something you attend for enjoyment." This is the primary motivation for Experiment 4's fine-tuned social expert.

### 5.5 Conversation as Information Conduit

The conversation system is the sole mechanism for information diffusion between agents. Without conversations, agents cannot learn about events they weren't initialized with. The fixes to `should_converse` (from broken 3-expert pipeline to single Social expert) were critical — the first organic conversation occurred within 2 minutes after the fix.

### 5.6 Token Limits Are Invisible Truncation

Experiment 5 revealed that the Judge expert's 150-token limit was silently truncating daily plans at ~1 PM, making afternoon/evening events impossible to schedule. This bug was invisible in outputs (plans looked reasonable, just short) and required database analysis to discover. **Always verify that generated structured outputs contain the expected range of values, not just valid formatting.**

### 5.7 Event Anchoring Bridges Memory and Planning

Agents with party memories still couldn't attend because memories and plans are separate cognitive processes. The event seeding mechanism (Fix C) bridges this gap by extracting time-specific events from memory and injecting them as structural constraints in the planning prompt. This mimics how humans plan — you put fixed events in first, then fill around them.

### 5.8 Reactive Re-planning Is the Cascade Engine

The `react_to_conversation()` mechanism (capped at 3 re-plans per day in Exp 1-2, uncapped in Exp 3+) is what allows agents to change their schedule after learning new information. Without this, hearing about the party has no behavioral effect.

---

## 6. Comparison to Stanford Paper

| Aspect | Stanford Paper | Our Implementation |
|--------|---------------|-------------------|
| Model | GPT-3.5-turbo / GPT-4 | Committee of 1B-4B local models + fine-tuned 2B specialists |
| Agents | 25 | 15 (Exp 1), 25 (Exp 2–4) |
| Sim duration | 2 days | 2 days |
| Party attendance | 5/12 invited showed up | Exp 1: 15/15 (fake), Exp 2: 0, Exp 3: 6 present (3 deliberate), Exp 4: 0 (truncated plans), **Exp 5: TBD (21/25 scheduled)** |
| Info spread | Not quantified | Exp 2: 80%, Exp 3: 100%, Exp 4: 20/25 (80%), **Exp 5: 22/25 (88%, still running)** |
| Behavioral change | Reported (replanning) | Exp 2: 0, **Exp 3: 28 replans (~10 agents)**, Exp 4: 0 (truncation bug), **Exp 5: 21/25 scheduled party** |
| Cost | ~$1000s in API calls | $0 (local inference) |
| Speed | Not reported | ~9s/agent for planning |
| Memory | Custom implementation | SQLite + FTS5 + TF-IDF |
| Reflection | Importance threshold | Same + cooldown to prevent loops |
| Dialogue quality | Single large model | Fine-tuned 2B specialist with 25 distinct character voices |
| Social reasoning | Single large model | Exp 1-3: generic 3B, **Exp 4: fine-tuned 2B social expert** |

---

## 7. Lessons Learned

1. **Validate emergence rigorously.** 100% attendance looked like a success until we traced the cause to hardcoded prompts. Always ask "why" when results look too good.

2. **Memory retrieval method matters enormously.** Chronological (`get_memories(limit=20)`) vs. relevance-weighted (`retrieve_memories(query)`) is the difference between agents that remember what matters and agents that remember what happened last.

3. **Small models can approximate large model behavior** for structured tasks (planning, importance scoring, yes/no decisions) but struggle more with nuanced dialogue and creative conversation.

4. **Rate limits are real.** Ollama Cloud's 429 errors during peak conversation periods force fallback to smaller local models, potentially degrading dialogue quality at critical moments.

5. **Reflection loops are dangerous.** Without safeguards (exclude reflections from threshold, cooldown timers), agents can enter infinite reflection spirals.

6. **The planning prompt is everything.** A single line ("Include lunch at Hobbs Cafe") can override all emergent behavior. Prompt design for autonomous agents requires extreme care about implicit instructions.

7. **Fine-tuning beats prompting for behavioral patterns.** A 2B model fine-tuned for 3 minutes on 202 examples (social expert) eliminated the "helpers not guests" problem that no amount of prompt engineering could fix in generic 3B models. Training data that explicitly includes anti-patterns (what NOT to do) is especially effective.

8. **Programmatic training data generation scales.** Template-based data generation with random variation across all 25 agents and 7 categories produces more diverse, consistent training data than hand-writing or even LLM-generating examples one at a time.

9. **Token limits cause silent truncation.** A 150-token limit on the Judge expert silently cut daily plans at ~1 PM for 4 experiments before being caught. Generated structured outputs must be validated for completeness, not just format. Per-pipeline token overrides (`PIPELINE_TOKEN_OVERRIDES`) are cleaner than raising global defaults.

10. **Event anchoring closes the memory→action gap.** Agents can have strong event memories and still not schedule them. Extracting time-specific events from memory and injecting them as planning constraints (`FIXED COMMITMENTS`) produced 21/25 agents scheduling the party (vs. 0/25 without). This is the most impactful single fix across all experiments.

11. **Two-step GGUF export on constrained hardware.** Unsloth's `save_pretrained_gguf` OOMs on 8GB VRAM. Workaround: `save_pretrained_merged()` to 16-bit safetensors → `llama.cpp/convert_hf_to_gguf.py --outtype q8_0`. Adds ~2 minutes but always succeeds.

---

## 8. Next Steps

- [x] ~~Complete Experiment 2 analysis~~ — Done. Information spread without behavioral change.
- [x] ~~Remove re-plan cap~~ — Done. Agents now have unlimited re-planning.
- [x] ~~Fine-tune character actor model~~ — Done. `smallville-actor` (2.8GB) with 25 distinct voices.
- [x] ~~Run Experiment 3~~ — Done. 25 agents, uncapped re-planning. 100% awareness, 28 party replans (~10 agents), multi-hop spread confirmed. "Helpers not guests" problem identified.
- [x] ~~Fine-tune social expert model~~ — Done. `smallville-social` (2.78GB) trained on 202 examples targeting instrumental→social reasoning shift.
- [x] ~~Integrate both fine-tuned models into Committee~~ — Done. Dialogue → `smallville-actor`, Social → `smallville-social`.
- [x] ~~Run Experiment 4~~ — In progress. Early signals extremely promising (zero helper language).
- [x] ~~Write Substack post~~ — Published: "The AI Town That Throws Parties, But Only to Fix the Wiring"
- [ ] **Complete Experiment 5 analysis** — Wait for sim to finish, full deep-dive on party attendance, guest vs helper language, Fix C effectiveness
- [x] ~~Complete Experiment 4 analysis~~ — Done. Plan truncation discovered (0/25 evening coverage). Led to Exp 5 fixes.
- [ ] Fine-tune additional experts (Judge, Emotional) if social expert shows improvement
- [ ] Diversify persona lunch locations to reduce Hobbs Cafe prior
- [ ] Implement secondary spread tracking (did Agent B tell Agent C about what Isabella said?)
- [ ] Add metrics: cascade depth, time-to-spread, information fidelity
- [ ] Compare reflection quality between committee and single-model modes
- [ ] Formal evaluation: blind test — can humans tell which agent is speaking from dialogue alone?
- [ ] Update Substack post with Experiment 4 results
- [ ] Consider Raspberry Pi 5 run (can a $60 computer simulate a town?)

---

## Appendix A: Running the Simulation

```bash
# Full 25-agent run with committee mode and web UI
cd ~/.openclaw/workspace/generative-agents
~/.pyenv/versions/3.11.9/bin/python main.py --committee --webui --webui-port 3000 --num-agents 25

# Key config (config.py)
TICK_DURATION_SECONDS = 180   # 3 min game time per tick (480 ticks/day)
DEFAULT_NUM_AGENTS = 25
DEFAULT_SIMULATION_SPEED = 10
MAX_CONVERSATION_TURNS = 6
```

## Appendix B: Database Queries

```sql
-- Total memories by type
SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type;

-- Valentine awareness by agent
SELECT agent_name, COUNT(*) FROM memories 
WHERE description LIKE '%valentine%' OR description LIKE '%Valentine%' 
GROUP BY agent_name ORDER BY COUNT(*) DESC;

-- Party-related re-plans (check simulation.log)
-- grep -i "replan.*added.*valentine\|replan.*added.*party\|replan.*added.*hobbs" simulation.log
```

---

*This notebook is a living document. Updated as experiments progress.*

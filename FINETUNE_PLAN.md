# Fine-Tuning Plan: Character-Consistent Dialogue for Generative Agents

**Goal:** Train a single small model to convincingly roleplay all 25 Smallville agents with distinct, consistent personalities — replacing the generic dialogue from qwen3-coder/llama3.2.

**Date:** February 15, 2026

---

## 1. Architecture Decision

### Single "Character Actor" LoRA (Approach 3)

One base model + one LoRA adapter that learns to switch personalities based on a structured system prompt containing the agent's identity.

**Why not per-agent LoRAs?**
- 25 separate adapters = 25 training runs, 25 files to manage
- Hot-swapping adapters per dialogue turn adds latency
- A single model that can "act" is more elegant and mirrors how human actors work
- Can always split into per-agent adapters later if quality demands it

**Base model:** `google/gemma-2-2b-it`
- 2B params — fits QLoRA training in 8GB VRAM (~5-6GB usage)
- Strong instruction-following for its size
- Already using gemma3 family in the committee, so tokenizer/behavior is familiar
- Exports cleanly to GGUF for Ollama

---

## 2. Rich Personality Profiles

Current personas.py has ~10 fields per agent. For fine-tuning, each agent needs a **deep personality profile** that captures how they *sound*, not just who they *are*.

### Profile Schema (per agent)

```yaml
name: "Isabella Rodriguez"
age: 34
occupation: "Cafe owner"

# Voice & Speech
speech_style: "Warm and inviting, uses food/nature metaphors. Speaks in flowing sentences. Asks follow-up questions to show genuine interest."
vocabulary_level: "Casual-educated. Avoids jargon. Uses sensory words (colors, textures, smells)."
catchphrases:
  - "Oh, you have to try..."
  - "That reminds me of..."
  - "Wouldn't it be lovely if..."
sentence_length: "Medium to long. Tends to chain thoughts with dashes and ellipses."
formality: "Informal but gracious. Never sloppy."

# Emotional Range
default_mood: "Warm, optimistic, slightly scattered"
when_excited: "Speaks faster, uses more exclamation marks, drops into planning mode"
when_stressed: "Gets quieter, more focused, drops the flourishes"
when_sad: "Reflective, uses past tense more, mentions her grandmother"
humor_style: "Gentle, observational. Laughs at her own overambition."

# Conversational Patterns
greeting_style: "Always asks about the other person first before talking about herself"
topic_gravity: ["community events", "food", "decorations", "nature", "creativity"]
avoids_topics: ["politics", "conflict"]
listening_style: "Active — mirrors back what people say, adds 'I love that' or 'tell me more'"
disagree_style: "Redirects rather than confronts. 'What if we tried...' instead of 'No, that won't work.'"

# Relationships Color
with_strangers: "Immediately warm, treats everyone like a regular"
with_close_friends: "More vulnerable, shares doubts about the cafe"
with_authority: "Respectful but doesn't defer — suggests rather than asks"

# Unique Quirks
quirks:
  - "Names her baked goods (calls her sourdough 'Old Faithful')"
  - "Remembers everyone's coffee order"
  - "Sketches decoration ideas on napkins mid-conversation"
```

### Implementation

Create profiles for all 25 agents in `finetune/profiles/`. Each profile ~50-80 lines of YAML. These become the system prompts during training AND inference.

**Key insight:** The system prompt at inference time must match the training format exactly. The model learns "when I see this profile structure, I become this person."

---

## 3. Training Data Generation

### 3.1 Data Format

ShareGPT/ChatML format for compatibility with unsloth:

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "[AGENT: Isabella Rodriguez]\n[AGE: 34] [OCCUPATION: Cafe owner]\n[SPEECH: Warm and inviting, uses food/nature metaphors...]\n[MOOD: Excited about Valentine's Day planning]\n[TALKING TO: Dr. Williams, town doctor, mentor figure]"
    },
    {
      "role": "user", 
      "content": "Dr. Williams: I noticed you've been collecting wildflowers from Johnson Park. What's the occasion?"
    },
    {
      "role": "assistant",
      "content": "Oh, Dr. Williams, you should see them — the most gorgeous purple asters and wild daisies! I'm thinking of weaving them into centerpieces for our Valentine's Day party at the cafe. Wouldn't it be lovely to bring a little bit of the park indoors?"
    }
  ]
}
```

### 3.2 Scenario Categories

For each agent, generate conversations across these 8 scenario types:

| Category | Purpose | Examples |
|----------|---------|----------|
| **Casual greeting** | Baseline personality | Running into someone on the street |
| **Work talk** | Domain vocabulary | Discussing their profession |
| **Event sharing** | Information diffusion | Telling someone about the party |
| **Responding to invitation** | Decision-making voice | Being invited to something |
| **Emotional moment** | Depth & vulnerability | Receiving bad/good news |
| **Multi-turn sustained** | Character persistence | 4-6 turn conversations |
| **Disagreement** | Conflict style | Politely declining or pushing back |
| **Gossip/secondhand info** | Information relay | Telling C what B told them |

### 3.3 Generation Pipeline

```
Step 1: Generate 25 rich YAML profiles (Claude)
Step 2: Generate 8 scenarios × 25 agents = 200 base conversations (Claude)
Step 3: Augment with 3 variations each = 600 conversations
Step 4: Add 150 cross-agent conversations (A talks to B, both in character)
Step 5: Quality filter — remove any that break character or sound generic
Step 6: Final dataset: ~700-750 high-quality conversations
```

**Why Claude for generation?** We need a model that can deeply inhabit a character to *write examples* of that character. Claude is our best available model for this. Cost: ~$2-5 for the full dataset at Sonnet tier.

### 3.4 Negative Examples (Optional but Valuable)

Include DPO (Direct Preference Optimization) pairs:
- **Chosen:** In-character response with personality
- **Rejected:** Generic chatbot response to same prompt

This teaches the model what *not* to do — bland, personality-less responses.

---

## 4. Fine-Tuning Configuration

### 4.1 Environment

```
Machine: Alienware m17 R4
GPU: RTX 3070 8GB VRAM
Framework: unsloth 2025.3.19 + peft 0.15.1
Base model: google/gemma-2-2b-it (download via HF)
Quantization: 4-bit (QLoRA)
```

### 4.2 Hyperparameters

```python
# QLoRA config
lora_r = 64              # Rank — higher = more capacity, more VRAM
lora_alpha = 128          # Scaling factor (2x rank is standard)
lora_dropout = 0.05       # Light dropout
target_modules = [        # Which layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",       # MLP
]

# Training config
epochs = 3                # 3 passes over data
batch_size = 2            # Micro-batch (limited by VRAM)
gradient_accumulation = 8 # Effective batch = 16
learning_rate = 2e-4      # Standard for QLoRA
warmup_ratio = 0.05       # 5% warmup
max_seq_length = 1024     # Enough for multi-turn conversations
lr_scheduler = "cosine"   # Cosine decay
weight_decay = 0.01       # Light regularization
```

### 4.3 Training Script Location

`finetune/train.py` — self-contained script that:
1. Loads base model with unsloth's `FastLanguageModel`
2. Applies QLoRA adapters
3. Loads dataset from `finetune/data/training_data.jsonl`
4. Trains with SFTTrainer from trl
5. Saves adapter to `finetune/output/smallville-actor-lora/`
6. Exports merged GGUF to `finetune/output/smallville-actor.gguf`

### 4.4 Estimated Resources

| Resource | Estimate |
|----------|----------|
| VRAM during training | ~5.5-6.5 GB |
| Training time | 30-90 minutes |
| Disk for base model | ~5 GB (HF cache) |
| Disk for LoRA adapter | ~50-100 MB |
| Disk for merged GGUF | ~1.5-2 GB |
| Dataset size | ~2-3 MB (750 conversations) |

---

## 5. Integration with Simulation

### 5.1 Ollama Modelfile

```dockerfile
FROM ./smallville-actor.gguf

PARAMETER temperature 0.85
PARAMETER num_predict 100
PARAMETER stop "<end_of_turn>"

TEMPLATE """<start_of_turn>system
{{ .System }}<end_of_turn>
<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
"""
```

```bash
ollama create smallville-actor -f Modelfile
```

### 5.2 Committee Integration

Modify `committee.py:generate_dialogue()` to:
1. Build a structured system prompt from the agent's rich profile
2. Route to `smallville-actor` model instead of `qwen3-coder:480b-cloud`
3. Fall back to cloud model if local inference fails

```python
# In generate_dialogue():
model = "smallville-actor"  # Our fine-tuned model
system_prompt = build_character_prompt(agent_name)  # From rich profiles
```

### 5.3 Profile Loader

New file `finetune/profiles.py`:
- Loads YAML profiles from `finetune/profiles/`
- Formats into the structured system prompt the model was trained on
- Called by `generate_dialogue()` at inference time

**Critical:** The system prompt format at inference MUST match training format exactly. Same field names, same ordering, same brackets.

---

## 6. Evaluation

### 6.1 Character Consistency Test

After training, run a blind test:
1. Generate 5 responses per agent to the same prompt
2. Shuffle all 125 responses
3. Ask Claude to classify which agent said each one
4. **Target: >60% correct classification** (chance = 4%)

### 6.2 A/B Comparison

Run two short simulations (100 ticks each):
- **A:** Current dialogue system (qwen3-coder cloud + llama3.2 fallback)
- **B:** Fine-tuned smallville-actor model

Compare:
- Dialogue distinctiveness (do agents sound different from each other?)
- Character drift (does personality hold across 5+ conversation turns?)
- Information accuracy (do they still correctly convey party details?)
- Naturalness (do responses feel human?)

### 6.3 Metrics to Track

| Metric | Method |
|--------|--------|
| Personality consistency | Classifier accuracy across conversations |
| Vocabulary diversity | Unique token ratio per agent |
| Catchphrase usage | Regex detection of trained catchphrases |
| Response length distribution | Should match training profile's sentence_length |
| Generic response rate | % of responses that could be anyone |

---

## 7. Implementation Phases

### Phase 1: Profiles (1-2 hours)
- [ ] Design rich YAML profile schema
- [ ] Write profiles for all 25 agents
- [ ] Review for distinctiveness — no two agents should sound alike

### Phase 2: Training Data (2-3 hours)
- [ ] Write data generation prompts for Claude
- [ ] Generate 200 base conversations (8 scenarios × 25 agents)
- [ ] Augment to 600+ with variations
- [ ] Generate 150 cross-agent conversations
- [ ] Quality filter and format as JSONL
- [ ] Validate dataset statistics

### Phase 3: Fine-Tuning (1-2 hours)
- [ ] Write `finetune/train.py` using unsloth
- [ ] Download gemma-2-2b-it base model
- [ ] Run training (~30-90 min)
- [ ] Export to GGUF
- [ ] Create Ollama modelfile and register

### Phase 4: Integration (1 hour)
- [ ] Write profile loader (`finetune/profiles.py`)
- [ ] Modify `generate_dialogue()` to use fine-tuned model
- [ ] Add structured system prompt builder
- [ ] Test with 2-3 agents manually

### Phase 5: Evaluation (1-2 hours)
- [ ] Run character consistency test
- [ ] Run A/B sim comparison (100 ticks each)
- [ ] Document results in LAB_NOTEBOOK.md

### Total estimated time: 6-10 hours of work (spread across sessions)

---

## 8. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Character bleed (agents sound like each other) | Medium | Ensure profiles are maximally distinct; increase LoRA rank |
| Overfitting to training phrases | Medium | Use diverse scenarios; monitor eval loss; early stopping |
| Loss of general capability | Low | LoRA preserves base model; we only adapt dialogue layers |
| VRAM OOM during training | Low | unsloth optimizations; reduce batch size if needed |
| Poor GGUF quantization quality | Low | Test Q5_K_M and Q4_K_M; compare outputs |
| Training data quality issues | Medium | Manual review of Claude-generated data; filter generic responses |
| Model too slow for real-time sim | Low | 2B model is fast; Ollama inference ~1-3s per response |

---

## 9. Future Extensions

1. **Per-agent LoRA adapters** — If single model can't differentiate 25 personalities, cluster into 5 archetype groups (academic, tradesperson, student, professional, elder) with separate adapters
2. **DPO training** — Add preference pairs to actively penalize generic responses
3. **Conversation memory fine-tuning** — Train the model to reference specific past conversations naturally
4. **Emotional state conditioning** — Dynamic mood tag in system prompt that changes based on recent events
5. **Voice-style transfer** — Map each agent's text style to a TTS voice profile for audio simulation

---

*Ready to execute. Start with Phase 1: profiles.*

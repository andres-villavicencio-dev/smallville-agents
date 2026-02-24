# Generative Agents — Lab Notebook

## Experiment 9: Post-Fix Validation Run (Feb 24, 2026)

**Start time (real):** ~7:48 PM NZST (Feb 24)  
**Start time (sim):** Resumed from saved state — tick 1800, sim time 2023-02-17 02:00:00  
**Duration:** ~35 minutes real time before manual stop  
**Architecture:** Committee of Experts (same as Exp 4 config)  
**Speed:** 1000x  
**Agents:** 25  
**WebUI:** port 3000  

### Purpose

Validation run after multiple fixes:
1. Added 8 missing residential locations to `config.py`
2. Integrated KittenTTS voice generation into conversation system
3. Restored all Ollama models (had been wiped — see incident below)

### Configuration

**Models (Committee):**
- Social: `smallville-social` (fine-tuned Gemma 2 2B, Q8_0 GGUF)
- Dialogue/Actor: `smallville-actor` (fine-tuned Gemma 2 2B, Q8_0 GGUF)
- Temporal: `gemma3:1b`
- Emotional: `llama3.2:3b`
- Memory: `gemma3:4b`
- Spatial: `qwen2.5:3b`
- Judge: `qwen2.5:3b`

**New locations added:**
- Williams Residence, Anderson Residence, Davis Residence, Mayor Residence
- Peterson Cottage, Thompson Residence, Wilson Apartment, Rodriguez Home

These were listed in `personas.py` as agent home_locations but missing from `SMALLVILLE_LOCATIONS` in `config.py`. The environment's `snap_to_valid_location()` was silently redirecting agents to Oak Hill College or other fallback locations at night — agents were "sleeping" at school or random places instead of going home.

**Voice integration:**
- `voice_map.py`: 25 agents mapped to 8 KittenTTS Nano voices (4M, 4F), deterministic hash-based assignment
- `voice_integration.py`: async hook in `conversation.py` `end_conversation()` — generates OGG audio for each completed conversation
- KittenTTS endpoint: `http://192.168.1.70:8377/tts` (Pi 5, systemd service)
- Set `SMALLVILLE_VOICES=1` to enable (default on)

### Observations

**Location fix confirmed:** Zero "Snapped location" warnings on startup. Agents initialized at their correct home locations:
- Mrs. Peterson → Peterson Cottage ✓
- Officer Thompson → Thompson Residence ✓  
- Miguel Rodriguez → Rodriguez Home ✓
- Frank Wilson → Wilson Apartment ✓
- Dr. Williams → Williams Residence (first load), then Hobbs Cafe (resumed state position) ✓

**Conversations flowing:** Social expert operational, agents initiating conversations naturally:
- Mrs. Peterson inviting Miguel Rodriguez to help arrange purple asters for the party
- Eddy Lin proposing community events to Mei Lin
- Conversations checked every 3rd tick as configured

**Sim time:** Resumed at 2023-02-17 02:00 (3 days past the party). This run was primarily for validation, not party observation.

### Incident: Ollama Models Wiped

**Discovered:** 8:01 PM NZST during heartbeat check  
**Impact:** All 6 committee models missing from Ollama (only cloud models + qwen2.5-coder:3b remained)  
**Symptom:** Every `should_converse` check returned `NO — empty` (404 from social expert)  
**Root cause:** Unknown — models may have been purged by an Ollama update or manual cleanup  
**Resolution:** Recreated fine-tuned models from existing GGUFs + Modelfiles, pulled base models:
```
ollama create smallville-social -f finetune/output-social/Modelfile
ollama create smallville-actor -f finetune/output/Modelfile
ollama pull gemma3:1b gemma3:4b llama3.2:3b qwen2.5:3b
```

**Critical finding from log analysis:** This same model wipe likely occurred earlier (around real time 08:53 on Feb 24), which means the simulation was running brain-dead through **the actual party time on sim Feb 14 ~5PM**. Agents physically converged on Hobbs Cafe as planned, but no conversations could occur — the party was a silent gathering. 11+ "Location at capacity" warnings confirm massive attendance with zero social interaction.

### Party Time Deep Dive (from pre-wipe logs)

**Replans explicitly mentioning the party:**
| Agent | Time | Plan |
|---|---|---|
| Mrs. Peterson | 23:22 (real) | 14:00 — Attend Valentine's Day party at Hobbs Cafe |
| Mayor Johnson | 00:20 (real) | 17:00 — Attend Valentine's Day party at Hobbs Cafe |
| Mayor Johnson | 02:27 (real) | 17:00 — Attend (duplicate replan) |
| Isabella Rodriguez | 03:03–03:20 (real) | 17:00 — Attend (3 separate replans!) |
| Professor Anderson | 04:46 (real) | 19:00 — Attend... at The Rose and Crown Pub (wrong venue!) |

**Word-of-mouth chain (traced from logs):**
1. Isabella → Rachel Kim (first invite, at Hobbs Cafe)
2. Isabella → Mrs. Peterson (invited, Mrs. Peterson replanned)
3. Isabella → Mayor Johnson (invited, Mayor replanned)
4. Isabella → Maria Santos (invited at Library)
5. Mayor Johnson → Dr. Williams (asked Dr. Williams to confirm Rachel's availability)
6. Mayor Johnson → Mike Johnson (coordinating party logistics)
7. Isabella → Mrs. Peterson (confirmation message sent)
8. Isabella → Eddy Lin (excited about Valentine's Day party)

**Hobbs Cafe convergence (01:03 real time ≈ sim midday Feb 14):**
- John Lin, Mei Lin, Eddy Lin, Tom Moreno, Carmen Moreno, Carlos Gomez, Maria Santos all moved to Hobbs Cafe
- Mike Johnson, Isabella Rodriguez, Rachel Kim already present
- Cafe hit capacity → overflow to Johnson Park (Dr. Williams, Prof Anderson, Prof Davis, Lisa Park, Mayor Johnson, Miguel, Mrs. Peterson, Officer Thompson, Rachel)
- Johnson Park also hit capacity
- Active conversations at Hobbs: Isabella buzzing about party, Carlos raving about art, Rachel plotting a surprise for Isabella

### What Needs to Happen Next

1. **Reset sim to pre-party state** (Feb 14 morning) and re-run with all models intact — the actual party has never been properly observed
2. **Monitor Ollama model persistence** — add a startup check in `main.py` that verifies all committee models are available before starting
3. **Increase Hobbs Cafe capacity** — the party overflows every time, preventing many agents from actually attending
4. **Voice test in live sim** — verify KittenTTS integration works during actual conversation flow (tested manually, not yet in sim)

---

## Previous Experiments Summary

### Exp 1–2 (Feb 13–14): Single Model Baselines
- Exp 1: 100% cafe visits — flawed metric (measured lunch, not party)
- Exp 2: 80% awareness, 0 attendees — replan cap (3/day) suppressed emergence

### Exp 3 (Feb 14–15): Committee of Experts
- 100% awareness, 28 party replans from 10 agents, 6 physically present
- "Helpers not guests" finding: small models default to instrumental tasks

### Exp 4 (Feb 17): Fine-tuned Experts
- QLoRA fine-tuned social + actor on Gemma 2 2B
- Improved awareness but "helpers not guests" persisted

### Exp 5–7: Iteration (partial runs)
- Various parameter tweaks, spatial fix experiments

### Exp 8 (Feb 22): RFM Neural Steering ⭐
- Single Gemma 3 1B + 27 RFM personality concept vectors
- 3,534 conversations, 588 reflections, 274 movements
- **51% replan rate** — helpers-not-guests FIXED
- Frank Wilson explicitly planned to "Attend Valentine's Day party"
- Only 1.95 GB VRAM

---

*Lab notebook maintained by Thelonious Crustaceous 🦀*

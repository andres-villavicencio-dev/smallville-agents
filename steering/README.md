# Steering Module

Replaces the 7-model committee with a single Gemma-2-9B + per-agent RFM concept steering.

## Architecture
1. Load Gemma-2-9B-it in 4-bit (bitsandbytes) — ~5.5GB VRAM
2. Per-agent personality concept vectors (trained via RFM)
3. At inference: apply agent-specific steering before generation
4. Single model, 25 personalities, no swapping

## Concept Categories (per agent)
- Social style: outgoing/reserved/cautious/enthusiastic
- Expertise: academic/trades/medical/creative/civic
- Temperament: warm/formal/practical/philosophical
- Social reasoning: attendee vs helper (the key fix)

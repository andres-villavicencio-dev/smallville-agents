# PROJECT_STATUS.md - Generative Agents Architecture

## 🚨 CURRENT ACTIVE STATE (DO NOT REVERT)
- **Model:** Qwen3-4B (Quantized)
- **Steering:** RFM Vectors (trained for Qwen3-4B)
- **Status:** **ACTIVE** - Running Qwen3-4B with `max_new_tokens` boost to handle `<think>` blocks.
- **Goal:** Use Qwen3's superior reasoning for agent planning and reflection.

## Configuration
- `steering/engine.py`: `MODEL_ID = "Qwen/Qwen3-4B"`
- Vectors: `steering/directions/rfm_*_qwen3_4b.pkl`

## Troubleshooting
- **Empty Responses:** If Qwen returns empty strings, ensure `max_new_tokens` is at least 1024 (it needs space to think).
- **Format:** Output is stripped of `<think>` tags by `_decode_new_tokens`.

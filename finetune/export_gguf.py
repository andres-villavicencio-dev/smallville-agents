#!/usr/bin/env python3
"""Export LoRA adapter to GGUF for Ollama.

Uses unsloth's built-in merged_16bit save, then llama.cpp to quantize.
"""

import gc
import torch
from pathlib import Path
from unsloth import FastLanguageModel

ADAPTER_PATH = Path(__file__).parent / "output" / "smallville-actor-lora"
MERGED_DIR = Path(__file__).parent / "output" / "merged_f16"
MAX_SEQ_LENGTH = 1024

print("Loading base model + LoRA adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(ADAPTER_PATH),
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Use unsloth's proper merged save — saves as float16, handles dequant
MERGED_DIR.mkdir(parents=True, exist_ok=True)
print(f"Saving merged float16 model to {MERGED_DIR}...")
model.save_pretrained_merged(
    str(MERGED_DIR),
    tokenizer,
    save_method="merged_16bit",
)

print(f"\nMerged f16 model saved!")
print(f"Now run: python llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outtype q8_0")

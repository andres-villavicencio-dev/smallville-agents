#!/usr/bin/env python3
"""Fine-tune gemma-2-2b-it with QLoRA for Smallville character acting.

Uses unsloth for memory-efficient training on RTX 3070 (8GB VRAM).
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# === Config ===
BASE_MODEL = "unsloth/gemma-2-2b-it"  # unsloth optimized version
OUTPUT_DIR = Path(__file__).parent / "output"
DATA_FILE = Path(__file__).parent / "data" / "training_data.jsonl"
MAX_SEQ_LENGTH = 1024
DTYPE = None  # auto-detect (float16 for RTX 3070)

# QLoRA config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# Training config
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8  # effective batch = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01


def load_data():
    """Load and format training data from JSONL."""
    examples = []
    with open(DATA_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                convos = ex.get("conversations", [])
                if len(convos) >= 3:
                    examples.append({
                        "system": convos[0]["content"],
                        "user": convos[1]["content"],
                        "assistant": convos[2]["content"],
                    })
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(examples)} training examples")
    return examples


def format_prompt(example):
    """Format example into gemma-2 chat template."""
    return (
        f"<start_of_turn>user\n"
        f"System context: {example['system']}\n\n"
        f"{example['user']}<end_of_turn>\n"
        f"<start_of_turn>model\n"
        f"{example['assistant']}<end_of_turn>"
    )


def main():
    print("=" * 60)
    print("Smallville Character Actor — Fine-Tuning")
    print("=" * 60)
    
    # Load data
    examples = load_data()
    if not examples:
        print("ERROR: No training data found!")
        return
    
    # Format into prompt strings
    formatted = [{"text": format_prompt(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)
    print(f"Dataset: {len(dataset)} examples")
    print(f"Sample:\n{formatted[0]['text'][:300]}...\n")
    
    # Load model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=True,
    )
    
    # Apply LoRA
    print("Applying QLoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # memory optimization
        random_state=42,
    )
    
    # Training args
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        save_strategy="epoch",
        report_to="none",
    )
    
    # Trainer
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,  # pack multiple short examples into one sequence
    )
    
    # Train
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")
    print(f"  Samples/sec: {stats.metrics['train_samples_per_second']:.2f}")
    
    # Save LoRA adapter
    adapter_path = OUTPUT_DIR / "smallville-actor-lora"
    print(f"\nSaving LoRA adapter to {adapter_path}...")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    
    # Export to GGUF for Ollama
    print("\nExporting to GGUF (Q5_K_M)...")
    gguf_path = OUTPUT_DIR / "smallville-actor-Q5_K_M.gguf"
    model.save_pretrained_gguf(
        str(OUTPUT_DIR / "gguf_export"),
        tokenizer,
        quantization_method="q5_k_m",
    )
    
    print(f"\n{'=' * 60}")
    print("DONE! Next steps:")
    print(f"  1. Create Ollama modelfile")
    print(f"  2. ollama create smallville-actor -f Modelfile")
    print(f"  3. Test: ollama run smallville-actor")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

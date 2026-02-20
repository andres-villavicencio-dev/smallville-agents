#!/usr/bin/env python3
"""Fine-tune gemma-2-2b-it with QLoRA for Smallville Social Expert.

Teaches social reasoning: parties = attend for enjoyment, not instrumental tasks.
Same pipeline as smallville-actor training.
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# === Config ===
BASE_MODEL = "unsloth/gemma-2-2b-it"
OUTPUT_DIR = Path(__file__).parent / "output-social"
DATA_FILE = Path(__file__).parent / "data" / "social_expert_training.jsonl"
MAX_SEQ_LENGTH = 512  # Social expert outputs are short (1-2 sentences)
DTYPE = None

# QLoRA config
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05

# Training config — more epochs for smaller dataset
EPOCHS = 5
BATCH_SIZE = 2
GRAD_ACCUM = 4  # effective batch = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

MODEL_NAME = "smallville-social"


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
    print(f"Smallville Social Expert — Fine-Tuning")
    print("=" * 60)
    
    examples = load_data()
    if not examples:
        print("ERROR: No training data found!")
        return
    
    formatted = [{"text": format_prompt(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)
    print(f"Dataset: {len(dataset)} examples")
    print(f"Sample:\n{formatted[0]['text'][:300]}...\n")
    
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=True,
    )
    
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
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    
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
    
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
    )
    
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")
    print(f"  Samples/sec: {stats.metrics['train_samples_per_second']:.2f}")
    
    # Save LoRA adapter
    adapter_path = OUTPUT_DIR / f"{MODEL_NAME}-lora"
    print(f"\nSaving LoRA adapter to {adapter_path}...")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    
    # Export to GGUF Q8_0 (same as smallville-actor)
    print("\nExporting to GGUF (Q8_0)...")
    model.save_pretrained_gguf(
        str(OUTPUT_DIR / "gguf_export"),
        tokenizer,
        quantization_method="q8_0",
    )
    
    print(f"\n{'=' * 60}")
    print("DONE! Next steps:")
    print(f"  1. Create Modelfile pointing to the GGUF")
    print(f"  2. ollama create {MODEL_NAME} -f Modelfile")
    print(f"  3. Test: ollama run {MODEL_NAME}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

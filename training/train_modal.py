"""
Fine-tune Qwen 2.5 0.5B/1.5B on Chirp cleanup data using Modal + Unsloth.

Runs QLoRA fine-tuning on an L40S GPU, exports to GGUF Q4_K_M.

Usage:
    python -m modal run train_modal.py
    python -m modal run train_modal.py --model 1.5b --epochs 5
    python -m modal run train_modal.py --model 0.5b --epochs 3 --lr 2e-4
"""

import modal
from pathlib import Path

MINUTES = 60

app = modal.App("chirp-finetune")

training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("git", "cmake", "curl", "libssl-dev", "libcurl4-openssl-dev", "build-essential")
    .run_commands(
        # Pre-install llama.cpp so Unsloth doesn't try to do it interactively
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /root/.unsloth/llama.cpp",
        "cd /root/.unsloth/llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)",
    )
    .uv_pip_install(
        "unsloth[cu128]",
        "xformers",
        "trl",
        "datasets",
        "huggingface-hub",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("chirp-models", create_if_missing=True)


MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b": "Qwen/Qwen2.5-3B-Instruct",
}


@app.function(
    image=training_image,
    gpu="L40S",
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/output": output_vol,
    },
)
def train(
    dataset_jsonl: str,
    model_size: str = "0.5b",
    epochs: int = 3,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 8,
    grad_accum: int = 2,
    max_seq_length: int = 1024,
):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    import json, os

    model_name = MODELS[model_size]
    run_name = f"chirp-cleanup-{model_size}"
    output_dir = f"/output/{run_name}"

    print(f"{'='*50}")
    print(f"Chirp Cleanup Model Fine-Tuning")
    print(f"{'='*50}")
    print(f"Base model: {model_name}")
    print(f"Dataset: {len(dataset_jsonl.splitlines())} examples")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"LoRA r={lora_r}, alpha={lora_alpha}")
    print(f"Batch: {batch_size} x {grad_accum} grad_accum = {batch_size * grad_accum} effective")
    print(f"Max seq length: {max_seq_length}")
    print()

    # Write dataset to temp file
    dataset_path = "/tmp/training_data.jsonl"
    with open(dataset_path, "w") as f:
        f.write(dataset_jsonl)

    # Load model with Unsloth (4-bit QLoRA)
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Apply chat template
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    print(f"Dataset: {len(dataset)} examples")
    print(f"Sample (first 500 chars): {dataset[0]['text'][:500]}")
    print()

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=lr,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            optim="adamw_8bit",
            seed=42,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            packing=True,
        ),
    )

    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Training loss: {stats.training_loss:.4f}")

    # Save LoRA adapter
    adapter_path = f"{output_dir}/lora"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"  LoRA adapter saved to {adapter_path}")

    # Export to GGUF Q4_K_M
    print("\nExporting to GGUF Q4_K_M...")
    gguf_path = f"{output_dir}/gguf"
    model.save_pretrained_gguf(
        gguf_path,
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Find the GGUF file — Unsloth may nest it in a _gguf subdirectory
    import shutil
    gguf_files = list(Path(gguf_path).rglob("*Q4_K_M*.gguf"))
    if not gguf_files:
        gguf_files = list(Path(gguf_path).rglob("*.gguf"))

    final_name = f"chirp-cleanup-{model_size}-q4_k_m.gguf"
    if gguf_files:
        # Pick the Q4_K_M file, or the first one
        gguf_file = gguf_files[0]
        size_mb = gguf_file.stat().st_size / (1024 * 1024)
        print(f"  GGUF found: {gguf_file} ({size_mb:.0f} MB)")

        final_path = f"/output/{final_name}"
        shutil.copy2(str(gguf_file), final_path)
        print(f"  Copied to: {final_path}")
    else:
        print("  Warning: No GGUF file found. Listing export directory:")
        for p in Path(gguf_path).rglob("*"):
            print(f"    {p}")

    # Commit volume
    output_vol.commit()

    print(f"\n{'='*50}")
    print(f"Done! Model saved to chirp-models volume.")
    print(f"Download with: python -m modal volume get chirp-models {final_name}")
    print(f"{'='*50}")

    return {
        "model": model_name,
        "loss": stats.training_loss,
        "steps": stats.global_step,
        "epochs": epochs,
    }


@app.local_entrypoint()
def main(
    model: str = "0.5b",
    epochs: int = 3,
    lr: float = 2e-4,
    dataset: str = "data/training_v2.jsonl",
):
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run prepare_dataset.py first.")
        return

    # Read dataset
    dataset_jsonl = dataset_path.read_text(encoding="utf-8")
    line_count = len(dataset_jsonl.strip().splitlines())
    print(f"Loaded {line_count} training examples from {dataset_path}")
    print(f"Model: Qwen 2.5 {model.upper()} Instruct")
    print(f"Epochs: {epochs}, LR: {lr}")
    print(f"GPU: L40S on Modal")
    print()

    result = train.remote(
        dataset_jsonl=dataset_jsonl,
        model_size=model,
        epochs=epochs,
        lr=lr,
    )

    print(f"\nResults: {result}")
    print(f"\nTo download the model:")
    print(f"  python -m modal volume get chirp-models chirp-cleanup-{model}-q4_k_m.gguf")

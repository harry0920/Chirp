"""
Fine-tune FLAN-T5-small on Chirp cleanup data using Modal.

Uses HuggingFace Seq2SeqTrainer (full fine-tune, not LoRA — model is only 77M params).
Exports to CTranslate2 INT8 format for production use.

Usage:
    python -m modal run train_t5_modal.py
    python -m modal run train_t5_modal.py --epochs 5 --lr 3e-4
    python -m modal run train_t5_modal.py --model flan-t5-base --epochs 3
"""

import modal
from pathlib import Path

MINUTES = 60

app = modal.App("chirp-t5-finetune")

training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "transformers",
        "datasets",
        "accelerate",
        "sentencepiece",
        "protobuf",
        "torch",
        "ctranslate2",
        "huggingface-hub",
        "evaluate",
        "rouge_score",
        "nltk",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("chirp-models", create_if_missing=True)

MODELS = {
    "flan-t5-small": "google/flan-t5-small",
    "flan-t5-base": "google/flan-t5-base",
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
    model_key: str = "flan-t5-small",
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 16,
    grad_accum: int = 2,
    max_input_length: int = 256,
    max_target_length: int = 256,
    warmup_ratio: float = 0.05,
):
    import json
    import torch
    from transformers import (
        AutoTokenizer,
        T5ForConditionalGeneration,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset

    model_name = MODELS[model_key]
    run_name = f"chirp-{model_key}"
    output_dir = f"/output/{run_name}"

    print(f"{'='*60}")
    print(f"  Chirp T5 Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Base model: {model_name}")
    print(f"  Dataset: {len(dataset_jsonl.splitlines())} examples")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"  Batch: {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Max lengths: input={max_input_length}, target={max_target_length}")
    print()

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Full fine-tune — T5-small is only 77M params, no need for LoRA
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params/1e6:.0f}M total, {trainable/1e6:.0f}M trainable")

    # Load dataset
    print("Loading dataset...")
    dataset_path = "/tmp/training_data.jsonl"
    with open(dataset_path, "w") as f:
        f.write(dataset_jsonl)

    raw_data = []
    with open(dataset_path) as f:
        for line in f:
            raw_data.append(json.loads(line))

    dataset = Dataset.from_list(raw_data)

    # Split 95/5 for train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Tokenize
    def preprocess(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            examples["target"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=["input", "target"])
    eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=["input", "target"])

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        predict_with_generate=False,  # just track loss for speed
        seed=42,
        report_to="none",
    )

    # Train
    print("Starting training...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Steps: {result.global_step}")
    print(f"  Train loss: {result.training_loss:.4f}")

    # Evaluate
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    # Save the fine-tuned model (HuggingFace format)
    hf_path = f"{output_dir}/hf"
    trainer.save_model(hf_path)
    tokenizer.save_pretrained(hf_path)
    print(f"  HF model saved to {hf_path}")

    # Convert to CTranslate2 INT8
    print("\nConverting to CTranslate2 INT8...")
    import ctranslate2

    # Monkey-patch for version compatibility
    _orig = T5ForConditionalGeneration.from_pretrained.__func__
    def _patched(cls, *args, **kwargs):
        kwargs.pop("dtype", None)
        return _orig(cls, *args, **kwargs)
    T5ForConditionalGeneration.from_pretrained = classmethod(_patched)

    ct2_path = f"{output_dir}/ct2-int8"
    converter = ctranslate2.converters.TransformersConverter(hf_path)
    converter.convert(ct2_path, quantization="int8", force=True)

    # Calculate sizes
    import shutil
    ct2_size = sum(f.stat().st_size for f in Path(ct2_path).rglob("*") if f.is_file())
    print(f"  CT2 INT8 model: {ct2_size/1e6:.0f} MB at {ct2_path}")

    # Also make a zip for easy download
    zip_name = f"chirp-{model_key}-ct2-int8"
    shutil.make_archive(f"/output/{zip_name}", "zip", ct2_path)
    print(f"  Zipped to /output/{zip_name}.zip")

    # Commit volume
    output_vol.commit()

    print(f"\n{'='*60}")
    print(f"  Done! Model saved to chirp-models volume.")
    print(f"  Download:")
    print(f"    modal volume get chirp-models {zip_name}.zip")
    print(f"{'='*60}")

    return {
        "model": model_name,
        "train_loss": result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "steps": result.global_step,
        "epochs": epochs,
        "ct2_size_mb": ct2_size / 1e6,
    }


@app.local_entrypoint()
def main(
    model: str = "flan-t5-small",
    epochs: int = 5,
    lr: float = 3e-4,
    batch_size: int = 16,
    dataset: str = "data/training_t5.jsonl",
):
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Run generate_data_t5.py first:")
        print("  python -m modal run generate_data_t5.py --pairs 2000")
        return

    dataset_jsonl = dataset_path.read_text(encoding="utf-8")
    line_count = len(dataset_jsonl.strip().splitlines())

    print(f"=== Chirp T5 Fine-Tuning ===")
    print(f"Model: {model}")
    print(f"Dataset: {line_count} examples from {dataset_path}")
    print(f"Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"GPU: L40S on Modal")
    print()

    result = train.remote(
        dataset_jsonl=dataset_jsonl,
        model_key=model,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    print(f"\nResults: {result}")
    print(f"\nTo download:")
    print(f"  modal volume get chirp-models chirp-{model}-ct2-int8.zip")

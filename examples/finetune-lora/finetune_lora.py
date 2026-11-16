#!/usr/bin/env python3
"""
Fine-tune an open-weight HuggingFace LLM with QLoRA/LoRA for vLLM serving.

Designed to run on a single 24GB NVIDIA L4 GPU (e.g. GCE g2-standard-4) or Colab.
Outputs a lightweight LoRA adapter (~30-100 MB) that vLLM can hot-load via
`--enable-lora --lora-modules`, or optionally merges weights into a standalone
16-bit model.

Example:
  python3 finetune_lora.py \
    --model-id Qwen/Qwen2.5-3B-Instruct \
    --dataset sample_dataset.jsonl \
    --output-dir ./outputs/qwen2.5-3b-lora
"""

import argparse
import gc
import inspect
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_DATASET = str(Path(__file__).resolve().parent / "sample_dataset.jsonl")


def parse_args():
    """Parse CLI arguments for LoRA fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune an open-weight LLM with QLoRA/LoRA.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to JSONL dataset")
    parser.add_argument("--output-dir", default="./outputs/qwen2.5-3b-lora", help="Adapter output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit QLoRA and train in 16-bit LoRA")
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo ID to push adapter (e.g. user/my-lora)")
    parser.add_argument("--merge-and-save", default=None, help="Optional directory to save 16-bit merged model")
    parser.add_argument("--report-to", default="none", help="Experiment tracker (e.g. none, tensorboard)")
    return parser.parse_args()


def get_hf_token():
    """Return HuggingFace token from environment if configured."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_compute_dtype():
    """Select bfloat16 when supported by the GPU, otherwise float16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def format_chat_dataset(dataset_path, tokenizer):
    """Load a JSONL dataset and render chat messages into training text."""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    if "text" in dataset.column_names and "messages" not in dataset.column_names:
        return dataset
    if "messages" not in dataset.column_names:
        raise ValueError("Dataset must contain either a 'messages' or 'text' column.")

    def render_messages(example):
        messages = example["messages"]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            # Some model templates (e.g. Gemma) disallow a separate system role;
            # fold system instructions into the first user message as a fallback.
            if messages and messages[0]["role"] == "system" and len(messages) > 1:
                merged = [
                    {"role": "user", "content": f"{messages[0]['content']}\n\n{messages[1]['content']}"},
                    *messages[2:],
                ]
                text = tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=False)
            else:
                raise
        return {"text": text}

    return dataset.map(render_messages, remove_columns=dataset.column_names)


def load_model_and_tokenizer(model_id, use_4bit, compute_dtype, hf_token):
    """Load tokenizer and base model (quantized in 4-bit when use_4bit is True)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def build_sft_config(args, use_4bit, compute_dtype):
    """Build SFTConfig compatible across TRL releases."""
    sig_params = inspect.signature(SFTConfig.__init__).parameters
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 2,
        "logging_steps": 1,
        "save_strategy": "no",
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "optim": "paged_adamw_8bit" if use_4bit else "adamw_torch",
        "gradient_checkpointing": True,
        "report_to": args.report_to,
        "dataset_text_field": "text",
        "packing": False,
    }

    if "max_length" in sig_params:
        kwargs["max_length"] = args.max_seq_length
    elif "max_seq_length" in sig_params:
        kwargs["max_seq_length"] = args.max_seq_length

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
    return SFTConfig(**filtered_kwargs)


def build_trainer(model, tokenizer, dataset, sft_config):
    """Instantiate SFTTrainer across TRL API versions."""
    sig_params = inspect.signature(SFTTrainer.__init__).parameters
    kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
    }
    if "processing_class" in sig_params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sig_params:
        kwargs["tokenizer"] = tokenizer
    return SFTTrainer(**kwargs)


def merge_and_save_model(model_id, adapter_dir, merge_dir, compute_dtype, hf_token):
    """Reload base model in 16-bit, merge LoRA weights, and save standalone model."""
    print(f"\nMerging LoRA adapter from {adapter_dir} into 16-bit base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        device_map="auto",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=hf_token)

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()

    Path(merge_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merge_dir, safe_serialization=True)
    tokenizer.save_pretrained(merge_dir)
    print(f"Saved merged 16-bit model to {merge_dir}")


def main():
    args = parse_args()
    use_4bit = not args.no_4bit
    compute_dtype = get_compute_dtype()
    hf_token = get_hf_token()

    print(f"Loading model: {args.model_id} (4-bit QLoRA: {use_4bit}, dtype: {compute_dtype})")
    model, tokenizer = load_model_and_tokenizer(args.model_id, use_4bit, compute_dtype, hf_token)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = format_chat_dataset(args.dataset, tokenizer)
    print(f"Loaded {len(dataset)} training examples from {args.dataset}")

    sft_config = build_sft_config(args, use_4bit, compute_dtype)
    trainer = build_trainer(model, tokenizer, dataset, sft_config)

    print("\nStarting fine-tuning...")
    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved LoRA adapter to {args.output_dir}")

    if args.push_to_hub:
        print(f"Pushing LoRA adapter to HuggingFace Hub: {args.push_to_hub}")
        trainer.model.push_to_hub(args.push_to_hub, token=hf_token)
        tokenizer.push_to_hub(args.push_to_hub, token=hf_token)

    if args.merge_and_save:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        merge_and_save_model(args.model_id, args.output_dir, args.merge_and_save, compute_dtype, hf_token)

    print("\nDone! Next steps to serve with vLLM:")
    if args.push_to_hub:
        print(
            f'  terraform apply -var="model_id={args.model_id}" '
            f'-var="extra_vllm_args=--enable-lora --lora-modules my-finetune={args.push_to_hub}"'
        )
    else:
        print(
            f"  Mount {Path(args.output_dir).resolve()} into the vLLM container and pass: "
            "--enable-lora --lora-modules my-finetune=/lora"
        )


if __name__ == "__main__":
    main()

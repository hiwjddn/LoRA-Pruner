#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ----------------------------
# Task -> dataset chooser
# ----------------------------

def load_task_dataset(task: str, seed: int = 42, max_train: Optional[int] = None, max_eval: Optional[int] = None):

    if task == "instruct":
        ds = load_dataset("tatsu-lab/alpaca")
        full = ds["train"].shuffle(seed=seed)
        split = full.train_test_split(test_size=0.02, seed=seed)
        train_raw, eval_raw = split["train"], split["test"]

        def map_fn(ex):
            inst = ex.get("instruction", "").strip()
            inp = ex.get("input", "").strip()
            out = ex.get("output", "").strip()

            if inp:
                prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
            return {"prompt": prompt, "response": out}

        train = train_raw.map(map_fn, remove_columns=train_raw.column_names)
        evald = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    elif task == "math":
        ds = load_dataset("gsm8k", "main")
        train_raw, eval_raw = ds["train"], ds["test"]

        def map_fn(ex):
            q = ex["question"].strip()
            a = ex["answer"].strip()
            prompt = f"Solve the problem step by step.\n\n### Problem:\n{q}\n\n### Solution:\n"
            return {"prompt": prompt, "response": a}

        train = train_raw.map(map_fn, remove_columns=train_raw.column_names)
        evald = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    elif task == "summarize":
        ds = load_dataset("xsum")
        train_raw, eval_raw = ds["train"], ds["validation"]

        def map_fn(ex):
            doc = ex["document"].strip()
            summ = ex["summary"].strip()
            prompt = f"Summarize the following document.\n\n### Document:\n{doc}\n\n### Summary:\n"
            return {"prompt": prompt, "response": summ}

        train = train_raw.map(map_fn, remove_columns=train_raw.column_names)
        evald = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    elif task == "qa":
        ds = load_dataset("squad_v2")
        train_raw, eval_raw = ds["train"], ds["validation"]

        def map_fn(ex):
            ctx = ex["context"].strip()
            q = ex["question"].strip()

            ans_list = ex["answers"]["text"]
            ans = ans_list[0].strip() if len(ans_list) else "unanswerable"
            prompt = f"Answer the question using the context.\n\n### Context:\n{ctx}\n\n### Question:\n{q}\n\n### Answer:\n"
            return {"prompt": prompt, "response": ans}

        train = train_raw.map(map_fn, remove_columns=train_raw.column_names)
        evald = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    elif task == "sentiment":
        ds = load_dataset("glue", "sst2")
        train_raw, eval_raw = ds["train"], ds["validation"]

        def map_fn(ex):
            sent = ex["sentence"].strip()
            label = ex["label"]
            
            ans = "positive" if label == 1 else "negative"
            prompt = f"Classify the sentiment.\n\n### Text:\n{sent}\n\n### Sentiment:\n"
            return {"prompt": prompt, "response": ans}

        train = train_raw.map(map_fn, remove_columns=train_raw.column_names)
        evald = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    else:
        raise ValueError(f"Unknown task: {task}")

    if max_train is not None:
        train = train.select(range(min(max_train, len(train))))
    if max_eval is not None:
        evald = evald.select(range(min(max_eval, len(evald))))

    return train, evald, task


def load_multitask_dataset(seed: int, max_train: int, max_eval: int):
    
    tasks = ["instruct", "math", "summarize", "qa", "sentiment"]
    trains, evals = [], []
    per_task_train = max_train // len(tasks)
    per_task_eval = max_eval // len(tasks)

    for t in tasks:
        tr, ev, _ = load_task_dataset(t, seed=seed, max_train=per_task_train, max_eval=per_task_eval)
        # add a task tag for debugging
        tr = tr.map(lambda ex: {"prompt": f"[TASK={t}]\n" + ex["prompt"], "response": ex["response"]})
        ev = ev.map(lambda ex: {"prompt": f"[TASK={t}]\n" + ex["prompt"], "response": ex["response"]})
        trains.append(tr)
        evals.append(ev)

    train = concatenate_datasets(trains).shuffle(seed=seed)
    evald = concatenate_datasets(evals).shuffle(seed=seed)
    return train, evald


# ----------------------------
# Tokenization / formatting
# ----------------------------

def build_text(prompt: str, response: str) -> str:
    
    return prompt + response

def tokenize_sft(ex, tokenizer, max_length: int):
    text = build_text(ex["prompt"], ex["response"])
    out = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return out


# ----------------------------
# LoRA target modules (model families)
# ----------------------------

def lora_targets_for_model(model_name: str) -> List[str]:
    
    n = model_name.lower()
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="HF model name or path. e.g. mistralai/Mistral-7B-v0.1, Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.2-3B")
    p.add_argument("--task", type=str, default="instruct",
                   choices=["instruct", "math", "summarize", "qa", "sentiment", "multitask"])
    p.add_argument("--output_dir", type=str, default="./lora_out")
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--max_train", type=int, default=50_000, help="Cap train samples (useful for quick runs)")
    p.add_argument("--max_eval", type=int, default=1_000, help="Cap eval samples")
    p.add_argument("--max_length", type=int, default=1024)

    # Training
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Memory options
    p.add_argument("--use_4bit", action="store_true", help="Use bitsandbytes 4-bit quantization for base model")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    args = p.parse_args()
    set_seed(args.seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if args.task == "multitask":
        train_ds, eval_ds = load_multitask_dataset(args.seed, args.max_train, args.max_eval)
    else:
        train_ds, eval_ds, _ = load_task_dataset(args.task, seed=args.seed, max_train=args.max_train, max_eval=args.max_eval)

    # Tokenize
    def tok_map(ex):
        return tokenize_sft(ex, tokenizer, args.max_length)

    train_tok = train_ds.map(tok_map, remove_columns=train_ds.column_names, desc="Tokenizing train")
    eval_tok = eval_ds.map(tok_map, remove_columns=eval_ds.column_names, desc="Tokenizing eval")

    # Model load
    quant_config = None
    model_kwargs = {"device_map": "auto"}
    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quant_config
        except Exception as e:
            raise RuntimeError("bitsandbytes/4bit requested but not available. Install bitsandbytes.") from e

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        **model_kwargs,
    )
    model.config.use_cache = False

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    target_modules = lora_targets_for_model(args.model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    output_dir = f"{args.output_dir}/r{args.lora_r}a{args.lora_alpha}/{args.seed}"
    # Train args
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        ddp_find_unused_parameters=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n[done] saved adapter + config to: {output_dir}")
    print("Tip: for inference, load base model + PeftModel.from_pretrained(base, output_dir)")


if __name__ == "__main__":
    main()

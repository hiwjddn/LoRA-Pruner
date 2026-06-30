#!/usr/bin/env python3
import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Basic utils
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_mask_key_for_module(module_name: str, mask_keys: List[str]):
    matched = None

    for key in mask_keys:
        if module_name.endswith(key):
            if matched is not None:
                raise RuntimeError(
                    f"Ambiguous mask key match for module={module_name}: "
                    f"{matched} and {key}"
                )
            matched = key

    return matched


def get_lora_weights(module, adapter_name: str):
    if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
        return None, None

    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        return None, None

    lora_A = module.lora_A[adapter_name].weight
    lora_B = module.lora_B[adapter_name].weight

    return lora_A, lora_B


# ============================================================
# LoRA mask logic
# ============================================================

def make_q_B_grad_mask(
    lora_B: torch.Tensor,
    head_mask: torch.Tensor,
) -> torch.Tensor:

    out_features, r = lora_B.shape
    num_heads = head_mask.numel()

    if out_features % num_heads != 0:
        raise ValueError(
            f"q_proj lora_B out_features={out_features} is not divisible "
            f"by num_heads={num_heads}"
        )

    head_dim = out_features // num_heads

    row_mask = torch.zeros(out_features, dtype=torch.bool)

    for h, enabled in enumerate(head_mask.tolist()):
        if enabled:
            start = h * head_dim
            end = (h + 1) * head_dim
            row_mask[start:end] = True

    grad_mask = row_mask[:, None].expand(out_features, r)

    return grad_mask


def make_o_A_grad_mask(
    lora_A: torch.Tensor,
    head_mask: torch.Tensor,
) -> torch.Tensor:

    r, in_features = lora_A.shape
    num_heads = head_mask.numel()

    if in_features % num_heads != 0:
        raise ValueError(
            f"o_proj lora_A in_features={in_features} is not divisible "
            f"by num_heads={num_heads}"
        )

    head_dim = in_features // num_heads

    col_mask = torch.zeros(in_features, dtype=torch.bool)

    for h, enabled in enumerate(head_mask.tolist()):
        if enabled:
            start = h * head_dim
            end = (h + 1) * head_dim
            col_mask[start:end] = True
        else:
            start = h * head_dim
            end = (h + 1) * head_dim
            lora_A.data[:, start:end] = 0

    grad_mask = col_mask[None, :].expand(r, in_features)

    return grad_mask


def register_gradient_mask_hook(
    param: torch.nn.Parameter,
    grad_mask: torch.Tensor,
    name: str,
):

    grad_mask = grad_mask.to(device=param.device, dtype=param.dtype)

    def hook(grad):
        return grad * grad_mask

    param.register_hook(hook)
    # print(f"[HOOK] {name}, shape={tuple(param.shape)}")


def apply_lora_mask_hooks(
    model: torch.nn.Module,
    mask_path: str,
    adapter_name: str = "default",
):

    mask_pack = torch.load(mask_path, map_location="cpu")
    masks: Dict[str, Dict[str, Any]] = mask_pack["masks"]
    mask_keys = list(masks.keys())

    hooked_modules = 0
    skipped_modules = 0

    for module_name, module in model.named_modules():
        mask_key = find_mask_key_for_module(module_name, mask_keys)

        if mask_key is None:
            continue

        mask_info = masks[mask_key]
        proj = mask_info["proj"]
        mask_type = mask_info["type"]

        lora_A, lora_B = get_lora_weights(module, adapter_name)

        if lora_A is None or lora_B is None:
            skipped_modules += 1
            print(f"[WARN] matched mask but no LoRA weights: {module_name}")
            continue

        if mask_type == "head":
            head_mask = mask_info["head_mask"].to(dtype=torch.bool)

            if proj == "q":
                B_mask = make_q_B_grad_mask(lora_B, head_mask)
                register_gradient_mask_hook(
                    lora_B,
                    B_mask,
                    name=f"{module_name}.lora_B[{adapter_name}].weight",
                )
                hooked_modules += 1

            elif proj == "o":
                A_mask = make_o_A_grad_mask(lora_A, head_mask)
                register_gradient_mask_hook(
                    lora_A,
                    A_mask,
                    name=f"{module_name}.lora_A[{adapter_name}].weight",
                )

                hooked_modules += 1

            else:
                raise ValueError(f"Unexpected head mask proj={proj}")

        elif mask_type == "module":
            enabled = bool(mask_info["enabled"])

            if enabled:
                hooked_modules += 1

            else:
                A_mask = torch.zeros_like(lora_A, dtype=torch.bool)
                B_mask = torch.zeros_like(lora_B, dtype=torch.bool)

                register_gradient_mask_hook(
                    lora_A,
                    A_mask,
                    name=f"{module_name}.lora_A[{adapter_name}].weight",
                )
                register_gradient_mask_hook(
                    lora_B,
                    B_mask,
                    name=f"{module_name}.lora_B[{adapter_name}].weight",
                )

                hooked_modules += 1

        else:
            raise ValueError(f"Unknown mask type={mask_type}")

    print(f"[MASK] hooked modules: {hooked_modules}")
    print(f"[MASK] skipped matched modules without LoRA: {skipped_modules}")
    print(f"[MASK] stats: {mask_pack.get('stats', {})}")

    return mask_pack


# ============================================================
# Dataset
# ============================================================

def normalize_task_name(task: str) -> str:
    """
    CLI에서 --task sum으로 넣어도 summarize와 동일하게 처리한다.
    """
    if task == "sum":
        return "summarize"
    return task


def load_task_dataset(
    task: str,
    seed: int = 42,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
):

    task = normalize_task_name(task)

    if task == "sentiment":
        raw = load_dataset("glue", "sst2")
        train_raw, eval_raw = raw["train"], raw["validation"]

        def map_fn(ex):
            sent = ex["sentence"].strip()
            answer = "positive" if int(ex["label"]) == 1 else "negative"
            prompt = f"Classify the sentiment.\n\n### Text:\n{sent}\n\n### Sentiment:\n"
            return {"prompt": prompt, "response": answer}

    elif task == "summarize":
        raw = load_dataset("xsum")
        train_raw, eval_raw = raw["train"], raw["validation"]

        def map_fn(ex):
            doc = ex["document"].strip()
            summary = ex["summary"].strip()
            prompt = f"Summarize the following document.\n\n### Document:\n{document}\n\n### Summary:\n"
            return {"prompt": prompt, "response": summary}

    elif task == "qa":
        raw = load_dataset("squad_v2")
        train_raw, eval_raw = raw["train"], raw["validation"]

        def map_fn(ex):
            context = ex["context"].strip()
            question = ex["question"].strip()
            answers = ex["answers"]["text"]
            answer = answers[0].strip() if len(answers) > 0 else "unanswerable"
            prompt = f"Answer the question using the context.\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n"
            return {"prompt": prompt, "response": answer}

    elif task == "math":
        raw = load_dataset("gsm8k", "main")
        train_raw, eval_raw = raw["train"], raw["test"]

        def map_fn(ex):
            question = ex["question"].strip()
            answer = ex["answer"].strip()
            prompt = f"Solve the problem step by step.\n\n### Problem:\n{question}\n\n### Solution:\n"
            return {"prompt": prompt, "response": answer}

    else:
        raise ValueError(
            f"Unknown task={task}. Choose one of: sentiment, summarize, sum, qa, math"
        )

    train_ds = train_raw.map(map_fn, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    if max_train is not None and max_train >= 0:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_eval is not None and max_eval >= 0:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    return train_ds, eval_ds


def tokenize_prompt_response(example, tokenizer, max_length: int):
    
    prompt = example["prompt"]
    response = example["response"]

    if tokenizer.eos_token is None:
        full_text = prompt + response
    else:
        full_text = prompt + response + tokenizer.eos_token

    prompt_enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    full_enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    labels = input_ids.copy()

    prompt_len = len(prompt_enc["input_ids"])
    prompt_len = min(prompt_len, len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_task_dataset(
    task: str,
    tokenizer,
    max_length: int = 512,
    seed: int = 42,
    max_train: Optional[int] = None,
    max_eval: Optional[int] = None,
):
    train_raw, eval_raw = load_task_dataset(
        task=task,
        seed=seed,
        max_train=max_train,
        max_eval=max_eval,
    )

    def preprocess(example):
        return tokenize_prompt_response(example, tokenizer, max_length=max_length)

    train_ds = train_raw.map(
        preprocess,
        remove_columns=train_raw.column_names,
        desc=f"Tokenizing train[{normalize_task_name(task)}]",
    )
    eval_ds = eval_raw.map(
        preprocess,
        remove_columns=eval_raw.column_names,
        desc=f"Tokenizing eval[{normalize_task_name(task)}]",
    )

    return train_ds, eval_ds


@dataclass
class CausalLMDataCollator:
    
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            }
            for f in features
        ]

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]

        labels = []

        for f in features:
            label = f["labels"]
            pad_len = max_len - len(label)

            if pad_len < 0:
                label = label[:max_len]
                pad_len = 0

            label = label + [self.label_pad_token_id] * pad_len
            labels.append(label)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


# ============================================================
# Train
# ============================================================

def train(args):
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None

    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None,
        device_map="auto" if args.use_4bit else None,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --------------------------------------------------------
    # mask.pt에서 target_modules 자동 추출
    # --------------------------------------------------------
    mask_pack = torch.load(args.mask_path, map_location="cpu")
    
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    print(f"[LoRA] target_modules = {target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --------------------------------------------------------
    # mask.pt 기반 gradient hook 등록
    # --------------------------------------------------------
    apply_lora_mask_hooks(
        model=model,
        mask_path=args.mask_path,
        adapter_name=args.adapter_name,
    )

    apply_lora_mask_hooks(
        model=model,
        mask_path=args.mask_path,
        adapter_name=args.adapter_name,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    train_ds, eval_ds = build_task_dataset(
        task=args.task,
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        max_train=args.max_train,
        max_eval=args.max_eval,
    )

    print(f"[DATA] task={normalize_task_name(args.task)} train={len(train_ds)} eval={len(eval_ds)}")

    data_collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    torch.save(mask_pack, os.path.join(args.output_dir, "mask.pt"))

    print(f"[DONE] saved masked LoRA adapter to: {args.output_dir}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="sentiment",
        choices=["sentiment", "summarize", "sum", "qa", "math"],
        help="Training/eval task. sum is an alias of summarize.",
    )

    parser.add_argument("--adapter_name", type=str, default="default")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train", type=int, default=None, help="Cap train samples. Use -1 or omit for full train split.")
    parser.add_argument("--max_eval", type=int, default=None, help="Cap eval samples. Use -1 or omit for full eval split.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--logging_steps", type=int, default=20)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
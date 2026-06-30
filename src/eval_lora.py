#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import string
from collections import Counter
from typing import Optional, List

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel



def extract_qa_answer(text: str) -> str:
    text = text.strip()

    # 혹시 모델이 Answer: 같은 마커를 다시 출력한 경우
    markers = [
        "### Answer:",
        "Answer:",
        "answer:",
    ]
    for m in markers:
        if m in text:
            text = text.split(m, 1)[-1].strip()

    # 다음 섹션을 생성한 경우 자르기
    stop_markers = [
        "\n###",
        "\nContext:",
        "\nQuestion:",
        "\nExplanation:",
        "\nSolution:",
    ]
    for s in stop_markers:
        if s in text:
            text = text.split(s, 1)[0].strip()

    # QA는 보통 짧은 span이므로 첫 줄만 사용
    text = text.split("\n")[0].strip()

    # 따옴표 제거
    text = text.strip(" \"'`")

    return text


def load_sentiment_dataset(seed: int = 42, max_eval: Optional[int] = None):
    ds = load_dataset("glue", "sst2")
    eval_raw = ds["validation"]

    def map_fn(ex):
        sent = ex["sentence"].strip()
        label = int(ex["label"])
        answer = "positive" if label == 1 else "negative"

        prompt = f"Classify the sentiment.\n\n### Text:\n{sent}\n\n### Sentiment:\n"

        return {
            "prompt": prompt,
            "label": label,
            "answer": answer,
            "input_text": sent,
        }

    eval_ds = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    if max_eval is not None and max_eval >= 0:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    return eval_ds


def load_summarize_dataset(seed: int = 42, max_eval: Optional[int] = None):
    ds = load_dataset("EdinburghNLP/xsum")
    eval_raw = ds["test"]

    def map_fn(ex):
        document = ex["document"].strip()
        summary = ex["summary"].strip()

        prompt = f"Summarize the following document.\n\n### Document:\n{document}\n\n### Summary:\n"

        return {
            "prompt": prompt,
            "answer": summary,
            "input_text": document,
        }

    eval_ds = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    if max_eval is not None and max_eval >= 0:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    return eval_ds


def load_math_dataset(seed: int = 42, max_eval: Optional[int] = None):
    ds = load_dataset("gsm8k", "main")
    eval_raw = ds["test"]

    def extract_gsm8k_answer(answer_text: str):
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip()
        return answer_text.strip()

    def map_fn(ex):
        question = ex["question"].strip()
        answer_full = ex["answer"].strip()
        final_answer = extract_gsm8k_answer(answer_full)

        prompt = f"Solve the problem step by step.\n\n### Problem:\n{question}\n\n### Solution:\n"

        return {
            "prompt": prompt,
            "answer": final_answer,
            "answer_full": answer_full,
            "input_text": question,
        }

    eval_ds = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    if max_eval is not None and max_eval >= 0:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    return eval_ds


def load_qa_dataset(seed: int = 42, max_eval: Optional[int] = None):
    ds = load_dataset("squad_v2")
    eval_raw = ds["validation"]

    eval_raw = eval_raw.filter(lambda ex: len(ex["answers"]["text"]) > 0)

    def map_fn(ex):
        context = ex["context"].strip()
        question = ex["question"].strip()
        gold_answers = [a.strip() for a in ex["answers"]["text"] if a.strip()]
        answer = gold_answers[0]

        prompt = f"Answer the question using the context.\n\n### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n"

        return {
            "prompt": prompt,
            "answer": answer,
            "gold_answers": gold_answers,
            "input_text": question,
            "context": context,
        }

    eval_ds = eval_raw.map(map_fn, remove_columns=eval_raw.column_names)

    if max_eval is not None and max_eval >= 0:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    return eval_ds


def load_eval_dataset(task: str, seed: int = 42, max_eval: Optional[int] = None):
    if task == "sentiment":
        return load_sentiment_dataset(seed=seed, max_eval=max_eval)
    if task == "summarize":
        return load_summarize_dataset(seed=seed, max_eval=max_eval)
    if task == "math":
        return load_math_dataset(seed=seed, max_eval=max_eval)
    if task == "qa":
        return load_qa_dataset(seed=seed, max_eval=max_eval)

    raise ValueError(f"Unknown task: {task}")


# =========================================================
# Generation
# =========================================================

def get_model_input_device(model):
    """
    With device_map='auto', the model can be dispatched across devices.
    For normal single-GPU use, putting inputs on cuda:0 is correct.
    If CUDA is unavailable, fall back to the first parameter device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return next(model.parameters()).device


@torch.no_grad()
def generate_text_batch(model, tokenizer, prompts: List[str], max_new_tokens: int):
    device = get_model_input_device(model)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    input_len = inputs["input_ids"].shape[1]

    gen_texts = []
    for output_ids in outputs:
        gen_ids = output_ids[input_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gen_texts.append(gen_text.strip())

    return gen_texts


# =========================================================
# Metric utils
# =========================================================

def normalize_sentiment_pred(text: str):
    text = text.strip().lower()

    if text.startswith("positive"):
        return 1, "positive"
    if text.startswith("negative"):
        return 0, "negative"

    first_words = text.replace("\n", " ").split()
    joined = " ".join(first_words[:10])

    if "positive" in joined and "negative" not in joined:
        return 1, "positive"
    if "negative" in joined and "positive" not in joined:
        return 0, "negative"

    return -1, text


def binary_acc_macro_f1(y_true: List[int], y_pred: List[int]):
    total = len(y_true)
    correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
    acc = correct / total if total > 0 else 0.0

    def class_f1(label: int):
        tp = sum((yt == label) and (yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label) and (yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label) and (yp != label) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    f1_negative = class_f1(0)
    f1_positive = class_f1(1)
    macro_f1 = (f1_negative + f1_positive) / 2

    return acc, macro_f1


def normalize_answer_for_qa(s: str):
    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str):
    return normalize_answer_for_qa(prediction) == normalize_answer_for_qa(ground_truth)


def f1_score(prediction: str, ground_truth: str):
    pred_tokens = normalize_answer_for_qa(prediction).split()
    gold_tokens = normalize_answer_for_qa(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    if len(ground_truths) == 0:
        ground_truths = [""]

    return max(metric_fn(prediction, gt) for gt in ground_truths)


def normalize_math_answer(text: str):
    text = text.strip()

    if "####" in text:
        text = text.split("####")[-1].strip()

    text = text.replace(",", "")

    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) > 0:
        return numbers[-1]

    return text.lower().strip()


def iter_batches(eval_ds, batch_size: int):
    for start in range(0, len(eval_ds), batch_size):
        end = min(start + batch_size, len(eval_ds))
        yield [eval_ds[i] for i in range(start, end)]


# =========================================================
# Evaluators
# =========================================================

@torch.no_grad()
def evaluate_sentiment(model, tokenizer, eval_ds, max_new_tokens: int, batch_size: int):
    model.eval()

    preds = []
    y_true = []
    y_pred = []

    for batch in tqdm(iter_batches(eval_ds, batch_size), total=(len(eval_ds) + batch_size - 1) // batch_size, desc="Evaluating sentiment"):
        prompts = [ex["prompt"] for ex in batch]
        gen_texts = generate_text_batch(model, tokenizer, prompts, max_new_tokens)

        for ex, gen_text in zip(batch, gen_texts):
            gold_label = int(ex["label"])
            gold_text = ex["answer"]
            pred_label, pred_text = normalize_sentiment_pred(gen_text)

            y_true.append(gold_label)
            y_pred.append(pred_label)

            preds.append({
                "input_text": ex["input_text"],
                "prompt": ex["prompt"],
                "gold_label": gold_label,
                "gold_text": gold_text,
                "raw_generation": gen_text,
                "pred_label": pred_label,
                "pred_text": pred_text,
                "correct": pred_label == gold_label,
            })

    acc, f1 = binary_acc_macro_f1(y_true=y_true, y_pred=y_pred)

    return {
        "metric_name": "accuracy_f1",
        "accuracy": acc,
        "f1": f1,
        "total": len(y_true),
        "predictions": preds,
    }


@torch.no_grad()
def evaluate_summarize(model, tokenizer, eval_ds, max_new_tokens: int, batch_size: int):
    model.eval()

    predictions = []
    references = []
    preds = []

    for batch in tqdm(iter_batches(eval_ds, batch_size), total=(len(eval_ds) + batch_size - 1) // batch_size, desc="Evaluating summarize"):
        prompts = [ex["prompt"] for ex in batch]
        gen_texts = generate_text_batch(model, tokenizer, prompts, max_new_tokens)

        for ex, gen_text in zip(batch, gen_texts):
            gold = ex["answer"]
            predictions.append(gen_text)
            references.append(gold)

            preds.append({
                "input_text": ex["input_text"],
                "prompt": ex["prompt"],
                "gold_text": gold,
                "raw_generation": gen_text,
            })

    try:
        import evaluate
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )

        rouge1 = float(rouge_result["rouge1"])
        rouge2 = float(rouge_result["rouge2"])
        rougeL = float(rouge_result["rougeL"])

    except Exception as e:
        print("[warning] ROUGE calculation failed.")
        print("[warning] Install with: pip install evaluate rouge_score")
        print(f"[warning] Error: {e}")

        rouge1 = None
        rouge2 = None
        rougeL = None

    return {
        "metric_name": "rouge",
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "total": len(eval_ds),
        "predictions": preds,
    }


@torch.no_grad()
def evaluate_math(model, tokenizer, eval_ds, max_new_tokens: int, batch_size: int):
    model.eval()

    correct = 0
    total = 0
    preds = []

    for batch in tqdm(iter_batches(eval_ds, batch_size), total=(len(eval_ds) + batch_size - 1) // batch_size, desc="Evaluating math"):
        prompts = [ex["prompt"] for ex in batch]
        gen_texts = generate_text_batch(model, tokenizer, prompts, max_new_tokens)

        for ex, gen_text in zip(batch, gen_texts):
            gold = normalize_math_answer(ex["answer"])
            pred = normalize_math_answer(gen_text)
            is_correct = pred == gold

            correct += int(is_correct)
            total += 1

            preds.append({
                "input_text": ex["input_text"],
                "prompt": ex["prompt"],
                "gold_text": ex["answer"],
                "gold_normalized": gold,
                "raw_generation": gen_text,
                "pred_normalized": pred,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0

    return {
        "metric_name": "exact_match_accuracy",
        "exact_match": accuracy,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": preds,
    }


@torch.no_grad()
def evaluate_qa(model, tokenizer, eval_ds, max_new_tokens: int, batch_size: int):
    model.eval()

    exact_sum = 0.0
    f1_sum = 0.0
    total = 0
    preds = []

    for batch in tqdm(iter_batches(eval_ds, batch_size), total=(len(eval_ds) + batch_size - 1) // batch_size, desc="Evaluating qa"):
        prompts = [ex["prompt"] for ex in batch]
        gen_texts = generate_text_batch(model, tokenizer, prompts, max_new_tokens)

        for ex, gen_text in zip(batch, gen_texts):
            gold_answers = list(ex["gold_answers"])
            pred_answer = extract_qa_answer(gen_text)

            em = metric_max_over_ground_truths(
                exact_match_score,
                pred_answer,
                gold_answers,
            )
            f1 = metric_max_over_ground_truths(
                f1_score,
                pred_answer,
                gold_answers,
            )

            exact_sum += float(em)
            f1_sum += float(f1)
            total += 1

            preds.append({
                "question": ex["input_text"],
                "context": ex["context"],
                "prompt": ex["prompt"],
                "gold_answers": gold_answers,
                "raw_generation": gen_text,
                "pred_answer_for_metric": pred_answer,
                "exact_match": float(em),
                "f1": float(f1),
            })

    exact_match = exact_sum / total if total > 0 else 0.0
    f1 = f1_sum / total if total > 0 else 0.0

    return {
        "metric_name": "exact_match_and_f1",
        "exact_match": exact_match,
        "f1": f1,
        "total": total,
        "predictions": preds,
    }


def evaluate_task(model, tokenizer, eval_ds, task: str, max_new_tokens: int, batch_size: int):
    if task == "sentiment":
        return evaluate_sentiment(model, tokenizer, eval_ds, max_new_tokens, batch_size)
    if task == "summarize":
        return evaluate_summarize(model, tokenizer, eval_ds, max_new_tokens, batch_size)
    if task == "math":
        return evaluate_math(model, tokenizer, eval_ds, max_new_tokens, batch_size)
    if task == "qa":
        return evaluate_qa(model, tokenizer, eval_ds, max_new_tokens, batch_size)

    raise ValueError(f"Unknown task: {task}")


def default_max_new_tokens(task: str):
    if task == "sentiment":
        return 8
    if task == "summarize":
        return 128
    if task == "math":
        return 256
    if task == "qa":
        return 64
    return 128


# =========================================================
# Main
# =========================================================

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--lora_path", type=str, required=True)

    p.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["sentiment", "summarize", "math", "qa"],
    )

    p.add_argument("--max_eval", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--max_new_tokens", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--save_name", type=str, default=None)

    args = p.parse_args()
    set_seed(args.seed)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    if args.max_new_tokens < 0:
        args.max_new_tokens = default_max_new_tokens(args.task)

    if args.save_name is None:
        args.save_name = f"eval_{args.task}.json"

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Important for batched generation with decoder-only LMs.
    tokenizer.padding_side = "left"

    model_kwargs = {"device_map": "auto"}

    if args.use_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    dtype = (
        torch.bfloat16
        if args.bf16
        else torch.float16
        if args.fp16
        else torch.float32
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        **model_kwargs,
    )

    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"input device: {get_model_input_device(model)}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"hf_device_map: {getattr(model, 'hf_device_map', None)}")

    max_eval = None if args.max_eval < 0 else args.max_eval

    eval_ds = load_eval_dataset(
        task=args.task,
        seed=args.seed,
        max_eval=max_eval,
    )

    result = evaluate_task(
        model=model,
        tokenizer=tokenizer,
        eval_ds=eval_ds,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    result["base_model"] = args.base_model
    result["lora_path"] = args.lora_path
    result["task"] = args.task
    result["seed"] = args.seed
    result["max_eval"] = args.max_eval
    result["max_new_tokens"] = args.max_new_tokens
    result["batch_size"] = args.batch_size

    save_path = os.path.join(args.lora_path, args.save_name)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n[done] saved eval result to: {save_path}")
    print(f"task: {args.task}")
    print(f"metric_name: {result['metric_name']}")

    if args.task == "sentiment":
        print(f"accuracy: {result['accuracy']:.4f}")
        print(f"f1: {result['f1']:.4f}")
        print(f"total: {result['total']}")

    elif args.task == "summarize":
        print(f"rouge1: {result['rouge1']}")
        print(f"rouge2: {result['rouge2']}")
        print(f"rougeL: {result['rougeL']}")
        print(f"total: {result['total']}")

    elif args.task == "math":
        print(f"exact_match/accuracy: {result['accuracy']:.4f}")
        print(f"correct: {result['correct']} / {result['total']}")

    elif args.task == "qa":
        print(f"exact_match: {result['exact_match']:.4f}")
        print(f"f1: {result['f1']:.4f}")
        print(f"total: {result['total']}")


if __name__ == "__main__":
    main()
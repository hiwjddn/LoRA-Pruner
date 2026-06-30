#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import torch


QO_PROJS = {"q", "o"}
MLP_PROJS = {"up", "down", "gate"}
EXCLUDE_PROJS = {"k", "v"}


def strip_weight_suffix(param_name: str) -> str:
    if param_name.endswith(".weight"):
        return param_name[: -len(".weight")]
    return param_name


def topk_count(num_candidates: int, top_ratio: float) -> int:
    if num_candidates <= 0:
        return 0
    return max(1, math.ceil(num_candidates * top_ratio))


def load_metric_records(metric_jsonl: str) -> List[Dict[str, Any]]:
    records = []

    with open(metric_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] skip invalid JSON line {line_idx}")
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def make_lora_mask(
    metric_jsonl: str,
    output_path: str,
    num_heads: int,
    top_ratio: float = 0.10,
    metric_key: str = "out_gram_rel_fro",
) -> Dict[str, Any]:

    if not (0.0 < top_ratio <= 1.0):
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")

    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    records = load_metric_records(metric_jsonl)

    qo_candidates: List[Dict[str, Any]] = []
    mlp_candidates: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Collect valid candidates
    # ------------------------------------------------------------------
    for record in records:
        param_name = record.get("param_name")
        proj = record.get("proj")

        if param_name is None or proj is None:
            continue

        if metric_key not in record:
            continue

        if proj in EXCLUDE_PROJS:
            continue

        try:
            score = float(record[metric_key])
        except (TypeError, ValueError):
            continue

        module_name = strip_weight_suffix(param_name)

        if proj in QO_PROJS:
            if "head" not in record:
                continue

            try:
                head = int(record["head"])
            except (TypeError, ValueError):
                continue

            if head < 0 or head >= num_heads:
                print(
                    f"[WARN] skip invalid head index: "
                    f"module={module_name}, head={head}, num_heads={num_heads}"
                )
                continue

            qo_candidates.append(
                {
                    "module_name": module_name,
                    "proj": proj,
                    "head": head,
                    "score": score,
                }
            )

        elif proj in MLP_PROJS:
            mlp_candidates.append(
                {
                    "module_name": module_name,
                    "proj": proj,
                    "score": score,
                }
            )

        else:
            continue

    # ------------------------------------------------------------------
    # 2. Select global top-k separately for q and o heads
    # ------------------------------------------------------------------
    q_candidates = [item for item in qo_candidates if item["proj"] == "q"]
    o_candidates = [item for item in qo_candidates if item["proj"] == "o"]

    q_k = topk_count(len(q_candidates), top_ratio)
    o_k = topk_count(len(o_candidates), top_ratio)

    q_selected = sorted(
        q_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )[:q_k]

    o_selected = sorted(
        o_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )[:o_k]

    qo_selected = q_selected + o_selected

    selected_qo_pairs = {
        (item["module_name"], item["head"]) for item in qo_selected
    }

    # ------------------------------------------------------------------
    # 3. Select global top-k for up/down/gate modules
    # ------------------------------------------------------------------
    mlp_k = topk_count(len(mlp_candidates), top_ratio)

    mlp_selected = sorted(
        mlp_candidates,
        key=lambda x: x["score"],
        reverse=True,
    )[:mlp_k]

    selected_mlp_modules = {
        item["module_name"] for item in mlp_selected
    }

    # ------------------------------------------------------------------
    # 4. Build mask pack
    # ------------------------------------------------------------------
    masks: Dict[str, Dict[str, Any]] = {}

    qo_modules: Dict[str, str] = {}
    for item in qo_candidates:
        qo_modules[item["module_name"]] = item["proj"]

    for module_name, proj in sorted(qo_modules.items()):
        head_mask = torch.zeros(num_heads, dtype=torch.bool)

        for head_idx in range(num_heads):
            if (module_name, head_idx) in selected_qo_pairs:
                head_mask[head_idx] = True

        masks[module_name] = {
            "proj": proj,
            "type": "head",
            "head_mask": head_mask,
        }

    mlp_modules: Dict[str, str] = {}
    for item in mlp_candidates:
        mlp_modules[item["module_name"]] = item["proj"]

    for module_name, proj in sorted(mlp_modules.items()):
        masks[module_name] = {
            "proj": proj,
            "type": "module",
            "enabled": module_name in selected_mlp_modules,
        }

    mask_pack = {
        "version": 1,
        "metric_key": metric_key,
        "top_ratio": top_ratio,
        "num_heads": num_heads,
        "masks": masks,
        "stats": {
            "q_total_candidates": len(q_candidates),
            "q_selected": len(q_selected),
            "o_total_candidates": len(o_candidates),
            "o_selected": len(o_selected),
            "qo_total_candidates": len(qo_candidates),
            "qo_selected": len(qo_selected),
            "mlp_total_candidates": len(mlp_candidates),
            "mlp_selected": len(mlp_selected),
        },
    }

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(mask_pack, output_path)

    print(f"[DONE] saved mask to: {output_path}")
    print(f"q total candidates:   {len(q_candidates)}")
    print(f"q selected:           {len(q_selected)}")
    print(f"o total candidates:   {len(o_candidates)}")
    print(f"o selected:           {len(o_selected)}")
    print(f"q/o total candidates: {len(qo_candidates)}")
    print(f"q/o selected:         {len(qo_selected)}")
    print(f"MLP total candidates: {len(mlp_candidates)}")
    print(f"MLP selected:         {len(mlp_selected)}")

    return mask_pack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create LoRA training mask.pt from metric JSONL."
    )

    parser.add_argument(
        "--metric_jsonl",
        type=str,
        required=True,
        help="Path to metric JSONL file.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output mask.pt file.",
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        required=True,
        help="Number of attention heads for q/o head masks.",
    )

    parser.add_argument(
        "--top_ratio",
        type=float,
        default=0.10,
        help="Top ratio to select. Default: 0.10",
    )

    parser.add_argument(
        "--metric_key",
        type=str,
        default="out_gram_rel_fro",
        help="Metric key used for ranking. Default: out_gram_rel_fro",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    make_lora_mask(
        metric_jsonl=args.metric_jsonl,
        output_path=args.output_path,
        num_heads=args.num_heads,
        top_ratio=args.top_ratio,
        metric_key=args.metric_key,
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from safetensors import safe_open
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel, AutoConfig
from peft import PeftModel


# -------------------------
# Utils: linear algebra
# -------------------------

def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.clamp(torch.linalg.norm(x), min=eps)

def frob_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b)

def operator_cosine(W: torch.Tensor, Wp: torch.Tensor, eps: float = 1e-12) -> float:
    num = frob_inner(W, Wp)
    den = _safe_norm(W, eps) * _safe_norm(Wp, eps)
    return (num / den).item()

def rel_fro_error(W: torch.Tensor, Wp: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.norm(W - Wp)
    den = _safe_norm(W, eps)
    return (num / den).item()

def out_gram_rel_fro(W: torch.Tensor, Wp: torch.Tensor, eps: float = 1e-12) -> float:
    # output covariance under isotropic input: C = W W^T
    C = W @ W.T
    Cp = Wp @ Wp.T
    num = torch.linalg.norm(C - Cp)
    den = _safe_norm(C, eps)
    return (num / den).item()

def singular_values(W: torch.Tensor) -> torch.Tensor:
    # returns descending sv
    return torch.linalg.svdvals(W)

def sv_l2(W: torch.Tensor, Wp: torch.Tensor, eps: float = 1e-12) -> float:
    s = singular_values(W)
    sp = singular_values(Wp)
    # match lengths (should match, but just in case)
    m = min(s.numel(), sp.numel())
    s = s[:m]
    sp = sp[:m]
    num = torch.linalg.norm(s - sp)
    den = _safe_norm(s, eps)
    return (num / den).item()

def sv_cosine(W: torch.Tensor, Wp: torch.Tensor, eps: float = 1e-12) -> float:
    s = singular_values(W)
    sp = singular_values(Wp)
    m = min(s.numel(), sp.numel())
    s = s[:m]
    sp = sp[:m]
    num = torch.dot(s, sp)
    den = _safe_norm(s, eps) * _safe_norm(sp, eps)
    return (num / den).item()

def topk_singular_subspaces(W: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # full_matrices=False for efficiency
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    k = min(k, U.shape[1], Vh.shape[0])
    U_k = U[:, :k]
    V_k = Vh[:k, :].T  # (in, k)
    return U_k, V_k

def subspace_sinF(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> float:
    # Ensure same k by truncation
    k = min(A.shape[1], B.shape[1])
    A = A[:, :k]
    B = B[:, :k]
    # sv of cross-correlation
    M = A.T @ B
    s = torch.linalg.svdvals(M)
    s = torch.clamp(s, 0.0, 1.0)
    sin_sq = 1.0 - s**2
    return torch.sqrt(torch.sum(sin_sq)).item()

def left_subspace_sinF(W: torch.Tensor, Wp: torch.Tensor, k: int) -> float:
    U, _ = topk_singular_subspaces(W, k)
    Up, _ = topk_singular_subspaces(Wp, k)
    return subspace_sinF(U, Up)

def right_subspace_sinF(W: torch.Tensor, Wp: torch.Tensor, k: int) -> float:
    _, V = topk_singular_subspaces(W, k)
    _, Vp = topk_singular_subspaces(Wp, k)
    return subspace_sinF(V, Vp)

def update_alignment(W: torch.Tensor, Wp: torch.Tensor, eps: float = 1e-12) -> float:
    dW = Wp - W
    num = frob_inner(W, dW)
    den = _safe_norm(W, eps) * _safe_norm(dW, eps)
    return (num / den).item()


# -------------------------
# Metrics registry (8 total)
# -------------------------

METRIC_FUNCS = {
    "operator_cosine": lambda W, Wp, cfg: operator_cosine(W, Wp, cfg.eps), # frobenius cosine
    "rel_fro_error": lambda W, Wp, cfg: rel_fro_error(W, Wp, cfg.eps), # frobenius error
    "out_gram_rel_fro": lambda W, Wp, cfg: out_gram_rel_fro(W, Wp, cfg.eps), # C_w = WW^T, C_w' = W'W'^T, frobenius error between C_w, C_w'
    "sv_l2": lambda W, Wp, cfg: sv_l2(W, Wp, cfg.eps), #
    "sv_cosine": lambda W, Wp, cfg: sv_cosine(W, Wp, cfg.eps),
    "left_subspace_sinF": lambda W, Wp, cfg: left_subspace_sinF(W, Wp, cfg.subspace_k), # |sim(U, U')|_f
    "right_subspace_sinF": lambda W, Wp, cfg: right_subspace_sinF(W, Wp, cfg.subspace_k), # |sim(V, V')|_f
    "update_alignment": lambda W, Wp, cfg: update_alignment(W, Wp, cfg.eps),
}

ALL_METRICS = list(METRIC_FUNCS.keys())


# -------------------------
# Model scanning & head slicing
# -------------------------

@dataclass
class CompareConfig:
    eps: float = 1e-12
    subspace_k: int = 8
    device: str = "cuda"
    dtype: str = "float32"
    metrics: Optional[List[str]] = None
    max_layers: Optional[int] = None  # optionally limit layers for speed
    verbose: bool = False


def _to_dtype(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def infer_attention_hparams(model) -> Tuple[int, int, int]:
    cfg = model.config
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None) or getattr(cfg, "n_embd", None)
    nheads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if hidden is None or nheads is None:
        raise ValueError("Cannot infer hidden_size/num_attention_heads from model.config.")
    head_dim = hidden // nheads
    return int(nheads), int(hidden), int(head_dim)


def is_linear_weight(param: torch.Tensor) -> bool:
    return param.ndim == 2


def detect_proj_kind(name: str) -> Optional[str]:
    n = name.lower()
    # Llama-like
    if n.endswith("q_proj.weight") or ".q_proj.weight" in n or n.endswith("query.weight") or ".query.weight" in n:
        return "q"
    if n.endswith("k_proj.weight") or ".k_proj.weight" in n or n.endswith("key.weight") or ".key.weight" in n:
        return "k"
    if n.endswith("v_proj.weight") or ".v_proj.weight" in n or n.endswith("value.weight") or ".value.weight" in n:
        return "v"
    if n.endswith("o_proj.weight") or ".o_proj.weight" in n or n.endswith("out_proj.weight") or ".output.dense.weight" in n:
        return "o"
    # GPT2 fused qkv
    if n.endswith("c_attn.weight") or ".c_attn.weight" in n:
        return "qkv"
    return None


def extract_heads_from_weight(
    W: torch.Tensor,
    proj_kind: str,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
) -> List[Tuple[str, int, torch.Tensor]]:

    out_dim, in_dim = W.shape
    res = []

    if proj_kind in ("q", "k", "v"):
        if out_dim != hidden_size or out_dim != num_heads * head_dim:
            # try alternative: some models store (in,out) style; user can transpose externally if needed
            raise ValueError(f"Unexpected shape for {proj_kind}: {W.shape}, expected out_dim==hidden_size.")
        for h in range(num_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            res.append((proj_kind, h, W[sl, :].contiguous()))
        return res

    if proj_kind == "o":
        if in_dim != hidden_size or in_dim != num_heads * head_dim:
            raise ValueError(f"Unexpected shape for o: {W.shape}, expected in_dim==hidden_size.")
        for h in range(num_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            res.append((proj_kind, h, W[:, sl].contiguous()))
        return res

    if proj_kind == "qkv":
        # GPT2 c_attn: out = 3*hidden_size, in=hidden_size
        if out_dim != 3 * hidden_size:
            raise ValueError(f"Unexpected shape for fused qkv: {W.shape}, expected out_dim==3*hidden_size.")
        # chunks: q,k,v each (hidden_size, in_dim)
        Wq = W[0:hidden_size, :]
        Wk = W[hidden_size:2*hidden_size, :]
        Wv = W[2*hidden_size:3*hidden_size, :]
        for subkind, Wsub in [("q", Wq), ("k", Wk), ("v", Wv)]:
            if Wsub.shape[0] != num_heads * head_dim:
                raise ValueError("Hidden size not divisible by heads for qkv slicing.")
            for h in range(num_heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                res.append((subkind, h, Wsub[sl, :].contiguous()))
        return res

    return res


def collect_head_sliceable_params(state_dict: Dict[str, torch.Tensor]) -> List[str]:

    names = []
    for k, v in state_dict.items():
        if not is_linear_weight(v):
            continue
        pk = detect_proj_kind(k)
        if pk is None:
            continue
        names.append(k)
    return sorted(names)


# -------------------------
# Main comparison logic
# -------------------------

@torch.no_grad()
def compare_models_heads(
    base_model,
    tuned_model_merged,
    cfg: CompareConfig,
) -> List[Dict]:
    device = torch.device(cfg.device)
    dtype = _to_dtype(cfg.dtype)

    # Pull state dicts (on CPU), then move head blocks to device for compute
    base_sd = base_model.state_dict()
    tuned_sd = tuned_model_merged.state_dict()

    num_heads, hidden_size, head_dim = infer_attention_hparams(base_model)

    # Pick head-sliceable weights based on base_sd keys intersection with tuned_sd
    cand = [k for k in collect_head_sliceable_params(base_sd) if k in tuned_sd]

    # Optional: limit layers (rough heuristic)
    if cfg.max_layers is not None:
        # keep only keys whose layer index < max_layers if pattern exists
        kept = []
        for k in cand:
            m = re.search(r"\.layers\.(\d+)\.", k)
            if m and int(m.group(1)) >= cfg.max_layers:
                continue
            kept.append(k)
        cand = kept

    if cfg.verbose:
        print(f"[info] inferred: num_heads={num_heads}, hidden={hidden_size}, head_dim={head_dim}")
        print(f"[info] found {len(cand)} head-sliceable projection weights")

    metrics = cfg.metrics or ALL_METRICS
    for m in metrics:
        if m not in METRIC_FUNCS:
            raise ValueError(f"Unknown metric '{m}'. Available: {ALL_METRICS}")

    results: List[Dict] = []

    for name in cand:
        proj_kind = detect_proj_kind(name)
        if proj_kind is None:
            continue

        W = base_sd[name].to(device=device, dtype=dtype)
        Wp = tuned_sd[name].to(device=device, dtype=dtype)

        # Head slicing (may produce multiple subkinds for fused qkv)
        try:
            heads_W = extract_heads_from_weight(W, proj_kind, num_heads, head_dim, hidden_size)
            heads_Wp = extract_heads_from_weight(Wp, proj_kind, num_heads, head_dim, hidden_size)
        except Exception as e:
            if cfg.verbose:
                print(f"[warn] skip {name} due to slicing error: {e}")
            continue

        # Build mapping (subkind, head)->matrix
        mapW = {(sk, h): mat for (sk, h, mat) in heads_W}
        mapWp = {(sk, h): mat for (sk, h, mat) in heads_Wp}
        keys = sorted(set(mapW.keys()) & set(mapWp.keys()))

        for (sk, h) in keys:
            Wh = mapW[(sk, h)]
            Wph = mapWp[(sk, h)]

            entry = {
                "param_name": name,
                "proj": sk,            # q/k/v/o (even if fused)
                "head": h,
                "shape": list(Wh.shape),
            }

            for met in metrics:
                entry[met] = float(METRIC_FUNCS[met](Wh, Wph, cfg))

            results.append(entry)

    return results


def load_base_model(base_model_name_or_path: str, device: str, dtype: str):
    torch_dtype = _to_dtype(dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        return model
    except Exception:
        # fallback
        model = AutoModel.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        return model


def load_peft_and_merge(base_model, adapter_path: str):
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = peft_model.merge_and_unload()  # returns base class model with weights merged
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True, help="Base HF model name/path")
    # ap.add_argument("--adapter", type=str, required=True, help="PEFT adapter path (LoRA)")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    base_for_merge = load_base_model(args.base_model, device=args.device, dtype=args.dtype)
    base_for_merge.eval()

    for k in base_for_merge.state_dict():
        print(k)


if __name__ == "__main__":
    main()

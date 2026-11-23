#!/usr/bin/env python3
"""
Phase 7: Attack 1 - Closed-Form Adversarial Perturbation (SVD vs Random Control).
Compares white-box SVD dominant-singular-vector perturbation against matched-norm
random noise on the cross-model KV transfer pathway across 30 GSM8K examples.
"""

import os
import sys
import time
import re
import csv
import argparse
from typing import List, Dict, Tuple, Optional, Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Add platform/LatentMAS to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from data import load_gsm8k
from utils import normalize_answer, extract_gsm8k_answer
from cross_model_mapper import CrossModelKVMapper, strip_rope, apply_rope

def check_output_health(raw_text: str, pred: Optional[str]) -> Tuple[bool, str]:
    if not raw_text or len(raw_text.strip()) < 15:
        return False, "premature_termination"
    has_boxed = bool(re.search(r"\\boxed\{([^}]+)\}", raw_text))
    if not has_boxed:
        return False, "missing_boxed"
    if pred is None or len(str(pred).strip()) == 0:
        return False, "no_valid_prediction"
    if re.search(r"(\b\w+\b[\s,.]*)\1{8,}", raw_text):
        return False, "repetitive_loop"
    if re.search(r"([!?.=,;:\-\+])\1{10,}", raw_text):
        return False, "repetitive_punctuation"
    return True, "healthy"

def build_gsm8k_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                "Please reason step-by-step to solve the question. "
                "Output your final answer inside \\boxed{YOUR_FINAL_ANSWER}."
            )
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def precompute_svd_directions(mapper: CrossModelKVMapper, device: torch.device) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Precomputes the unit-norm dominant left-singular vector u_1 of W_k for each (layer, head).
    Since mapped keys are y = x @ W_k, the direction of maximum amplification is the
    first column of U in W_k = U @ Sigma @ Vh.
    """
    svd_dirs = {}
    for l in range(mapper.num_layers):
        for h in range(mapper.num_kv_heads):
            W = mapper.weights_k[(l, h)].float()
            U, S, Vh = torch.linalg.svd(W)
            u1 = U[:, 0].to(device)  # shape [128], norm = 1.0
            svd_dirs[(l, h)] = u1
    return svd_dirs

def get_random_direction(
    device: torch.device,
    head_dim: int,
    seed: int,
    req_id: Any,
    layer_idx: int,
    head_idx: int,
) -> torch.Tensor:
    """
    Generates a deterministic pseudo-random unit vector for (req_id, layer_idx, head_idx).
    """
    req_int = int(req_id) if str(req_id).isdigit() else hash(str(req_id)) % 100000
    seed_val = (seed * 1000003 + req_int * 1009 + int(layer_idx) * 31 + int(head_idx) * 7) % (2**31 - 1)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_val)
    z = torch.randn(head_dim, generator=g, dtype=torch.float32)
    z = z / z.norm().clamp_min(1e-8)
    return z.to(device)

def map_cache_with_attack(
    src_cache: DynamicCache,
    mapper: CrossModelKVMapper,
    svd_dirs: Dict[Tuple[int, int], torch.Tensor],
    src_rotary_emb,
    tgt_rotary_emb,
    device: torch.device,
    attack_type: str = "clean",  # "clean", "svd", "random"
    norm: float = 0.0,
    seed: int = 42,
    req_id: Any = 0,
    carrier: str = "K",
    target_dtype: torch.dtype = torch.bfloat16,
) -> DynamicCache:
    """
    Maps source DynamicCache to target DynamicCache with optional closed-form perturbation.
    Perturbation delta is added to unrotated source representation:
      - "svd": delta = norm * u_1 (dominant singular vector of W_k)
      - "random": delta = norm * z / ||z||_2 (random unit vector)
      - "clean": delta = 0
    Both SVD and Random perturbations have EXACTLY matched L2 norm = norm.
    """
    tgt_cache = DynamicCache()
    seq_len = src_cache.get_seq_length()
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    dummy_k = src_cache.layers[0].keys
    cos_src, sin_src = src_rotary_emb(dummy_k, pos_ids)
    cos_tgt, sin_tgt = tgt_rotary_emb(dummy_k, pos_ids)

    for l_tgt in range(mapper.num_layers):
        l_src = mapper.layer_map[l_tgt]
        k_src = src_cache.layers[l_src].keys.to(device)    # [1, 8, seq_len, 128]
        v_src = src_cache.layers[l_src].values.to(device)  # [1, 8, seq_len, 128]

        # 1. Strip RoPE
        k_unrot = strip_rope(k_src, cos_src, sin_src)

        k_mapped_unrot = torch.zeros_like(k_unrot)
        v_mapped = torch.zeros_like(v_src)

        for h in range(mapper.num_kv_heads):
            W_k = mapper.weights_k[(l_tgt, h)].to(device=device, dtype=torch.float32)
            W_v = mapper.weights_v[(l_tgt, h)].to(device=device, dtype=torch.float32)

            k_h = k_unrot[:, h, :, :].float()  # [1, seq_len, 128]
            v_h = v_src[:, h, :, :].float()

            # Apply perturbation to carrier
            if norm > 0.0 and attack_type in ["svd", "random"]:
                if attack_type == "svd":
                    unit_dir = svd_dirs[(l_tgt, h)].unsqueeze(0).unsqueeze(0)  # [1, 1, 128]
                else:
                    rand_vec = get_random_direction(device, mapper.head_dim, seed, req_id, l_tgt, h)
                    unit_dir = rand_vec.unsqueeze(0).unsqueeze(0)  # [1, 1, 128]

                delta = norm * unit_dir

                if carrier in ["K", "both"]:
                    k_h = k_h + delta
                if carrier in ["V", "both"]:
                    v_h = v_h + delta

            # Linear projection
            k_mapped_unrot[:, h, :, :] = torch.matmul(k_h, W_k).to(k_unrot.dtype)
            v_mapped[:, h, :, :] = torch.matmul(v_h, W_v).to(v_src.dtype)

        # 3. Apply target RoPE
        k_mapped = apply_rope(k_mapped_unrot, cos_tgt, sin_tgt)

        # 4. Target cache update
        tgt_cache.update(
            k_mapped.to(target_dtype),
            v_mapped.to(target_dtype),
            layer_idx=l_tgt
        )

    return tgt_cache

def main():
    parser = argparse.ArgumentParser(description="Phase 7: Closed-Form Adversarial Perturbation vs Random Noise Sweep")
    parser.add_argument("--max_samples", type=int, default=30, help="Number of GSM8K test examples")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--mapper_path", type=str, default=os.path.join(ROOT_DIR, "logs", "cross_model_mapper.pt"))
    parser.add_argument("--results_csv", type=str, default=os.path.join(ROOT_DIR, "results", "phase7_attack_results.csv"))
    parser.add_argument("--max_new_tokens", type=int, default=384, help="Max new tokens generated")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random control draws")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=================================================================")
    print("   PHASE 7: CLOSED-FORM ADVERSARIAL PERTURBATION (SVD VS RANDOM)  ")
    print("=================================================================")
    print(f"Device        : {device}")
    print(f"Eval Samples  : {args.max_samples} GSM8K problems")
    print(f"Mapper Path   : {args.mapper_path}")

    # Load mapper
    mapper = CrossModelKVMapper.load(args.mapper_path)
    svd_dirs = precompute_svd_directions(mapper, device)
    print(f"[Attack Setup] Precomputed {len(svd_dirs)} dominant singular directions.")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    print("\n[Models] Loading source (0.6B) and target (1.7B) models on MPS...")
    m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
    m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
    m06.eval()
    m17.eval()
    rotary_06 = m06.model.rotary_emb
    rotary_17 = m17.model.rotary_emb
    print("[Models] Both models ready.")

    # Load GSM8K test data
    print(f"\n[Data] Loading {args.max_samples} GSM8K test examples...")
    dataset_iter = load_gsm8k(split="test")
    samples = []
    for idx, item in enumerate(dataset_iter):
        if idx >= args.max_samples:
            break
        item["request_id"] = idx
        samples.append(item)
    print(f"[Data] Loaded {len(samples)} samples.")

    # Precompute source prefill for all 30 samples to avoid redundant forward passes
    print(f"\n[Optimization] Precomputing source (0.6B) KV caches for {len(samples)} samples...")
    cached_src_kvs: Dict[int, DynamicCache] = {}
    cached_encodings: Dict[int, Any] = {}

    with torch.no_grad():
        for item in tqdm(samples, desc="Precomputing source KVs"):
            req_id = item["request_id"]
            question = item["question"]
            prompt_text = build_gsm8k_prompt(tokenizer, question)
            enc = tokenizer(prompt_text, return_tensors="pt").to(device)
            out_src = m06(**enc, use_cache=True)
            cached_src_kvs[req_id] = out_src.past_key_values
            cached_encodings[req_id] = enc
    print("[Optimization] Source KV precomputation complete.")

    # Define Attack Configurations to Compare:
    # 1. Clean Mapped Baseline
    # 2. SVD-direction attack at norms: 0.02, 0.08, 0.20
    # 3. Random-direction control at matched norms: 0.02, 0.08, 0.20
    norms_to_test = [0.02, 0.08, 0.20]
    configs = [{"name": "clean", "type": "clean", "norm": 0.0}]

    for norm in norms_to_test:
        configs.append({"name": f"svd_norm_{norm}", "type": "svd", "norm": norm})
        configs.append({"name": f"rand_norm_{norm}", "type": "random", "norm": norm})

    print(f"\n[Sweep Configuration] {len(configs)} conditions across {len(samples)} samples:")
    for c in configs:
        print(f"  Condition: {c['name']:<16} | Type={c['type']:<7} | Norm={c['norm']}")

    all_results = []
    summary_records = []

    for cfg in configs:
        c_name = cfg["name"]
        c_type = cfg["type"]
        norm = cfg["norm"]
        print(f"\n--- Running Condition: {c_name} (Type={c_type}, Norm={norm}) ---")

        n_correct = 0
        n_healthy = 0
        t0 = time.time()

        for item in tqdm(samples, desc=f"Evaluating {c_name}"):
            req_id = item["request_id"]
            gold = str(item["gold"]).strip()
            enc = cached_encodings[req_id]
            prompt_len = enc["input_ids"].shape[-1]
            src_cache = cached_src_kvs[req_id]

            with torch.no_grad():
                mapped_cache = map_cache_with_attack(
                    src_cache=src_cache,
                    mapper=mapper,
                    svd_dirs=svd_dirs,
                    src_rotary_emb=rotary_06,
                    tgt_rotary_emb=rotary_17,
                    device=device,
                    attack_type=c_type,
                    norm=norm,
                    seed=args.seed,
                    req_id=req_id,
                    carrier="K",
                    target_dtype=torch.bfloat16,
                )

                out = m17.generate(
                    **enc,
                    past_key_values=mapped_cache,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            raw_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
            pred = normalize_answer(extract_gsm8k_answer(raw_text))
            correct = (pred == gold) if (pred and gold) else False
            healthy, reason = check_output_health(raw_text, pred)

            if correct:
                n_correct += 1
            if healthy:
                n_healthy += 1

            all_results.append({
                "condition": c_name,
                "attack_type": c_type,
                "norm": norm,
                "request_id": req_id,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "output_healthy": healthy,
                "reason": reason,
                "raw_text_len": len(raw_text),
                "snippet": raw_text[:90].replace("\n", " "),
            })

        acc = (n_correct / len(samples)) * 100
        health_pct = (n_healthy / len(samples)) * 100
        elapsed = time.time() - t0
        summary_records.append({
            "condition": c_name,
            "type": c_type,
            "norm": norm,
            "correct": n_correct,
            "total": len(samples),
            "accuracy": round(acc, 2),
            "healthy": n_healthy,
            "health_pct": round(health_pct, 2),
            "elapsed_sec": round(elapsed, 1),
        })
        print(f"Result for {c_name:<16}: Accuracy = {acc:5.1f}% ({n_correct}/{len(samples)}) | Health = {health_pct:5.1f}% ({n_healthy}/{len(samples)})")

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("     PHASE 7 RESULTS: CLOSED-FORM SVD ATTACK VS MATCHED RANDOM NOISE       ")
    print("=" * 80)
    print(f"{'Condition':<18} | {'Type':<8} | {'Perturb Norm':<12} | {'Accuracy':<10} | {'Health %':<10} | {'Damage vs Clean':<15}")
    print("-" * 80)

    clean_acc = summary_records[0]["accuracy"]
    for rec in summary_records:
        drop = clean_acc - rec["accuracy"]
        drop_str = f"-{drop:4.1f}% pts" if drop > 0 else "Baseline"
        print(f"{rec['condition']:<18} | {rec['type']:<8} | {rec['norm']:<12.2f} | {rec['accuracy']:>6.1f}%    | {rec['health_pct']:>6.1f}%    | {drop_str:<15}")
    print("=" * 80)

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n[Saved] Detailed per-sample attack results saved to {args.results_csv}")

    summary_csv = args.results_csv.replace(".csv", "_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"[Saved] Summary table saved to {summary_csv}")

if __name__ == "__main__":
    main()

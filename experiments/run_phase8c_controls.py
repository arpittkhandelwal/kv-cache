#!/usr/bin/env python3
"""
Phase 8c: Controlled Experiment on SVD vs. Random Perturbations.
Isolates whether the SVD attack's damage is due to its larger target-side norm
(||delta @ W||) or genuine directional structure.

Conditions evaluated at eps in {0.08, 0.20} on 30 GSM8K examples:
- Clean baseline (reference)
- SVD attack (deterministic dominant singular vector)
- rand_srcnorm: isotropic random noise matched to source-side norm (seeds 0, 1, 2)
- rand_tgtnorm: random noise rescaled so target-side norm equals SVD target-side norm (seeds 0, 1, 2)
"""

import os
import sys
import shutil
import time
import math
import csv
import re
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from data import load_gsm8k
from utils import normalize_answer, extract_gsm8k_answer, set_seed
from cross_model_mapper import strip_rope, apply_rope
from experiments.run_phase8_complete import Top2RidgeMapper, check_output_health, build_gsm8k_prompt

def get_free_disk_gb(path: str = "/") -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024 ** 3)

def map_cache_with_control(
    mapper: Top2RidgeMapper,
    src_cache: DynamicCache,
    m06,
    m17,
    device: torch.device,
    attack_type: str, # "clean", "svd", "rand_srcnorm", "rand_tgtnorm"
    norm: float,
    seed: int = 0,
    req_id: int = 0,
    target_dtype: torch.dtype = torch.bfloat16,
) -> DynamicCache:
    tgt_cache = DynamicCache()
    seq_len = src_cache.get_seq_length()
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_k = src_cache.layers[0].keys
    cos_s, sin_s = m06.model.rotary_emb(dummy_k, pos_ids)
    cos_t, sin_t = m17.model.rotary_emb(dummy_k, pos_ids)

    for l_tgt in range(mapper.num_layers):
        src_idx = mapper.layer_map[l_tgt]
        k0 = src_cache.layers[src_idx[0]].keys.to(device)
        k1 = src_cache.layers[src_idx[1]].keys.to(device)
        v0 = src_cache.layers[src_idx[0]].values.to(device)
        v1 = src_cache.layers[src_idx[1]].values.to(device)

        k0_u = strip_rope(k0, cos_s, sin_s)
        k1_u = strip_rope(k1, cos_s, sin_s)

        k_mapped_unrot = torch.zeros(1, mapper.num_heads, seq_len, mapper.out_dim, device=device, dtype=torch.float32)
        v_mapped = torch.zeros(1, mapper.num_heads, seq_len, mapper.out_dim, device=device, dtype=torch.float32)

        for h in range(mapper.num_heads):
            W_k = mapper.weights_k[(l_tgt, h)].to(device=device, dtype=torch.float32)
            W_v = mapper.weights_v[(l_tgt, h)].to(device=device, dtype=torch.float32)

            x_k = torch.cat([k0_u[:, h, :, :].float(), k1_u[:, h, :, :].float()], dim=-1) # [1, seq_len, 256]
            x_v = torch.cat([v0[:, h, :, :].float(), v1[:, h, :, :].float()], dim=-1)

            if attack_type != "clean" and norm > 0.0:
                if attack_type == "svd":
                    # Dominant singular vector (left singular vector of W_k)
                    u1 = mapper.svd_dirs[(l_tgt, h)].to(device=device, dtype=torch.float32)
                    delta = norm * u1.view(1, 1, mapper.in_dim)

                elif attack_type == "rand_srcnorm":
                    # Isotropic random noise with source norm = norm
                    gen = torch.Generator(device="cpu").manual_seed(seed * 100000 + req_id * 1000 + l_tgt * 8 + h)
                    r = torch.randn(mapper.in_dim, generator=gen, dtype=torch.float32).to(device)
                    r_unit = r / (torch.norm(r) + 1e-12)
                    delta = norm * r_unit.view(1, 1, mapper.in_dim)

                elif attack_type == "rand_tgtnorm":
                    # Random noise rescaled so target-side norm ||delta @ W_k|| = norm * sigma_max(W_k)
                    gen = torch.Generator(device="cpu").manual_seed(seed * 100000 + req_id * 1000 + l_tgt * 8 + h)
                    r = torch.randn(mapper.in_dim, generator=gen, dtype=torch.float32).to(device)
                    r_unit = r / (torch.norm(r) + 1e-12)

                    # Compute target shift of unit vector: r_unit @ W_k
                    r_tgt_shift = r_unit.view(1, mapper.in_dim) @ W_k # [1, 128]
                    r_tgt_norm = torch.norm(r_tgt_shift).item()

                    # SVD target norm for this head is norm * sigma_max
                    # Compute sigma_max from SVD of W_k
                    U, S, Vh = torch.linalg.svd(W_k, full_matrices=False)
                    sigma_max = S[0].item()
                    target_goal_norm = norm * sigma_max

                    # Scale factor alpha so that ||(alpha * r_unit) @ W_k|| = target_goal_norm
                    scale = target_goal_norm / (r_tgt_norm + 1e-12)
                    delta = (scale * r_unit).view(1, 1, mapper.in_dim)
                else:
                    delta = 0.0

                x_k = x_k + delta

            k_mapped_unrot[:, h, :, :] = torch.matmul(x_k, W_k)
            v_mapped[:, h, :, :] = torch.matmul(x_v, W_v)

        k_mapped = apply_rope(k_mapped_unrot, cos_t, sin_t)
        tgt_cache.update(k_mapped.to(target_dtype), v_mapped.to(target_dtype), layer_idx=l_tgt)

    return tgt_cache

def main():
    device = torch.device("mps")
    set_seed(42)

    print("=================================================================")
    print("   PHASE 8c: SVD VS. TARGET-NORM MATCHED RANDOM NOISE CONTROLS  ")
    print("=================================================================")

    # Disk Space Check
    free_gb = get_free_disk_gb(ROOT_DIR)
    print(f"[Disk Guardrail] Free disk space: {free_gb:.2f} GB")
    if free_gb < 3.0:
        raise RuntimeError(f"ABORT: Free disk space ({free_gb:.2f} GB) is below 3.0 GB safety limit!")

    # Load Models
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print("Loading models on MPS...")
    m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
    m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
    m06.eval()
    m17.eval()

    # Load Top-2 Mapper
    mapper_path = os.path.join(ROOT_DIR, "logs", "top2_ridge_mapper.pt")
    if not os.path.exists(mapper_path):
        raise FileNotFoundError(f"Top-2 mapper checkpoint not found at {mapper_path}!")
    print(f"Loading trained top-2 mapper from {mapper_path}...")
    ckpt = torch.load(mapper_path, map_location="cpu", weights_only=False)
    mapper = Top2RidgeMapper(lambda_reg=0.01)
    mapper.weights_k = {eval(k): v for k, v in ckpt["weights_k"].items()}
    mapper.weights_v = {eval(k): v for k, v in ckpt["weights_v"].items()}
    mapper.svd_dirs = {eval(k): v for k, v in ckpt["svd_dirs"].items()}

    # Load 30 GSM8K test samples
    print("\n[Data] Loading 30 GSM8K test examples...")
    dataset_iter = load_gsm8k(split="test")
    samples = []
    for idx, item in enumerate(dataset_iter):
        if idx >= 30:
            break
        item["request_id"] = idx
        samples.append(item)
    print(f"[Data] Loaded {len(samples)} samples.")

    # Precompute source model (0.6B) prefill KV caches for all 30 samples ONCE
    print("\n[Precompute] Caching source model prefill KVs for all 30 samples...")
    cached_src_kvs = {}
    cached_encodings = {}
    for item in tqdm(samples, desc="Precomputing source KVs"):
        req_id = item["request_id"]
        prompt = build_gsm8k_prompt(tokenizer, item["question"])
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_src = m06(**enc, use_cache=True)
            cached_src_kvs[req_id] = out_src.past_key_values
            cached_encodings[req_id] = enc
    print("[Precompute] Source KVs cached.")

    # Step 1: Run Clean Baseline to obtain paired baseline predictions
    print("\n--- Running Condition: Clean Baseline ---")
    clean_preds = {}
    clean_corrects = {}
    clean_correct_count = 0

    for item in tqdm(samples, desc="Evaluating Clean Baseline"):
        req_id = item["request_id"]
        gold = str(item["gold"]).strip()
        enc = cached_encodings[req_id]
        plen = enc["input_ids"].shape[-1]
        src_cache = cached_src_kvs[req_id]

        with torch.no_grad():
            mapped_cache = mapper.map_cache(
                src_cache=src_cache,
                m06=m06,
                m17=m17,
                device=device,
                attack_type="clean",
                norm=0.0,
            )
            out = m17.generate(
                **enc,
                past_key_values=mapped_cache,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        txt = tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()
        pred = normalize_answer(extract_gsm8k_answer(txt))
        corr = (pred == gold) if (pred and gold) else False
        clean_preds[req_id] = pred
        clean_corrects[req_id] = corr
        if corr:
            clean_correct_count += 1
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    clean_acc = (clean_correct_count / len(samples)) * 100
    print(f"-> Clean Baseline Accuracy: {clean_acc:5.1f}% ({clean_correct_count}/{len(samples)})\n")

    # Conditions to run:
    # eps in [0.08, 0.20]:
    # - svd (deterministic, run once)
    # - rand_srcnorm (seeds 0, 1, 2)
    # - rand_tgtnorm (seeds 0, 1, 2)
    eps_list = [0.08, 0.20]
    seeds = [0, 1, 2]

    run_configs = []
    for eps in eps_list:
        run_configs.append({"name": f"svd_eps_{eps}", "type": "svd", "norm": eps, "seed": 42})
        for s in seeds:
            run_configs.append({"name": f"rand_srcnorm_eps_{eps}_s{s}", "type": "rand_srcnorm", "norm": eps, "seed": s})
            run_configs.append({"name": f"rand_tgtnorm_eps_{eps}_s{s}", "type": "rand_tgtnorm", "norm": eps, "seed": s})

    all_eval_records = []
    detailed_per_sample = []

    for cfg in run_configs:
        c_name = cfg["name"]
        c_type = cfg["type"]
        norm = cfg["norm"]
        seed = cfg["seed"]

        print(f"\n--- Evaluating Condition: {c_name} (Norm={norm}, Seed={seed}) ---")
        n_correct = 0
        n_healthy = 0
        flips_c2w = 0 # correct -> wrong
        flips_w2c = 0 # wrong -> correct
        t0 = time.time()

        for item in tqdm(samples, desc=f"Eval {c_name}"):
            req_id = item["request_id"]
            gold = str(item["gold"]).strip()
            enc = cached_encodings[req_id]
            plen = enc["input_ids"].shape[-1]
            src_cache = cached_src_kvs[req_id]

            with torch.no_grad():
                mapped_cache = map_cache_with_control(
                    mapper=mapper,
                    src_cache=src_cache,
                    m06=m06,
                    m17=m17,
                    device=device,
                    attack_type=c_type,
                    norm=norm,
                    seed=seed,
                    req_id=req_id,
                )
                out = m17.generate(
                    **enc,
                    past_key_values=mapped_cache,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            txt = tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()
            pred = normalize_answer(extract_gsm8k_answer(txt))
            corr = (pred == gold) if (pred and gold) else False
            healthy, reason = check_output_health(txt, pred)

            if corr:
                n_correct += 1
            if healthy:
                n_healthy += 1

            # Paired flip analysis against clean baseline
            c_corr = clean_corrects[req_id]
            if c_corr and not corr:
                flips_c2w += 1
            elif not c_corr and corr:
                flips_w2c += 1

            detailed_per_sample.append({
                "condition": c_name,
                "type": c_type,
                "norm": norm,
                "seed": seed,
                "request_id": req_id,
                "gold": gold,
                "clean_pred": clean_preds[req_id],
                "clean_correct": c_corr,
                "pred": pred,
                "correct": corr,
                "healthy": healthy,
                "flip_c2w": (c_corr and not corr),
                "flip_w2c": (not c_corr and corr),
            })

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        acc = (n_correct / len(samples)) * 100
        health_pct = (n_healthy / len(samples)) * 100
        elapsed = time.time() - t0

        rec = {
            "condition": c_name,
            "type": c_type,
            "norm": norm,
            "seed": seed,
            "correct": n_correct,
            "total": len(samples),
            "accuracy": round(acc, 2),
            "health_pct": round(health_pct, 2),
            "flips_c2w": flips_c2w,
            "flips_w2c": flips_w2c,
            "elapsed_sec": round(elapsed, 1),
        }
        all_eval_records.append(rec)
        print(f"-> {c_name:<28}: Acc = {acc:5.1f}% ({n_correct}/{len(samples)}) | Flips C->W = {flips_c2w} | Health = {health_pct:5.1f}%")

    # Aggregate Statistics Across Seeds
    print("\n" + "=" * 115)
    print("                 PHASE 8c: SVD VS. MATCHED CONTROLS EXIT TABLE                         ")
    print("=" * 115)
    print(f"{'Eps':<6} | {'Clean':<8} | {'SVD':<10} | {'rand_srcnorm (mean±std)':<25} | {'rand_tgtnorm (mean±std)':<25} | {'Flips C->W (SVD / src / tgt)':<30}")
    print("-" * 115)

    summary_exit_table = []

    for eps in eps_list:
        # SVD
        svd_rec = [r for r in all_eval_records if r["type"] == "svd" and r["norm"] == eps][0]
        svd_acc = svd_rec["accuracy"]
        svd_flips = svd_rec["flips_c2w"]

        # rand_srcnorm (3 seeds)
        src_recs = [r for r in all_eval_records if r["type"] == "rand_srcnorm" and r["norm"] == eps]
        src_accs = [r["accuracy"] for r in src_recs]
        src_flips = [r["flips_c2w"] for r in src_recs]
        src_acc_mean = np.mean(src_accs)
        src_acc_std = np.std(src_accs)
        src_flips_mean = np.mean(src_flips)

        # rand_tgtnorm (3 seeds)
        tgt_recs = [r for r in all_eval_records if r["type"] == "rand_tgtnorm" and r["norm"] == eps]
        tgt_accs = [r["accuracy"] for r in tgt_recs]
        tgt_flips = [r["flips_c2w"] for r in tgt_recs]
        tgt_acc_mean = np.mean(tgt_accs)
        tgt_acc_std = np.std(tgt_accs)
        tgt_flips_mean = np.mean(tgt_flips)

        src_str = f"{src_acc_mean:5.1f}% ± {src_acc_std:4.1f}%"
        tgt_str = f"{tgt_acc_mean:5.1f}% ± {tgt_acc_std:4.1f}%"
        flips_str = f"{svd_flips} / {src_flips_mean:4.1f} / {tgt_flips_mean:4.1f}"

        print(f"{eps:<6.2f} | {clean_acc:5.1f}%  | {svd_acc:5.1f}%   | {src_str:<25} | {tgt_str:<25} | {flips_str:<30}")

        summary_exit_table.append({
            "eps": eps,
            "clean_accuracy": clean_acc,
            "svd_accuracy": svd_acc,
            "svd_flips_c2w": svd_flips,
            "rand_srcnorm_mean": round(src_acc_mean, 2),
            "rand_srcnorm_std": round(src_acc_std, 2),
            "rand_srcnorm_flips_mean": round(src_flips_mean, 2),
            "rand_tgtnorm_mean": round(tgt_acc_mean, 2),
            "rand_tgtnorm_std": round(tgt_acc_std, 2),
            "rand_tgtnorm_flips_mean": round(tgt_flips_mean, 2),
        })

    print("=" * 115)

    # Save CSVs with phase8c_ prefix
    detail_csv = os.path.join(ROOT_DIR, "results", "phase8c_per_sample_details.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detailed_per_sample[0].keys()))
        writer.writeheader()
        writer.writerows(detailed_per_sample)
    print(f"\n[Saved] Detailed per-sample records saved to {detail_csv}")

    summary_csv = os.path.join(ROOT_DIR, "results", "phase8c_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_exit_table[0].keys()))
        writer.writeheader()
        writer.writerows(summary_exit_table)
    print(f"[Saved] Phase 8c exit summary saved to {summary_csv}")

if __name__ == "__main__":
    main()

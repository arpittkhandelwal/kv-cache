#!/usr/bin/env python3
"""
Phase 8c Extension: 8-Seed Evaluation at eps=0.08
Runs 5 additional seeds (seeds 3, 4, 5, 6, 7) for rand_srcnorm and rand_tgtnorm
at eps=0.08, combining with seeds 0, 1, 2 (8 total seeds) to rigorously test
whether the SVD-vs-tgtnorm gap holds up with statistical confidence.
"""

import os
import sys
import shutil
import time
import csv
import torch
import numpy as np
import pandas as pd
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
from experiments.run_phase8_complete import Top2RidgeMapper, check_output_health, build_gsm8k_prompt
from experiments.run_phase8c_controls import map_cache_with_control, get_free_disk_gb

def main():
    device = torch.device("mps")
    set_seed(42)

    print("=================================================================")
    print("   PHASE 8c EXTENSION: 8-SEED ROBUSTNESS EVALUATION (eps=0.08)   ")
    print("=================================================================")

    free_gb = get_free_disk_gb(ROOT_DIR)
    print(f"[Disk Guardrail] Free disk space: {free_gb:.2f} GB")
    if free_gb < 3.0:
        raise RuntimeError(f"ABORT: Free disk space ({free_gb:.2f} GB) is below 3.0 GB safety limit!")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print("Loading models on MPS...")
    m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
    m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
    m06.eval()
    m17.eval()

    mapper_path = os.path.join(ROOT_DIR, "logs", "top2_ridge_mapper.pt")
    ckpt = torch.load(mapper_path, map_location="cpu", weights_only=False)
    mapper = Top2RidgeMapper(lambda_reg=0.01)
    mapper.weights_k = {eval(k): v for k, v in ckpt["weights_k"].items()}
    mapper.weights_v = {eval(k): v for k, v in ckpt["weights_v"].items()}
    mapper.svd_dirs = {eval(k): v for k, v in ckpt["svd_dirs"].items()}

    # Load 30 GSM8K test samples
    dataset_iter = load_gsm8k(split="test")
    samples = []
    for idx, item in enumerate(dataset_iter):
        if idx >= 30:
            break
        item["request_id"] = idx
        samples.append(item)

    # Read clean predictions from existing details CSV
    existing_details_path = os.path.join(ROOT_DIR, "results", "phase8c_per_sample_details.csv")
    df_existing = pd.read_csv(existing_details_path)
    # Map request_id -> clean_correct, clean_pred
    clean_info = {}
    for _, row in df_existing[df_existing["condition"] == "clean"].iterrows():
        clean_info[int(row["request_id"])] = {
            "clean_pred": str(row["clean_pred"]),
            "clean_correct": bool(row["clean_correct"]),
        }
    if not clean_info:
        # Fallback to any row
        for _, row in df_existing.drop_duplicates(subset=["request_id"]).iterrows():
            clean_info[int(row["request_id"])] = {
                "clean_pred": str(row["clean_pred"]),
                "clean_correct": bool(row["clean_correct"]),
            }

    # Precompute source model prefill KVs
    print("[Precompute] Caching source model prefill KVs for all 30 samples...")
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

    # Run 5 new seeds: 3, 4, 5, 6, 7 at eps=0.08
    new_seeds = [3, 4, 5, 6, 7]
    eps = 0.08
    new_configs = []
    for s in new_seeds:
        new_configs.append({"name": f"rand_srcnorm_eps_{eps}_s{s}", "type": "rand_srcnorm", "norm": eps, "seed": s})
        new_configs.append({"name": f"rand_tgtnorm_eps_{eps}_s{s}", "type": "rand_tgtnorm", "norm": eps, "seed": s})

    new_eval_records = []
    new_detailed = []

    for cfg in new_configs:
        c_name = cfg["name"]
        c_type = cfg["type"]
        seed = cfg["seed"]

        print(f"\n--- Evaluating Condition: {c_name} (Norm={eps}, Seed={seed}) ---")
        n_correct = 0
        n_healthy = 0
        flips_c2w = 0
        flips_w2c = 0
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
                    norm=eps,
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

            c_corr = clean_info[req_id]["clean_correct"]
            c_pred = clean_info[req_id]["clean_pred"]
            if c_corr and not corr:
                flips_c2w += 1
            elif not c_corr and corr:
                flips_w2c += 1

            new_detailed.append({
                "condition": c_name,
                "type": c_type,
                "norm": eps,
                "seed": seed,
                "request_id": req_id,
                "gold": gold,
                "clean_pred": c_pred,
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
        print(f"-> {c_name:<28}: Acc = {acc:5.1f}% ({n_correct}/{len(samples)}) | Flips C->W = {flips_c2w} | Health = {health_pct:5.1f}%")

    # Combine existing details with new details
    df_new = pd.DataFrame(new_detailed)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(existing_details_path, index=False)
    print(f"\n[Updated] Saved combined per-sample details to {existing_details_path}")

    # Now aggregate all 8 seeds (0..7) for rand_srcnorm and rand_tgtnorm at eps=0.08
    all_seeds = list(range(8))
    src_8_accs = []
    src_8_flips = []
    tgt_8_accs = []
    tgt_8_flips = []

    for s in all_seeds:
        # srcnorm
        c_src = f"rand_srcnorm_eps_0.08_s{s}"
        sub_src = df_combined[df_combined["condition"] == c_src]
        src_8_accs.append(sub_src["correct"].mean() * 100)
        src_8_flips.append(sub_src["flip_c2w"].sum())

        # tgtnorm
        c_tgt = f"rand_tgtnorm_eps_0.08_s{s}"
        sub_tgt = df_combined[df_combined["condition"] == c_tgt]
        tgt_8_accs.append(sub_tgt["correct"].mean() * 100)
        tgt_8_flips.append(sub_tgt["flip_c2w"].sum())

    svd_acc = df_combined[df_combined["condition"] == "svd_eps_0.08"]["correct"].mean() * 100
    svd_flips = df_combined[df_combined["condition"] == "svd_eps_0.08"]["flip_c2w"].sum()
    clean_acc = df_combined[df_combined["condition"] == "clean"]["correct"].mean() * 100

    src_mean, src_std = np.mean(src_8_accs), np.std(src_8_accs, ddof=1)
    tgt_mean, tgt_std = np.mean(tgt_8_accs), np.std(tgt_8_accs, ddof=1)
    src_flips_mean, src_flips_std = np.mean(src_8_flips), np.std(src_8_flips, ddof=1)
    tgt_flips_mean, tgt_flips_std = np.mean(tgt_8_flips), np.std(tgt_8_flips, ddof=1)

    # Statistical test: One-sample t-statistic comparing SVD accuracy (13.33) against 8-seed tgtnorm distribution
    se = tgt_std / math.sqrt(len(tgt_8_accs)) if tgt_std > 0 else 1e-6
    t_stat = (tgt_mean - svd_acc) / se
    gap = tgt_mean - svd_acc

    print("\n" + "=" * 115)
    print("             PHASE 8c: EXTENDED 8-SEED ROBUSTNESS EXIT TABLE (eps=0.08)                ")
    print("=" * 115)
    print(f"{'Condition':<22} | {'Seeds':<7} | {'Accuracy (Mean±Std)':<24} | {'Flips C->W (Mean±Std)':<24} | {'Gap vs SVD (pts)':<18}")
    print("-" * 115)
    print(f"{'Clean Baseline':<22} | {'—':<7} | {clean_acc:>5.1f}%                   | {'0 (Baseline)':<24} | {'—':<18}")
    print(f"{'SVD Attack':<22} | {'Det':<7} | {svd_acc:>5.1f}%                   | {int(svd_flips):<24} | {'Baseline (0.0)':<18}")
    print(f"{'rand_srcnorm':<22} | {'8 seeds':<7} | {src_mean:>5.1f}% ± {src_std:4.1f}%            | {src_flips_mean:>4.1f} ± {src_flips_std:3.1f}              | {src_mean - svd_acc:>+5.1f} pts")
    print(f"{'rand_tgtnorm (Fair)':<22} | {'8 seeds':<7} | {tgt_mean:>5.1f}% ± {tgt_std:4.1f}%            | {tgt_flips_mean:>4.1f} ± {tgt_flips_std:3.1f}              | {gap:>+5.1f} pts")
    print("=" * 115)
    print(f"\n[Statistical Evaluation]")
    print(f"  • SVD Accuracy                 : {svd_acc:.1f}% (4/30)")
    print(f"  • rand_tgtnorm 8-seed Accuracy : {tgt_mean:.1f}% ± {tgt_std:.1f}% (range: {min(tgt_8_accs):.1f}% - {max(tgt_8_accs):.1f}%)")
    print(f"  • Observed Gap                 : {gap:.1f} percentage points")
    print(f"  • One-sample t-statistic vs SVD: t = {t_stat:.3f} (degrees of freedom = 7, t > 2.365 indicates p < 0.05)")

    # Per-seed breakdown table
    print("\n[Per-Seed Breakdown for eps=0.08]")
    print(f"{'Seed':<6} | {'rand_srcnorm Acc':<18} | {'rand_srcnorm Flips':<20} | {'rand_tgtnorm Acc':<18} | {'rand_tgtnorm Flips':<20}")
    print("-" * 88)
    for s in all_seeds:
        print(f"Seed {s:<1} | {src_8_accs[s]:>5.1f}%             | {src_8_flips[s]:<20} | {tgt_8_accs[s]:>5.1f}%             | {tgt_8_flips[s]:<20}")
    print("-" * 88)

    # Save summary CSV
    summary_8seeds_path = os.path.join(ROOT_DIR, "results", "phase8c_8seeds_summary.csv")
    with open(summary_8seeds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "clean", "svd", "rand_srcnorm_8seeds_mean", "rand_srcnorm_8seeds_std", "rand_tgtnorm_8seeds_mean", "rand_tgtnorm_8seeds_std", "gap_pts", "t_statistic"])
        writer.writerow(["accuracy", clean_acc, svd_acc, round(src_mean, 2), round(src_std, 2), round(tgt_mean, 2), round(tgt_std, 2), round(gap, 2), round(t_stat, 3)])
        writer.writerow(["flips_c2w", 0, svd_flips, round(src_flips_mean, 2), round(src_flips_std, 2), round(tgt_flips_mean, 2), round(tgt_flips_std, 2), round(svd_flips - tgt_flips_mean, 2), "N/A"])
    print(f"[Saved] 8-seed summary saved to {summary_8seeds_path}")

if __name__ == "__main__":
    main()

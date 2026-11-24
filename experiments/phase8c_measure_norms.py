#!/usr/bin/env python3
"""
Phase 8c - Step 1: Measure Target-Side Norms ||delta @ W||
Computes the exact target-side perturbation norms produced by SVD dominant-vector
perturbation vs. isotropic random noise across all layers and heads for eps in {0.02, 0.08, 0.20}.
"""

import os
import sys
import csv
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from experiments.run_phase8_complete import Top2RidgeMapper
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    device = torch.device("mps")
    print("=================================================================")
    print("   PHASE 8c - STEP 1: TARGET-SIDE PERTURBATION NORM ANALYSIS     ")
    print("=================================================================")

    mapper_path = os.path.join(ROOT_DIR, "logs", "top2_ridge_mapper.pt")
    mapper = Top2RidgeMapper(lambda_reg=0.01)

    if os.path.exists(mapper_path):
        print(f"Loading existing top-2 mapper from {mapper_path}...")
        ckpt = torch.load(mapper_path, map_location="cpu", weights_only=False)
        mapper.weights_k = {eval(k): v for k, v in ckpt["weights_k"].items()}
        mapper.weights_v = {eval(k): v for k, v in ckpt["weights_v"].items()}
        mapper.svd_dirs = {eval(k): v for k, v in ckpt["svd_dirs"].items()}
    else:
        print("Fitting Top-2 Ridge Mapper online (80 WikiText-2 sequences)...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
        m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
        m06.eval()
        m17.eval()
        mapper.fit_online(m06, m17, tokenizer, device, num_sequences=80)
        # Save mapper checkpoint
        save_dict = {
            "weights_k": {str(k): v.cpu() for k, v in mapper.weights_k.items()},
            "weights_v": {str(k): v.cpu() for k, v in mapper.weights_v.items()},
            "svd_dirs": {str(k): v.cpu() for k, v in mapper.svd_dirs.items()},
        }
        torch.save(save_dict, mapper_path)
        print(f"Saved mapper checkpoint to {mapper_path}")

    eps_values = [0.02, 0.08, 0.20]
    num_layers = mapper.num_layers
    num_heads = mapper.num_heads
    in_dim = mapper.in_dim # 256

    print("\nComputing target-side norms across all 28 layers (averaged over 8 heads)...")
    layer_records = []
    
    # Store global sums for total average
    svd_norms_by_eps = {eps: [] for eps in eps_values}
    rand_norms_by_eps = {eps: [] for eps in eps_values}

    torch.manual_seed(42)
    N_DRAWS = 100 # Empirical draws per head to measure exact random target norm

    for l in range(num_layers):
        row = {"layer": l}
        for eps in eps_values:
            svd_h_norms = []
            rand_h_norms = []

            for h in range(num_heads):
                W_k = mapper.weights_k[(l, h)].float() # [256, 128]
                U, S, Vh = torch.linalg.svd(W_k, full_matrices=False)
                sigma_max = S[0].item()

                # SVD target norm = eps * sigma_max
                svd_tgt_norm = eps * sigma_max
                svd_h_norms.append(svd_tgt_norm)
                svd_norms_by_eps[eps].append(svd_tgt_norm)

                # Random source norm = eps
                # Draw 100 random unit vectors in source space
                rand_vecs = torch.randn(N_DRAWS, in_dim)
                rand_unit = rand_vecs / torch.norm(rand_vecs, dim=-1, keepdim=True)
                delta_rand = eps * rand_unit # [100, 256]
                tgt_shifts = delta_rand @ W_k # [100, 128]
                mean_rand_tgt_norm = torch.norm(tgt_shifts, dim=-1).mean().item()
                rand_h_norms.append(mean_rand_tgt_norm)
                rand_norms_by_eps[eps].append(mean_rand_tgt_norm)

            row[f"svd_eps_{eps}"] = np.mean(svd_h_norms)
            row[f"rand_eps_{eps}"] = np.mean(rand_h_norms)
            row[f"ratio_eps_{eps}"] = np.mean(svd_h_norms) / (np.mean(rand_h_norms) + 1e-12)

        layer_records.append(row)

    # Print Summary Table
    print("\n" + "=" * 90)
    print("      STEP 1: TARGET-SIDE PERTURBATION NORM COMPARISON (||delta @ W||)       ")
    print("=" * 90)
    print(f"{'Layer':<6} | {'eps=0.02 (SVD vs Rand)':<25} | {'eps=0.08 (SVD vs Rand)':<25} | {'eps=0.20 (SVD vs Rand)':<25} | {'Amplification':<12}")
    print("-" * 90)
    for l in [0, 1, 2, 5, 10, 15, 20, 25, 27]:
        r = layer_records[l]
        s02, r02 = r["svd_eps_0.02"], r["rand_eps_0.02"]
        s08, r08 = r["svd_eps_0.08"], r["rand_eps_0.08"]
        s20, r20 = r["svd_eps_0.2"], r["rand_eps_0.2"]
        ratio = r["ratio_eps_0.08"]
        print(f"L{l:<4d} | SVD={s02:5.3f} vs Rand={r02:5.3f}  | SVD={s08:5.3f} vs Rand={r08:5.3f}  | SVD={s20:5.3f} vs Rand={r20:5.3f}  | {ratio:5.2f}x")
    print("-" * 90)

    # Overall Means
    mean_s02 = np.mean(svd_norms_by_eps[0.02])
    mean_r02 = np.mean(rand_norms_by_eps[0.02])
    mean_s08 = np.mean(svd_norms_by_eps[0.08])
    mean_r08 = np.mean(rand_norms_by_eps[0.08])
    mean_s20 = np.mean(svd_norms_by_eps[0.20])
    mean_r20 = np.mean(rand_norms_by_eps[0.20])
    mean_ratio = mean_s08 / mean_r08

    print(f"{'MEAN':<6} | SVD={mean_s02:5.3f} vs Rand={mean_r02:5.3f}  | SVD={mean_s08:5.3f} vs Rand={mean_r08:5.3f}  | SVD={mean_s20:5.3f} vs Rand={mean_r20:5.3f}  | {mean_ratio:5.2f}x")
    print("=" * 90)
    print(f"\n[Finding] SVD produces a {mean_ratio:.2f}x larger target-side perturbation than isotropic random noise at the same source eps.")
    print("To test if SVD is genuinely structured, rand_tgtnorm must rescale random noise by this factor.")

    # Save CSV
    out_csv = os.path.join(ROOT_DIR, "results", "phase8c_target_norm_measurements.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_records[0].keys()))
        writer.writeheader()
        writer.writerows(layer_records)
    print(f"[Saved] Target norm measurements saved to {out_csv}")

if __name__ == "__main__":
    main()

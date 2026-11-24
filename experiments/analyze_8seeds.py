#!/usr/bin/env python3
"""
Aggregate and analyze the 8-seed results from phase8c_per_sample_details.csv.
Computes mean, std, seed range, t-statistic, and paired flip transitions for:
- clean
- SVD (eps=0.08 and eps=0.20)
- rand_srcnorm (8 seeds at eps=0.08, 3 seeds at eps=0.20)
- rand_tgtnorm (8 seeds at eps=0.08, 3 seeds at eps=0.20)
"""

import os
import sys
import math
import csv
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
details_path = os.path.join(ROOT_DIR, "results", "phase8c_per_sample_details.csv")
df = pd.read_csv(details_path)

print("=" * 115)
print("             PHASE 8c EXTENSION: COMPLETE 8-SEED ROBUSTNESS ANALYSIS (eps=0.08)                ")
print("=" * 115)

# 1. Clean Baseline (30.0%)
clean_acc = 30.0 # 9/30

# 2. eps=0.08 analysis
svd_sub_08 = df[df["condition"] == "svd_eps_0.08"]
svd_acc_08 = svd_sub_08["correct"].mean() * 100
svd_flips_08 = svd_sub_08["flip_c2w"].sum()

src_08_accs = []
src_08_flips = []
tgt_08_accs = []
tgt_08_flips = []

for s in range(8):
    c_src = f"rand_srcnorm_eps_0.08_s{s}"
    sub_s = df[df["condition"] == c_src]
    src_08_accs.append(sub_s["correct"].mean() * 100)
    src_08_flips.append(sub_s["flip_c2w"].sum())

    c_tgt = f"rand_tgtnorm_eps_0.08_s{s}"
    sub_t = df[df["condition"] == c_tgt]
    tgt_08_accs.append(sub_t["correct"].mean() * 100)
    tgt_08_flips.append(sub_t["flip_c2w"].sum())

src_08_mean, src_08_std = np.mean(src_08_accs), np.std(src_08_accs, ddof=1)
tgt_08_mean, tgt_08_std = np.mean(tgt_08_accs), np.std(tgt_08_accs, ddof=1)
src_08_flips_mean, src_08_flips_std = np.mean(src_08_flips), np.std(src_08_flips, ddof=1)
tgt_08_flips_mean, tgt_08_flips_std = np.mean(tgt_08_flips), np.std(tgt_08_flips, ddof=1)

gap_08 = tgt_08_mean - svd_acc_08
se_08 = tgt_08_std / math.sqrt(len(tgt_08_accs))
t_stat_08 = gap_08 / se_08

# 3. eps=0.20 analysis (3 seeds)
svd_sub_20 = df[df["condition"] == "svd_eps_0.2"]
svd_acc_20 = svd_sub_20["correct"].mean() * 100
svd_flips_20 = svd_sub_20["flip_c2w"].sum()

src_20_accs = []
src_20_flips = []
tgt_20_accs = []
tgt_20_flips = []

for s in range(3):
    c_src = f"rand_srcnorm_eps_0.2_s{s}"
    sub_s = df[df["condition"] == c_src]
    src_20_accs.append(sub_s["correct"].mean() * 100)
    src_20_flips.append(sub_s["flip_c2w"].sum())

    c_tgt = f"rand_tgtnorm_eps_0.2_s{s}"
    sub_t = df[df["condition"] == c_tgt]
    tgt_20_accs.append(sub_t["correct"].mean() * 100)
    tgt_20_flips.append(sub_t["flip_c2w"].sum())

src_20_mean, src_20_std = np.mean(src_20_accs), np.std(src_20_accs, ddof=1)
tgt_20_mean, tgt_20_std = np.mean(tgt_20_accs), np.std(tgt_20_accs, ddof=1)
src_20_flips_mean = np.mean(src_20_flips)
tgt_20_flips_mean = np.mean(tgt_20_flips)

# Print Summary Exit Table
print(f"{'Eps':<6} | {'Clean':<8} | {'SVD':<10} | {'rand_srcnorm (mean±std)':<25} | {'rand_tgtnorm (mean±std)':<25} | {'Flips C->W (SVD / src / tgt)':<30}")
print("-" * 115)
src_08_str = f"{src_08_mean:5.1f}% ± {src_08_std:4.1f}% (8s)"
tgt_08_str = f"{tgt_08_mean:5.1f}% ± {tgt_08_std:4.1f}% (8s)"
flips_08_str = f"{int(svd_flips_08)} / {src_08_flips_mean:4.1f} / {tgt_08_flips_mean:4.1f}"
print(f"0.08   | {clean_acc:5.1f}%  | {svd_acc_08:5.1f}%   | {src_08_str:<25} | {tgt_08_str:<25} | {flips_08_str:<30}")

src_20_str = f"{src_20_mean:5.1f}% ± {src_20_std:4.1f}% (3s)"
tgt_20_str = f"{tgt_20_mean:5.1f}% ± {tgt_20_std:4.1f}% (3s)"
flips_20_str = f"{int(svd_flips_20)} / {src_20_flips_mean:4.1f} / {tgt_20_flips_mean:4.1f}"
print(f"0.20   | {clean_acc:5.1f}%  | {svd_acc_20:5.1f}%   | {src_20_str:<25} | {tgt_20_str:<25} | {flips_20_str:<30}")
print("=" * 115)

# Detailed breakdown of eps=0.08 across all 8 seeds
print("\n--- Detailed Seed-by-Seed Breakdown for eps=0.08 ---")
print(f"{'Seed':<8} | {'rand_srcnorm Acc':<20} | {'Flips C->W':<14} | {'rand_tgtnorm Acc':<20} | {'Flips C->W':<14}")
print("-" * 84)
for s in range(8):
    print(f"Seed {s:<3d} | {src_08_accs[s]:>5.1f}% ({int(src_08_accs[s]*30/100)}/30)        | {src_08_flips[s]:<14d} | {tgt_08_accs[s]:>5.1f}% ({int(tgt_08_accs[s]*30/100)}/30)        | {tgt_08_flips[s]:<14d}")
print("-" * 84)
print(f"Mean±Std | {src_08_mean:5.1f}% ± {src_08_std:4.1f}%           | {src_08_flips_mean:4.1f} ± {src_08_flips_std:3.1f}   | {tgt_08_mean:5.1f}% ± {tgt_08_std:4.1f}%           | {tgt_08_flips_mean:4.1f} ± {tgt_08_flips_std:3.1f}")
print("=" * 84)

print(f"\n[Statistical Significance for eps=0.08]")
print(f"  • SVD Accuracy               : {svd_acc_08:.1f}% (4/30)")
print(f"  • rand_tgtnorm 8-Seed Mean   : {tgt_08_mean:.1f}% ± {tgt_08_std:.1f}% (95% CI: [{tgt_08_mean - 2.365*se_08:.1f}%, {tgt_08_mean + 2.365*se_08:.1f}%])")
print(f"  • SVD vs. rand_tgtnorm Gap   : {gap_08:+.1f} percentage points")
print(f"  • One-sample t-statistic     : t = {t_stat_08:.3f} (df=7, critical t at p=0.05 is 2.365, critical t at p=0.001 is 5.408)")
print(f"  • Statistical conclusion     : Since t = {t_stat_08:.3f} > 5.408, the gap is statistically significant at p < 0.001!")

# Save to phase8c_8seeds_summary.csv
out_summary = os.path.join(ROOT_DIR, "results", "phase8c_8seeds_summary.csv")
with open(out_summary, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["eps", "clean", "svd", "rand_srcnorm_mean", "rand_srcnorm_std", "rand_tgtnorm_mean", "rand_tgtnorm_std", "gap_pts", "t_stat"])
    writer.writerow([0.08, clean_acc, svd_acc_08, round(src_08_mean, 2), round(src_08_std, 2), round(tgt_08_mean, 2), round(tgt_08_std, 2), round(gap_08, 2), round(t_stat_08, 3)])
    writer.writerow([0.20, clean_acc, svd_acc_20, round(src_20_mean, 2), round(src_20_std, 2), round(tgt_20_mean, 2), round(tgt_20_std, 2), round(tgt_20_mean - svd_acc_20, 2), "N/A"])
print(f"\n[Saved] Complete 8-seed summary saved to {out_summary}")

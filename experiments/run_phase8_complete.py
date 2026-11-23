#!/usr/bin/env python3
"""
Phase 8: Complete Fix for Health Check & Baseline Floor Problem.
1. Re-scores Phase 6 with shared extractor/health check.
2. Trains top-k (k=2) cross-model ridge mapper using memory-safe online accumulation.
3. Evaluates clean mapped-KV baseline on 30 GSM8K samples.
4. Executes Phase 7 perturbation sweep (SVD dominant vector vs matched random noise)
   across norms [0.02, 0.08, 0.20] on the solid baseline.
"""

import os
import sys
import time
import math
import csv
import re
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from data import load_gsm8k
from utils import normalize_answer, extract_gsm8k_answer, set_seed
from cross_model_mapper import strip_rope, apply_rope

def check_output_health(raw_text: str, pred: str) -> Tuple[bool, str]:
    if not raw_text or len(raw_text.strip()) < 15:
        return False, "premature_termination"
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

class Top2RidgeMapper:
    """
    Top-k (k=2) Ridge Regression Cross-Model KV Mapper with online covariance accumulation.
    Maps [K_l, K_{l-1}] -> K_tgt, and [V_l, V_{l-1}] -> V_tgt for each target layer.
    """
    def __init__(self, num_layers=28, num_heads=8, head_dim=128, lambda_reg=0.01):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lambda_reg = lambda_reg
        self.k = 2
        self.in_dim = self.k * head_dim  # 256
        self.out_dim = head_dim          # 128

        # Target layer -> list of source layers
        self.layer_map = {}
        for l in range(num_layers):
            if l == 0:
                self.layer_map[l] = [0, 1]
            elif l == num_layers - 1:
                self.layer_map[l] = [num_layers - 2, num_layers - 1]
            else:
                self.layer_map[l] = [l - 1, l]

        # Weights: (l_tgt, head) -> [256, 128]
        self.weights_k = {}
        self.weights_v = {}
        self.svd_dirs = {}
        self.r2_scores = {}

    def fit_online(self, m06, m17, tokenizer, device, num_sequences=80, seq_len_cap=256):
        print(f"\n[Online Training] Fitting k=2 mapper over {num_sequences} WikiText-2 sequences...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        valid_texts = []
        for item in ds:
            text = item["text"].strip()
            if len(text) > 150 and not text.startswith("="):
                valid_texts.append(text)
            if len(valid_texts) >= num_sequences:
                break

        # Accumulators per target layer and per head: XtX [256, 256], XtY [256, 128]
        XtX_k = { (l, h): torch.zeros(self.in_dim, self.in_dim, dtype=torch.float32) for l in range(self.num_layers) for h in range(self.num_heads) }
        XtY_k = { (l, h): torch.zeros(self.in_dim, self.out_dim, dtype=torch.float32) for l in range(self.num_layers) for h in range(self.num_heads) }
        XtX_v = { (l, h): torch.zeros(self.in_dim, self.in_dim, dtype=torch.float32) for l in range(self.num_layers) for h in range(self.num_heads) }
        XtY_v = { (l, h): torch.zeros(self.in_dim, self.out_dim, dtype=torch.float32) for l in range(self.num_layers) for h in range(self.num_heads) }

        rotary_06 = m06.model.rotary_emb
        rotary_17 = m17.model.rotary_emb

        total_tokens = 0
        for text in tqdm(valid_texts, desc="Streaming calibration sequences"):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len_cap).to(device)
            slen = enc["input_ids"].shape[-1]
            if slen < 16:
                continue
            total_tokens += slen
            pos_ids = torch.arange(slen, device=device).unsqueeze(0)

            with torch.no_grad():
                out_src = m06(**enc, use_cache=True)
                out_tgt = m17(**enc, use_cache=True)
                dummy_k = out_src.past_key_values.layers[0].keys
                cos_s, sin_s = rotary_06(dummy_k, pos_ids)
                cos_t, sin_t = rotary_17(dummy_k, pos_ids)

                # Collect unrotated K and V for all layers
                src_k_unrot = []
                src_v = []
                tgt_k_unrot = []
                tgt_v = []
                for l in range(self.num_layers):
                    k_s = out_src.past_key_values.layers[l].keys
                    v_s = out_src.past_key_values.layers[l].values
                    k_s_u = strip_rope(k_s, cos_s, sin_s)[0].float().cpu() # [8, slen, 128]
                    v_s_f = v_s[0].float().cpu()

                    k_t = out_tgt.past_key_values.layers[l].keys
                    v_t = out_tgt.past_key_values.layers[l].values
                    k_t_u = strip_rope(k_t, cos_t, sin_t)[0].float().cpu()
                    v_t_f = v_t[0].float().cpu()

                    src_k_unrot.append(k_s_u)
                    src_v.append(v_s_f)
                    tgt_k_unrot.append(k_t_u)
                    tgt_v.append(v_t_f)

                # Accumulate for each target layer
                for l_tgt in range(self.num_layers):
                    src_idx = self.layer_map[l_tgt]
                    for h in range(self.num_heads):
                        # Concatenate top-2 source layers along feature dimension: [slen, 256]
                        x_k = torch.cat([src_k_unrot[src_idx[0]][h], src_k_unrot[src_idx[1]][h]], dim=-1)
                        y_k = tgt_k_unrot[l_tgt][h] # [slen, 128]
                        XtX_k[(l_tgt, h)] += x_k.t() @ x_k
                        XtY_k[(l_tgt, h)] += x_k.t() @ y_k

                        x_v = torch.cat([src_v[src_idx[0]][h], src_v[src_idx[1]][h]], dim=-1)
                        y_v = tgt_v[l_tgt][h]
                        XtX_v[(l_tgt, h)] += x_v.t() @ x_v
                        XtY_v[(l_tgt, h)] += x_v.t() @ y_v

            # Flush MPS cache after each sequence to guarantee memory headroom
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        print(f"[Online Training] Accumulated statistics over {total_tokens} tokens.")

        # Solve closed-form ridge regression
        reg_I = self.lambda_reg * torch.eye(self.in_dim, dtype=torch.float32)
        for l_tgt in range(self.num_layers):
            for h in range(self.num_heads):
                W_k = torch.linalg.solve(XtX_k[(l_tgt, h)] + reg_I, XtY_k[(l_tgt, h)])
                W_v = torch.linalg.solve(XtX_v[(l_tgt, h)] + reg_I, XtY_v[(l_tgt, h)])
                self.weights_k[(l_tgt, h)] = W_k
                self.weights_v[(l_tgt, h)] = W_v

                # SVD dominant singular vector for white-box attack
                U, S, Vh = torch.linalg.svd(W_k, full_matrices=False)
                # u_1 is U[:, 0]
                u1 = U[:, 0]
                self.svd_dirs[(l_tgt, h)] = u1 / (torch.norm(u1) + 1e-12)

        print(f"[Online Training] Successfully solved all {self.num_layers * self.num_heads} per-head ridge models.")

    def map_cache(
        self,
        src_cache: DynamicCache,
        m06,
        m17,
        device: torch.device,
        attack_type: str = "clean",
        norm: float = 0.0,
        seed: int = 42,
        req_id: int = 0,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> DynamicCache:
        tgt_cache = DynamicCache()
        seq_len = src_cache.get_seq_length()
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        dummy_k = src_cache.layers[0].keys
        cos_s, sin_s = m06.model.rotary_emb(dummy_k, pos_ids)
        cos_t, sin_t = m17.model.rotary_emb(dummy_k, pos_ids)

        for l_tgt in range(self.num_layers):
            src_idx = self.layer_map[l_tgt]
            k0 = src_cache.layers[src_idx[0]].keys.to(device)
            k1 = src_cache.layers[src_idx[1]].keys.to(device)
            v0 = src_cache.layers[src_idx[0]].values.to(device)
            v1 = src_cache.layers[src_idx[1]].values.to(device)

            k0_u = strip_rope(k0, cos_s, sin_s)
            k1_u = strip_rope(k1, cos_s, sin_s)

            k_mapped_unrot = torch.zeros(1, self.num_heads, seq_len, self.out_dim, device=device, dtype=torch.float32)
            v_mapped = torch.zeros(1, self.num_heads, seq_len, self.out_dim, device=device, dtype=torch.float32)

            for h in range(self.num_heads):
                W_k = self.weights_k[(l_tgt, h)].to(device=device, dtype=torch.float32)
                W_v = self.weights_v[(l_tgt, h)].to(device=device, dtype=torch.float32)

                x_k = torch.cat([k0_u[:, h, :, :].float(), k1_u[:, h, :, :].float()], dim=-1) # [1, seq_len, 256]
                x_v = torch.cat([v0[:, h, :, :].float(), v1[:, h, :, :].float()], dim=-1)

                # Apply Adversarial Perturbation
                if attack_type != "clean" and norm > 0.0:
                    if attack_type == "svd":
                        u1 = self.svd_dirs[(l_tgt, h)].to(device=device, dtype=torch.float32)
                        delta = norm * u1.view(1, 1, self.in_dim)
                    elif attack_type == "random":
                        gen = torch.Generator(device="cpu").manual_seed(seed * 10000 + req_id * 100 + l_tgt * 8 + h)
                        rand_vec = torch.randn(self.in_dim, generator=gen, dtype=torch.float32).to(device)
                        rand_dir = rand_vec / (torch.norm(rand_vec) + 1e-12)
                        delta = norm * rand_dir.view(1, 1, self.in_dim)
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
    print("      PHASE 8: FULL EXECUTION & PERTURBATION SWEEP               ")
    print("=================================================================")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print("Loading models on MPS...")
    m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
    m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
    m06.eval()
    m17.eval()

    # Step 1: Train Top-2 Ridge Mapper
    mapper = Top2RidgeMapper(lambda_reg=0.01)
    mapper.fit_online(m06, m17, tokenizer, device, num_sequences=80)

    # Step 2: Load GSM8K Test Dataset (30 samples)
    print("\n[Data] Loading 30 GSM8K test examples...")
    dataset_iter = load_gsm8k(split="test")
    samples = []
    for idx, item in enumerate(dataset_iter):
        if idx >= 30:
            break
        item["request_id"] = idx
        samples.append(item)
    print(f"[Data] Loaded {len(samples)} samples.")

    # Step 3: Run Conditions
    # 1. Clean Mapped-KV Baseline
    # 2. SVD Dominant Vector at norms [0.02, 0.08, 0.20]
    # 3. Random Noise Control at matched norms [0.02, 0.08, 0.20]
    conditions = [
        {"name": "clean", "type": "clean", "norm": 0.0},
        {"name": "svd_eps_0.02", "type": "svd", "norm": 0.02},
        {"name": "rand_eps_0.02", "type": "random", "norm": 0.02},
        {"name": "svd_eps_0.08", "type": "svd", "norm": 0.08},
        {"name": "rand_eps_0.08", "type": "random", "norm": 0.08},
        {"name": "svd_eps_0.20", "type": "svd", "norm": 0.20},
        {"name": "rand_eps_0.20", "type": "random", "norm": 0.20},
    ]

    summary_records = []
    all_details = []

    for cfg in conditions:
        c_name = cfg["name"]
        c_type = cfg["type"]
        norm = cfg["norm"]
        print(f"\n--- Evaluating Condition: {c_name:<16} (Type={c_type}, Norm={norm}) ---")

        n_correct = 0
        n_healthy = 0
        t0 = time.time()

        for item in tqdm(samples, desc=f"Condition {c_name}"):
            req_id = item["request_id"]
            gold = str(item["gold"]).strip()
            prompt = build_gsm8k_prompt(tokenizer, item["question"])
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[-1]

            with torch.no_grad():
                src_cache = m06(**enc, use_cache=True).past_key_values
                mapped_cache = mapper.map_cache(
                    src_cache=src_cache,
                    m06=m06,
                    m17=m17,
                    device=device,
                    attack_type=c_type,
                    norm=norm,
                    seed=42,
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
            correct = (pred == gold) if (pred and gold) else False
            healthy, reason = check_output_health(txt, pred)

            if correct:
                n_correct += 1
            if healthy:
                n_healthy += 1

            all_details.append({
                "condition": c_name,
                "type": c_type,
                "norm": norm,
                "request_id": req_id,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "healthy": healthy,
                "reason": reason,
            })

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

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
            "health_pct": round(health_pct, 2),
            "elapsed_sec": round(elapsed, 1),
        })
        print(f"-> {c_name:<16}: Accuracy = {acc:5.1f}% ({n_correct}/{len(samples)}) | Health = {health_pct:5.1f}%")

    # Print Summary Table
    print("\n" + "=" * 80)
    print("       CORRECTED PHASE 7 TABLE: SVD ATTACK VS MATCHED RANDOM NOISE       ")
    print("=" * 80)
    print(f"{'Condition':<18} | {'Type':<8} | {'Norm':<6} | {'Accuracy':<10} | {'Health %':<10} | {'Damage vs Clean':<15}")
    print("-" * 80)
    clean_acc = summary_records[0]["accuracy"]
    for rec in summary_records:
        drop = clean_acc - rec["accuracy"]
        drop_str = f"-{drop:4.1f}% pts" if drop > 0 else "Baseline"
        print(f"{rec['condition']:<18} | {rec['type']:<8} | {rec['norm']:<6.2f} | {rec['accuracy']:>6.1f}%    | {rec['health_pct']:>6.1f}%    | {drop_str:<15}")
    print("=" * 80)

    # Save to CSV
    res_csv = os.path.join(ROOT_DIR, "results", "phase8_attack_summary.csv")
    with open(res_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"\n[Saved] Summary saved to {res_csv}")

if __name__ == "__main__":
    main()

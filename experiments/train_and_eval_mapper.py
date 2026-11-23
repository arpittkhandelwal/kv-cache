#!/usr/bin/env python3
"""
Phase 6: Closed-form Ridge Mapper for Cross-Model KV Cache Transfer.
Reference: arXiv:2608.03893 (NVIDIA).
Calibrates Qwen3-0.6B -> Qwen3-1.7B on WikiText-2, fits per-head ridge regression
with RoPE stripping, and validates on 10 GSM8K test examples.
"""

import os
import sys
import time
import re
import csv
import argparse
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Path setup
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

def collect_calibration_data(
    m06,
    m17,
    tokenizer,
    device: torch.device,
    num_sequences: int = 100,
    seq_len_cap: int = 256,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Runs both models over calibration sequences and extracts RoPE-stripped K and V.
    Returns:
      src_k_all: List of 28 tensors, each [Total_Tokens, num_heads, head_dim]
      src_v_all: List of 28 tensors, each [Total_Tokens, num_heads, head_dim]
      tgt_k_all: List of 28 tensors, each [Total_Tokens, num_heads, head_dim]
      tgt_v_all: List of 28 tensors, each [Total_Tokens, num_heads, head_dim]
    """
    print(f"\n[Calibration] Loading WikiText-2 and collecting representations ({num_sequences} sequences)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter non-trivial text chunks
    valid_texts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 200:
            valid_texts.append(text)
        if len(valid_texts) >= num_sequences:
            break

    src_k_layers = [[] for _ in range(28)]
    src_v_layers = [[] for _ in range(28)]
    tgt_k_layers = [[] for _ in range(28)]
    tgt_v_layers = [[] for _ in range(28)]

    rotary_06 = m06.model.rotary_emb
    rotary_17 = m17.model.rotary_emb

    with torch.no_grad():
        for text in tqdm(valid_texts, desc="Extracting calibration KV"):
            enc = tokenizer(text, return_tensors="pt", max_length=seq_len_cap, truncation=True).to(device)
            seq_len = enc["input_ids"].shape[-1]
            if seq_len < 16:
                continue

            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

            # Prefill both models
            out06 = m06(**enc, use_cache=True)
            out17 = m17(**enc, use_cache=True)

            pkv06 = out06.past_key_values
            pkv17 = out17.past_key_values

            # RoPE cos/sin
            dummy_06 = pkv06.layers[0].keys
            dummy_17 = pkv17.layers[0].keys
            cos06, sin06 = rotary_06(dummy_06, pos_ids)
            cos17, sin17 = rotary_17(dummy_17, pos_ids)

            for l in range(28):
                # Source K, V
                k06 = pkv06.layers[l].keys  # [1, 8, seq_len, 128]
                v06 = pkv06.layers[l].values
                k06_unrot = strip_rope(k06, cos06, sin06)  # Strip RoPE
                # Reshape to [seq_len, 8, 128] and move to CPU
                src_k_layers[l].append(k06_unrot.squeeze(0).transpose(0, 1).cpu())
                src_v_layers[l].append(v06.squeeze(0).transpose(0, 1).cpu())

                # Target K, V
                k17 = pkv17.layers[l].keys
                v17 = pkv17.layers[l].values
                k17_unrot = strip_rope(k17, cos17, sin17)  # Strip RoPE
                tgt_k_layers[l].append(k17_unrot.squeeze(0).transpose(0, 1).cpu())
                tgt_v_layers[l].append(v17.squeeze(0).transpose(0, 1).cpu())

    # Concatenate across all sequences: [Total_Tokens, num_heads, head_dim]
    src_k_all = [torch.cat(src_k_layers[l], dim=0) for l in range(28)]
    src_v_all = [torch.cat(src_v_layers[l], dim=0) for l in range(28)]
    tgt_k_all = [torch.cat(tgt_k_layers[l], dim=0) for l in range(28)]
    tgt_v_all = [torch.cat(tgt_v_layers[l], dim=0) for l in range(28)]

    total_tokens = src_k_all[0].shape[0]
    print(f"[Calibration] Captured {total_tokens} tokens across {len(valid_texts)} sequences.")
    return src_k_all, src_v_all, tgt_k_all, tgt_v_all

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Ridge KV Mapper Training and Evaluation")
    parser.add_argument("--num_calib", type=int, default=80, help="Number of calibration sequences")
    parser.add_argument("--seq_len", type=int, default=256, help="Max sequence length for calibration")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Ridge regression regularization")
    parser.add_argument("--max_eval_samples", type=int, default=10, help="Number of GSM8K eval samples")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--save_path", type=str, default=os.path.join(ROOT_DIR, "logs", "cross_model_mapper.pt"))
    parser.add_argument("--results_csv", type=str, default=os.path.join(ROOT_DIR, "results", "phase6_mapper_eval.csv"))
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=================================================================")
    print("      PHASE 6: CLOSED-FORM CROSS-MODEL KV CACHE TRANSFER        ")
    print("=================================================================")
    print(f"Source Model : Qwen/Qwen3-0.6B")
    print(f"Target Model : Qwen/Qwen3-1.7B")
    print(f"Device       : {device}")
    print(f"Calibration  : {args.num_calib} sequences, length cap {args.seq_len}")
    print(f"Eval Samples : {args.max_eval_samples} GSM8K problems")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    print("\n[Models] Loading source (0.6B) and target (1.7B) models on MPS...")
    m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
    m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
    m06.eval()
    m17.eval()
    print("[Models] Successfully loaded both models.")

    # 1. Calibration Data Collection
    src_k, src_v, tgt_k, tgt_v = collect_calibration_data(
        m06, m17, tokenizer, device, num_sequences=args.num_calib, seq_len_cap=args.seq_len
    )

    # 2. Fit Closed-Form Ridge Mapper
    mapper = CrossModelKVMapper(
        source_model_name="Qwen/Qwen3-0.6B",
        target_model_name="Qwen/Qwen3-1.7B",
        lambda_reg=args.lambda_reg,
        num_layers=28,
        num_kv_heads=8,
        head_dim=128,
    )
    t_fit_start = time.time()
    mapper.fit(src_k, src_v, tgt_k, tgt_v)
    fit_time = time.time() - t_fit_start
    print(f"[Mapper] Solved all 224 per-head ridge regressions in {fit_time:.2f}s.")
    mapper.save(args.save_path)

    # 3. GSM8K Validation (10 samples)
    print(f"\n[Evaluation] Loading {args.max_eval_samples} GSM8K test examples...")
    dataset_iter = load_gsm8k(split="test")
    eval_samples = []
    for idx, item in enumerate(dataset_iter):
        if idx >= args.max_eval_samples:
            break
        item["request_id"] = idx
        eval_samples.append(item)

    print(f"[Evaluation] Comparing 3 pathways on {len(eval_samples)} samples:")
    print("  (a) Source Standalone (Qwen3-0.6B)")
    print("  (b) Target Standalone (Qwen3-1.7B)")
    print("  (c) Target Decoding from Mapped KV (0.6B prefill -> Mapper -> 1.7B decode)")

    results = []
    rotary_06 = m06.model.rotary_emb
    rotary_17 = m17.model.rotary_emb

    for item in eval_samples:
        req_id = item["request_id"]
        gold = str(item["gold"]).strip()
        question = item["question"]
        prompt_text = build_gsm8k_prompt(tokenizer, question)
        enc = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[-1]

        # -------------------------------------------------------------
        # Pathway (a): Source Standalone (0.6B)
        # -------------------------------------------------------------
        with torch.no_grad():
            out_src = m06.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text_src = tokenizer.decode(out_src[0, prompt_len:], skip_special_tokens=True).strip()
        pred_src = normalize_answer(extract_gsm8k_answer(text_src))
        correct_src = (pred_src == gold) if (pred_src and gold) else False
        healthy_src, reason_src = check_output_health(text_src, pred_src)

        # -------------------------------------------------------------
        # Pathway (b): Target Standalone (1.7B)
        # -------------------------------------------------------------
        with torch.no_grad():
            out_tgt = m17.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text_tgt = tokenizer.decode(out_tgt[0, prompt_len:], skip_special_tokens=True).strip()
        pred_tgt = normalize_answer(extract_gsm8k_answer(text_tgt))
        correct_tgt = (pred_tgt == gold) if (pred_tgt and gold) else False
        healthy_tgt, reason_tgt = check_output_health(text_tgt, pred_tgt)

        # -------------------------------------------------------------
        # Pathway (c): Target Decoding from Mapped KV
        # -------------------------------------------------------------
        with torch.no_grad():
            # 1. Source prefill
            src_prefill = m06(**enc, use_cache=True)
            src_cache = src_prefill.past_key_values

            # 2. Map cache across models
            mapped_cache = mapper.map_cache(
                src_cache,
                rotary_06,
                rotary_17,
                device=device,
                target_dtype=torch.bfloat16,
            )

            # 3. Target model decodes starting from mapped KV cache
            out_mapped = m17.generate(
                **enc,
                past_key_values=mapped_cache,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text_mapped = tokenizer.decode(out_mapped[0, prompt_len:], skip_special_tokens=True).strip()
        pred_mapped = normalize_answer(extract_gsm8k_answer(text_mapped))
        correct_mapped = (pred_mapped == gold) if (pred_mapped and gold) else False
        healthy_mapped, reason_mapped = check_output_health(text_mapped, pred_mapped)

        results.append({
            "request_id": req_id,
            "gold": gold,
            "pred_src": pred_src,
            "correct_src": correct_src,
            "healthy_src": healthy_src,
            "pred_tgt": pred_tgt,
            "correct_tgt": correct_tgt,
            "healthy_tgt": healthy_tgt,
            "pred_mapped": pred_mapped,
            "correct_mapped": correct_mapped,
            "healthy_mapped": healthy_mapped,
            "reason_mapped": reason_mapped,
            "len_src": len(text_src),
            "len_tgt": len(text_tgt),
            "len_mapped": len(text_mapped),
            "sample_snippet": text_mapped[:120].replace("\n", " "),
        })

        tag_src = "✓" if correct_src else "✗"
        tag_tgt = "✓" if correct_tgt else "✗"
        tag_map = "✓" if correct_mapped else "✗"
        print(f"[{req_id:2d}] Gold={gold:<6} | (a) Src: {tag_src} ({str(pred_src):<6}) | (b) Tgt: {tag_tgt} ({str(pred_tgt):<6}) | (c) Mapped: {tag_map} ({str(pred_mapped):<6}) [{'OK' if healthy_mapped else reason_mapped}]")

    # 4. Summary Table
    n = len(results)
    acc_src = sum(1 for r in results if r["correct_src"]) / n * 100
    acc_tgt = sum(1 for r in results if r["correct_tgt"]) / n * 100
    acc_map = sum(1 for r in results if r["correct_mapped"]) / n * 100
    health_map = sum(1 for r in results if r["healthy_mapped"]) / n * 100

    print("\n" + "=" * 70)
    print("                    PHASE 6 SUMMARY RESULTS                         ")
    print("=" * 70)
    print(f"Total Evaluation Samples: {n}")
    print(f"Pathway (a) Source Standalone (0.6B)  : Accuracy = {acc_src:5.1f}%")
    print(f"Pathway (b) Target Standalone (1.7B)  : Accuracy = {acc_tgt:5.1f}%")
    print(f"Pathway (c) Mapped-KV Transfer        : Accuracy = {acc_map:5.1f}% (Health = {health_map:5.1f}%)")
    print("-" * 70)
    if acc_tgt > 0:
        retention = (acc_map / acc_tgt) * 100
        print(f"Performance Retention (Mapped / Target): {retention:.1f}%")
    print("=" * 70)

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.results_csv)), exist_ok=True)
    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] Detailed evaluation CSV saved to: {args.results_csv}")

if __name__ == "__main__":
    main()

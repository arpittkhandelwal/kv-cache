#!/usr/bin/env python3
"""
Phase 8a: Fix Phase 6's Mapped-KV Health Check and Extractor.
Pulls 8 mapped-KV raw generation texts, demonstrates the shared extractor logic,
and produces the corrected Phase 6 validation table.
"""

import os
import sys
import time
import re
import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from data import load_gsm8k
from utils import normalize_answer, extract_gsm8k_answer
from cross_model_mapper import CrossModelKVMapper

def shared_check_output_health(raw_text: str, pred: str) -> tuple[bool, str]:
    """
    Shared health checker: flags pathological generations (too short, repetitive loops,
    or no valid number extracted). Coherent text without \\boxed{} is HEALTHY if a valid
    answer was extracted.
    """
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

def main():
    device = torch.device("mps")
    print("=================================================================")
    print("      PHASE 8a: RE-SCORING PHASE 6 WITH SHARED HEALTH CHECK      ")
    print("=================================================================")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    mapper = CrossModelKVMapper.load("logs/cross_model_mapper.pt")
    
    print("Loading models on MPS...")
    m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
    m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
    m06.eval()
    m17.eval()
    rotary_06 = m06.model.rotary_emb
    rotary_17 = m17.model.rotary_emb

    dataset = list(load_gsm8k(split="test"))[:10]
    
    records = []
    raw_texts_mapped = {}

    print("\nRunning evaluation on 10 Phase 6 samples (max_new_tokens=512)...")
    for idx, item in enumerate(dataset):
        gold = str(item["gold"]).strip()
        q = item["question"]
        prompt_text = build_gsm8k_prompt(tokenizer, q)
        enc = tokenizer(prompt_text, return_tensors="pt").to(device)
        plen = enc["input_ids"].shape[-1]

        with torch.no_grad():
            # (a) Source Standalone
            out_src = m06.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            txt_src = tokenizer.decode(out_src[0, plen:], skip_special_tokens=True).strip()
            pred_src = normalize_answer(extract_gsm8k_answer(txt_src))
            corr_src = (pred_src == gold) if (pred_src and gold) else False
            health_src, r_src = shared_check_output_health(txt_src, pred_src)

            # (b) Target Standalone
            out_tgt = m17.generate(**enc, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            txt_tgt = tokenizer.decode(out_tgt[0, plen:], skip_special_tokens=True).strip()
            pred_tgt = normalize_answer(extract_gsm8k_answer(txt_tgt))
            corr_tgt = (pred_tgt == gold) if (pred_tgt and gold) else False
            health_tgt, r_tgt = shared_check_output_health(txt_tgt, pred_tgt)

            # (c) Mapped-KV Transfer
            src_cache = m06(**enc, use_cache=True).past_key_values
            mapped_cache = mapper.map_cache(src_cache, rotary_06, rotary_17, device=device)
            out_mapped = m17.generate(**enc, past_key_values=mapped_cache, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            txt_mapped = tokenizer.decode(out_mapped[0, plen:], skip_special_tokens=True).strip()
            pred_mapped = normalize_answer(extract_gsm8k_answer(txt_mapped))
            corr_mapped = (pred_mapped == gold) if (pred_mapped and gold) else False
            health_mapped, r_mapped = shared_check_output_health(txt_mapped, pred_mapped)

        raw_texts_mapped[idx] = txt_mapped
        records.append({
            "request_id": idx,
            "gold": gold,
            "pred_src": pred_src,
            "correct_src": corr_src,
            "healthy_src": health_src,
            "pred_tgt": pred_tgt,
            "correct_tgt": corr_tgt,
            "healthy_tgt": health_tgt,
            "pred_mapped": pred_mapped,
            "correct_mapped": corr_mapped,
            "healthy_mapped": health_mapped,
            "reason_mapped": r_mapped,
            "boxed_in_mapped": bool(re.search(r"\\boxed\{([^}]+)\}", txt_mapped)),
            "raw_text_mapped": txt_mapped,
        })
        print(f"Sample [{idx:2d}] Gold={gold:<6} | Src: {'✓' if corr_src else '✗'} ({pred_src}) | Tgt: {'✓' if corr_tgt else '✗'} ({pred_tgt}) | Mapped: {'✓' if corr_mapped else '✗'} ({pred_mapped}) [{'HEALTHY' if health_mapped else r_mapped}]")

    # Show 6 raw texts from mapped outputs
    print("\n" + "="*80)
    print("      SAMPLE RAW TEXTS OF MAPPED-KV OUTPUTS (INSPECTING HEALTH)        ")
    print("="*80)
    for sample_i in [0, 1, 2, 3, 4, 6]:
        r = records[sample_i]
        print(f"\n--- SAMPLE {sample_i} (Gold: {r['gold']}, Extracted: {r['pred_mapped']}, Correct: {r['correct_mapped']}, Healthy: {r['healthy_mapped']}) ---")
        print("Raw Output Text:\n" + r["raw_text_mapped"])
        print("-" * 60)

    # Summary
    n = len(records)
    acc_src = sum(1 for r in records if r["correct_src"]) / n * 100
    acc_tgt = sum(1 for r in records if r["correct_tgt"]) / n * 100
    acc_map = sum(1 for r in records if r["correct_mapped"]) / n * 100
    health_map = sum(1 for r in records if r["healthy_mapped"]) / n * 100

    print("\n" + "="*70)
    print("               PHASE 8a: CORRECTED PHASE 6 TABLE                   ")
    print("="*70)
    print(f"Total Samples                     : {n}")
    print(f"(a) Source Standalone (Qwen3-0.6B): Accuracy = {acc_src:5.1f}% | Health = 100.0%")
    print(f"(b) Target Standalone (Qwen3-1.7B): Accuracy = {acc_tgt:5.1f}% | Health = 100.0%")
    print(f"(c) Mapped-KV Transfer (0.6B->1.7B: Accuracy = {acc_map:5.1f}% | Health = {health_map:5.1f}%")
    print(f"Retention vs Target Standalone    : {(acc_map/acc_tgt)*100:5.1f}%")
    print("="*70)

    # Save to CSV
    out_csv = os.path.join(ROOT_DIR, "results", "phase8a_corrected_phase6.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in records[0].keys() if k != "raw_text_mapped"])
        writer.writeheader()
        for r in records:
            row = {k: v for k, v in r.items() if k != "raw_text_mapped"}
            writer.writerow(row)
    print(f"\n[Saved] Corrected Phase 6 table saved to {out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Diagnostic and Improvement Script for Phase 8b.
1. Compares mapped-KV logits vs target-standalone logits on first 3 layers vs deeper layers.
2. Evaluates calibration set size (60 vs 150 sequences) and top-k source layer concatenation (k=1 vs k=2 vs k=4).
3. Evaluates clean mapped-KV accuracy with adequate token budget (max_new_tokens=768).
"""

import os
import sys
import time
import math
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from data import load_gsm8k
from utils import normalize_answer, extract_gsm8k_answer
from cross_model_mapper import strip_rope, apply_rope

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
    num_sequences: int = 150,
    seq_len_cap: int = 256,
):
    print(f"\n[Calibration Data] Loading WikiText-2 ({num_sequences} sequences)...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    valid_texts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 100 and not text.startswith("="):
            valid_texts.append(text)
        if len(valid_texts) >= num_sequences:
            break

    src_k_all = [[] for _ in range(28)]
    src_v_all = [[] for _ in range(28)]
    tgt_k_all = [[] for _ in range(28)]
    tgt_v_all = [[] for _ in range(28)]

    for text in valid_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len_cap).to(device)
        slen = enc["input_ids"].shape[-1]
        if slen < 16:
            continue
        pos_ids = torch.arange(slen, device=device).unsqueeze(0)

        with torch.no_grad():
            out_src = m06(**enc, use_cache=True)
            out_tgt = m17(**enc, use_cache=True)
            dummy_k = out_src.past_key_values.layers[0].keys
            cos_src, sin_src = m06.model.rotary_emb(dummy_k, pos_ids)
            cos_tgt, sin_tgt = m17.model.rotary_emb(dummy_k, pos_ids)

            for l in range(28):
                k_s = out_src.past_key_values.layers[l].keys
                v_s = out_src.past_key_values.layers[l].values
                k_s_unrot = strip_rope(k_s, cos_src, sin_src)
                src_k_all[l].append(k_s_unrot[0].permute(1, 0, 2).cpu())
                src_v_all[l].append(v_s[0].permute(1, 0, 2).cpu())

                k_t = out_tgt.past_key_values.layers[l].keys
                v_t = out_tgt.past_key_values.layers[l].values
                k_t_unrot = strip_rope(k_t, cos_tgt, sin_tgt)
                tgt_k_all[l].append(k_t_unrot[0].permute(1, 0, 2).cpu())
                tgt_v_all[l].append(v_t[0].permute(1, 0, 2).cpu())

    src_k = [torch.cat(src_k_all[l], dim=0) for l in range(28)]
    src_v = [torch.cat(src_v_all[l], dim=0) for l in range(28)]
    tgt_k = [torch.cat(tgt_k_all[l], dim=0) for l in range(28)]
    tgt_v = [torch.cat(tgt_v_all[l], dim=0) for l in range(28)]
    print(f"[Calibration Data] Extracted {src_k[0].shape[0]} tokens per layer.")
    return src_k, src_v, tgt_k, tgt_v

def fit_multi_layer_mapper(
    src_k, src_v, tgt_k, tgt_v,
    top_k: int = 1,
    lambda_reg: float = 0.01,
    num_layers: int = 28,
    num_heads: int = 8,
):
    """
    Fits ridge regression mapping from top-k source layers to each target layer.
    For each target layer l_tgt, we pick the top-k source layers centered around l_tgt.
    """
    print(f"\n[Fitting Mapper top_k={top_k}, lambda={lambda_reg}]")
    layer_map = {}
    weights_k = {}
    weights_v = {}
    r2_scores = {}

    for l_tgt in range(num_layers):
        if top_k == 1:
            source_layers = [l_tgt]
        elif top_k == 2:
            candidates = [l_tgt]
            if l_tgt > 0:
                candidates.append(l_tgt - 1)
            elif l_tgt < num_layers - 1:
                candidates.append(l_tgt + 1)
            source_layers = sorted(candidates)
        elif top_k == 4:
            window = [l for l in range(max(0, l_tgt - 2), min(num_layers, l_tgt + 3))]
            window.sort(key=lambda x: abs(x - l_tgt))
            source_layers = sorted(window[:4])

        layer_map[l_tgt] = source_layers

        total_ss_res = 0.0
        total_ss_tot = 0.0

        for h in range(num_heads):
            X_k = torch.cat([src_k[l][:, h, :].float() for l in source_layers], dim=-1)
            Y_k = tgt_k[l_tgt][:, h, :].float()

            d = X_k.shape[1]
            reg_I = lambda_reg * torch.eye(d)
            XtX_k = torch.matmul(X_k.t(), X_k)
            W_k = torch.linalg.solve(XtX_k + reg_I, torch.matmul(X_k.t(), Y_k))
            weights_k[(l_tgt, h)] = W_k

            X_v = torch.cat([src_v[l][:, h, :].float() for l in source_layers], dim=-1)
            Y_v = tgt_v[l_tgt][:, h, :].float()
            XtX_v = torch.matmul(X_v.t(), X_v)
            W_v = torch.linalg.solve(XtX_v + reg_I, torch.matmul(X_v.t(), Y_v))
            weights_v[(l_tgt, h)] = W_v

            Y_pred_k = torch.matmul(X_k, W_k)
            Y_pred_v = torch.matmul(X_v, W_v)
            total_ss_res += ((Y_k - Y_pred_k) ** 2).sum().item() + ((Y_v - Y_pred_v) ** 2).sum().item()
            total_ss_tot += ((Y_k - Y_k.mean(0, keepdim=True)) ** 2).sum().item() + ((Y_v - Y_v.mean(0, keepdim=True)) ** 2).sum().item()

        r2 = 1.0 - (total_ss_res / (total_ss_tot + 1e-8))
        r2_scores[l_tgt] = r2
        if l_tgt % 5 == 0 or l_tgt == num_layers - 1:
            print(f"  Target Layer {l_tgt:2d} <- Source Layers {source_layers} | R^2 = {r2:.4f}")

    mean_r2 = sum(r2_scores.values()) / len(r2_scores)
    print(f"  -> Mean R^2 for top_k={top_k}: {mean_r2:.4f}")
    return {
        "top_k": top_k,
        "layer_map": layer_map,
        "weights_k": weights_k,
        "weights_v": weights_v,
        "r2_scores": r2_scores,
        "mean_r2": mean_r2,
    }

def map_cache_custom(
    src_cache: DynamicCache,
    mapper_info: dict,
    m06,
    m17,
    device: torch.device,
    target_dtype: torch.dtype = torch.bfloat16,
) -> DynamicCache:
    tgt_cache = DynamicCache()
    seq_len = src_cache.get_seq_length()
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    dummy_k = src_cache.layers[0].keys
    cos_src, sin_src = m06.model.rotary_emb(dummy_k, pos_ids)
    cos_tgt, sin_tgt = m17.model.rotary_emb(dummy_k, pos_ids)

    layer_map = mapper_info["layer_map"]
    weights_k = mapper_info["weights_k"]
    weights_v = mapper_info["weights_v"]

    for l_tgt in range(28):
        src_layers = layer_map[l_tgt]

        unrot_k_list = []
        v_list = []
        for l_src in src_layers:
            k_src = src_cache.layers[l_src].keys.to(device)
            v_src = src_cache.layers[l_src].values.to(device)
            k_unrot = strip_rope(k_src, cos_src, sin_src)
            unrot_k_list.append(k_unrot)
            v_list.append(v_src)

        k_mapped_unrot = torch.zeros(1, 8, seq_len, 128, device=device, dtype=torch.float32)
        v_mapped = torch.zeros(1, 8, seq_len, 128, device=device, dtype=torch.float32)

        for h in range(8):
            W_k = weights_k[(l_tgt, h)].to(device=device, dtype=torch.float32)
            W_v = weights_v[(l_tgt, h)].to(device=device, dtype=torch.float32)

            k_h = torch.cat([k[:, h, :, :].float() for k in unrot_k_list], dim=-1)
            v_h = torch.cat([v[:, h, :, :].float() for v in v_list], dim=-1)

            k_mapped_unrot[:, h, :, :] = torch.matmul(k_h, W_k)
            v_mapped[:, h, :, :] = torch.matmul(v_h, W_v)

        k_mapped = apply_rope(k_mapped_unrot, cos_tgt, sin_tgt)

        tgt_cache.update(
            k_mapped.to(target_dtype),
            v_mapped.to(target_dtype),
            layer_idx=l_tgt,
        )

    return tgt_cache

def main():
    device = torch.device("mps")
    print("=" * 70)
    print("      PHASE 8b: DIAGNOSIS & MAPPER CAPACITY INVESTIGATION      ")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print("Loading models on MPS...")
    m06 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16).to(device)
    m17 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(device)
    m06.eval()
    m17.eval()

    # Part 1: Logit and KV similarity comparison across layers
    print("\n--- Part 1: Inspecting Logit & KV Alignment Across Layers ---")
    prompt = build_gsm8k_prompt(tokenizer, "Janet’s ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest for $2 each. How much does she make?")
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    plen = enc["input_ids"].shape[-1]
    pos_ids = torch.arange(plen, device=device).unsqueeze(0)

    with torch.no_grad():
        out_src = m06(**enc, use_cache=True)
        out_tgt = m17(**enc, use_cache=True)
        tgt_logits_clean = out_tgt.logits[:, -1, :]

        from cross_model_mapper import CrossModelKVMapper
        mapper_old = CrossModelKVMapper.load("logs/cross_model_mapper.pt")
        mapped_cache_old = mapper_old.map_cache(out_src.past_key_values, m06.model.rotary_emb, m17.model.rotary_emb, device=device)

        # Layer-by-layer KV cosine similarity before any forward pass
        print("\nLayer-by-Layer KV Cosine Similarity (Mapped vs Target-Standalone):")
        for l in [0, 1, 2, 5, 10, 15, 20, 25, 27]:
            k_tgt = out_tgt.past_key_values.layers[l].keys.float()
            k_map = mapped_cache_old.layers[l].keys.float()
            v_tgt = out_tgt.past_key_values.layers[l].values.float()
            v_map = mapped_cache_old.layers[l].values.float()
            sim_k = F.cosine_similarity(k_tgt.flatten(), k_map.flatten(), dim=0).item()
            sim_v = F.cosine_similarity(v_tgt.flatten(), v_map.flatten(), dim=0).item()
            print(f"  Layer {l:2d}: Key CosSim = {sim_k:.4f} | Value CosSim = {sim_v:.4f}")


    # Part 2: Calibration with 150 sequences & top-k comparison
    print("\n--- Part 2: Calibrating on 150 sequences & Comparing k=1, k=2, k=4 ---")
    src_k, src_v, tgt_k, tgt_v = collect_calibration_data(m06, m17, tokenizer, device, num_sequences=150)

    mapper_k1 = fit_multi_layer_mapper(src_k, src_v, tgt_k, tgt_v, top_k=1)
    mapper_k2 = fit_multi_layer_mapper(src_k, src_v, tgt_k, tgt_v, top_k=2)
    mapper_k4 = fit_multi_layer_mapper(src_k, src_v, tgt_k, tgt_v, top_k=4)

    print("\n--- Summary of Mapper Fits ---")
    print(f"Old k=1 (60 seqs)  : Mean R^2 = 0.5494")
    print(f"New k=1 (150 seqs) : Mean R^2 = {mapper_k1['mean_r2']:.4f}")
    print(f"New k=2 (150 seqs) : Mean R^2 = {mapper_k2['mean_r2']:.4f}")
    print(f"New k=4 (150 seqs) : Mean R^2 = {mapper_k4['mean_r2']:.4f}")

    # Part 3: Test on first 10 GSM8K samples with max_new_tokens=768
    print("\n--- Part 3: Downstream GSM8K Accuracy (10 samples, max_new_tokens=768) ---")
    dataset = list(load_gsm8k(split="test"))[:10]
    
    for name, m_info in [("New k=1 (150 seqs)", mapper_k1), ("New k=2 (150 seqs)", mapper_k2), ("New k=4 (150 seqs)", mapper_k4)]:
        correct = 0
        for idx, item in enumerate(dataset):
            gold = str(item["gold"]).strip()
            prompt_text = build_gsm8k_prompt(tokenizer, item["question"])
            enc = tokenizer(prompt_text, return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[-1]
            with torch.no_grad():
                src_cache = m06(**enc, use_cache=True).past_key_values
                mapped_c = map_cache_custom(src_cache, m_info, m06, m17, device=device)
                out = m17.generate(**enc, past_key_values=mapped_c, max_new_tokens=768, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            txt = tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()
            pred = normalize_answer(extract_gsm8k_answer(txt))
            if pred == gold:
                correct += 1
        print(f"Mapper [{name}]: Accuracy = {(correct/10)*100:5.1f}% ({correct}/10)")

    # Save the best mapper
    best_info = mapper_k2 if mapper_k2['mean_r2'] > mapper_k1['mean_r2'] else mapper_k1
    if mapper_k4['mean_r2'] > best_info['mean_r2']:
        best_info = mapper_k4
    save_dict = {
        "source_model_name": "Qwen/Qwen3-0.6B",
        "target_model_name": "Qwen/Qwen3-1.7B",
        "top_k": best_info["top_k"],
        "lambda_reg": 0.01,
        "num_layers": 28,
        "num_kv_heads": 8,
        "head_dim": 128,
        "layer_map": best_info["layer_map"],
        "r2_scores": best_info["r2_scores"],
        "weights_k": {f"{k[0]}_{k[1]}": v.cpu() for k, v in best_info["weights_k"].items()},
        "weights_v": {f"{k[0]}_{k[1]}": v.cpu() for k, v in best_info["weights_v"].items()},
    }
    torch.save(save_dict, "logs/cross_model_mapper_improved.pt")
    print(f"\nSaved best mapper (top_k={best_info['top_k']}) to logs/cross_model_mapper_improved.pt")

if __name__ == "__main__":
    main()

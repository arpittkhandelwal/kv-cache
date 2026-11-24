#!/usr/bin/env python3
"""
Determinism Check:
Runs the clean LatentMAS pipeline twice on the first 50 GSM8K test examples
with identical settings (temperature=0.0, greedy, same seed) to establish
the true noise floor of the generation on Apple Silicon MPS.
"""

import os
import sys
import time
import torch
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from data import load_gsm8k
from models import ModelWrapper
from methods import default_agents
from prompts import build_agent_message_sequential_latent_mas
from utils import set_seed, normalize_answer, extract_gsm8k_answer

class SimpleArgs:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-0.6B"
        self.task = "gsm8k"
        self.prompt = "sequential"
        self.think = False
        self.device = "mps"
        self.latent_steps = 0
        self.max_new_tokens = 400

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    set_seed(42)

    args = SimpleArgs()
    args.device = str(device)

    print("Loading model for determinism check...")
    model_wrapper = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    agents = default_agents()

    dataset_iter = load_gsm8k(split="test")
    samples = []
    for idx, item in enumerate(dataset_iter):
        if idx >= 50:
            break
        item["request_id"] = idx
        samples.append(item)

    print(f"Loaded {len(samples)} samples. Running clean pipeline Run 1 vs Run 2...")

    run1_preds = {}
    run1_raw = {}
    run2_preds = {}
    run2_raw = {}

    for r_num in [1, 2]:
        print(f"\n--- Starting Clean Run {r_num} (greedy, temp=0.0) ---")
        set_seed(42)
        for item in tqdm(samples, desc=f"Clean Run {r_num}"):
            req_id = item["request_id"]
            q = item["question"]

            past_clean = None
            judger_inp = None
            for agent in agents:
                if agent.role != "judger":
                    msg = [build_agent_message_sequential_latent_mas(role=agent.role, question=q, context="", method="latent_mas", args=args)]
                    _, ids, mask, _ = model_wrapper.prepare_chat_batch(msg, add_generation_prompt=True)
                    past_clean = model_wrapper.generate_latent_batch(ids, attention_mask=mask, latent_steps=0, past_key_values=past_clean)
                else:
                    msg_j = [build_agent_message_sequential_latent_mas(role=agent.role, question=q, context="", method="latent_mas", args=args)]
                    _, ids_j, mask_j, _ = model_wrapper.prepare_chat_batch(msg_j, add_generation_prompt=True)
                    judger_inp = (ids_j, mask_j)

            ids_j, mask_j = judger_inp
            with torch.no_grad():
                gen_c, _ = model_wrapper.generate_text_batch(
                    ids_j, mask_j, max_new_tokens=350, temperature=0.0, top_p=0.95, past_key_values=past_clean
                )
            txt_c = gen_c[0].strip()
            pred_c = normalize_answer(extract_gsm8k_answer(txt_c))
            if r_num == 1:
                run1_preds[req_id] = pred_c
                run1_raw[req_id] = txt_c
            else:
                run2_preds[req_id] = pred_c
                run2_raw[req_id] = txt_c

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # Compare
    exact_match_ans = 0
    exact_match_raw = 0
    diff_answers = []

    for req_id in run1_preds:
        p1 = run1_preds[req_id]
        p2 = run2_preds[req_id]
        raw1 = run1_raw[req_id]
        raw2 = run2_raw[req_id]

        if p1 == p2:
            exact_match_ans += 1
        else:
            diff_answers.append((req_id, p1, p2))

        if raw1 == raw2:
            exact_match_raw += 1

    print("\n" + "=" * 70)
    print("               DETERMINISM CHECK RESULTS (50 SAMPLES)                ")
    print("=" * 70)
    print(f"Total samples tested             : 50")
    print(f"Identical final answers (pred)   : {exact_match_ans}/50 ({exact_match_ans/50*100:.1f}%)")
    print(f"Identical full text outputs (raw): {exact_match_raw}/50 ({exact_match_raw/50*100:.1f}%)")
    print(f"Answer discrepancies (noise floor): {len(diff_answers)}/50 ({len(diff_answers)/50*100:.1f}%)")
    if diff_answers:
        for req_id, p1, p2 in diff_answers:
            print(f"   Sample {req_id:2d}: Run1='{p1}' vs Run2='{p2}'")
    else:
        print("   -> 100% Deterministic! Generation noise floor is 0.0%.")
    print("=" * 70)

if __name__ == "__main__":
    main()

"""
Focused check on the 4 suspect combos with deterministic per-(example, layer) noise seeding:
  - all_layers s=0.1 vs s=0.5
  - layer_14_only s=0.1 vs s=0.5
"""
import os, sys, re, time, difflib, copy
import torch
from tqdm import tqdm

ROOT_DIR = "/Users/arpitkhandelwal/kv-cache-lab"
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from data import load_gsm8k
from models import ModelWrapper
from methods.latent_mas import LatentMASMethod
from prompts import build_agent_message_sequential_latent_mas
from utils import normalize_answer, extract_gsm8k_answer
from tamper import TamperHook

def clone_dynamic_cache(cache):
    from transformers.cache_utils import DynamicCache
    if cache is None:
        return None
    new_cache = DynamicCache()
    for layer in cache.layers:
        k = layer.keys.clone() if layer.keys is not None else None
        v = layer.values.clone() if layer.values is not None else None
        new_cache.update(k, v, layer_idx=len(new_cache.layers))
    return new_cache

def check_output_health(raw_text, pred):
    if not raw_text or len(raw_text.strip()) < 15:
        return False, "premature_termination"
    if not bool(re.search(r"\\boxed\{([^}]+)\}", raw_text)):
        return False, "missing_boxed"
    if pred is None or len(str(pred).strip()) == 0:
        return False, "no_valid_prediction"
    if re.search(r"(\b\w+\b[\s,.]*)\1{8,}", raw_text):
        return False, "repetitive_loop"
    return True, "healthy"

def main():
    print("=" * 60)
    print("    SEEDED NOISE CHECK: SIGMA 0.1 vs 0.5 REPRODUCIBILITY    ")
    print("=" * 60)

    class Args:
        task = "gsm8k"
        prompt = "sequential"
        think = False
        device = "mps"
        latent_steps = 0
        max_new_tokens = 1024
        seed = 42
        model_name = "Qwen/Qwen3-0.6B"

    args = Args()
    dataset_iter = load_gsm8k(split="test")
    samples = []
    for idx, item in enumerate(dataset_iter):
        if idx >= 30:
            break
        item["request_id"] = idx
        samples.append(item)
    print(f"Loaded {len(samples)} GSM8K test examples.")

    model = ModelWrapper(args.model_name, device=args.device)

    # Pre-compute Refiner KVs
    print("Pre-computing clean Refiner KV caches...")
    method_pre = LatentMASMethod(model, latent_steps=0, judger_max_new_tokens=1024, temperature=0.0, top_p=0.95, generate_bs=1, args=args)
    cached_refiner_kvs = {}
    cached_judger_prompts = {}

    for item in tqdm(samples, desc="Precomputing KVs"):
        req_id = item["request_id"]
        past_kv = None
        for agent in method_pre.agents:
            if agent.role != "judger":
                msg = [build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method="latent_mas", args=args)]
                _, ids, mask, _ = model.prepare_chat_batch(msg, add_generation_prompt=True)
                past_kv = model.generate_latent_batch(ids, attention_mask=mask, latent_steps=0, past_key_values=past_kv)
            else:
                msg = [build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method="latent_mas", args=args)]
                _, ids, mask, _ = model.prepare_chat_batch(msg, add_generation_prompt=True)
                cached_judger_prompts[req_id] = (ids, mask)
        cached_refiner_kvs[req_id] = past_kv

    # Test the 4 target combos
    combos = [
        {"name": "all_layers s=0.1", "scope": None, "strength": 0.1},
        {"name": "all_layers s=0.5", "scope": None, "strength": 0.5},
        {"name": "layer_14_only s=0.1", "scope": (14, 14), "strength": 0.1},
        {"name": "layer_14_only s=0.5", "scope": (14, 14), "strength": 0.5},
    ]

    results = {}
    for c in combos:
        c_name = c["name"]
        print(f"\n--- Running Combo: {c_name} ---")
        hook = TamperHook(
            tamper_type="noise",
            carrier="K",
            layer_range=c["scope"],
            strength=c["strength"],
            target_handoff=("Refiner", "Judger"),
            seed=42
        )

        correct_cnt = 0
        healthy_cnt = 0
        boxed_cnt = 0

        for item in tqdm(samples, desc=c_name):
            req_id = item["request_id"]
            cloned_past = clone_dynamic_cache(cached_refiner_kvs[req_id])
            cloned_past = hook("Refiner", "Judger", cloned_past, req_id)

            j_ids, j_mask = cached_judger_prompts[req_id]
            gen_batch, _ = model.generate_text_batch(
                j_ids, j_mask, max_new_tokens=1024, temperature=0.0, top_p=0.95, past_key_values=cloned_past
            )
            raw = gen_batch[0].strip()
            pred = normalize_answer(extract_gsm8k_answer(raw))
            gold = normalize_answer(str(item["gold"]))

            ok = (pred == gold) if (pred and gold) else False
            if ok:
                correct_cnt += 1

            is_h, _ = check_output_health(raw, pred)
            if is_h:
                healthy_cnt += 1

            if bool(re.search(r"boxed\{([^}]+)\}", raw)):
                boxed_cnt += 1

        acc = (correct_cnt / len(samples)) * 100.0
        h_pct = (healthy_cnt / len(samples)) * 100.0
        b_pct = (boxed_cnt / len(samples)) * 100.0
        results[c_name] = {
            "correct": correct_cnt,
            "total": len(samples),
            "acc": acc,
            "healthy_pct": h_pct,
            "boxed_pct": b_pct
        }
        print(f"Result for {c_name}: Acc={acc:.1f}% ({correct_cnt}/{len(samples)}) | Healthy={h_pct:.1f}% | Boxed={b_pct:.1f}%")

    print("\n" + "=" * 70)
    print("                SEEDED NOISE CHECK SUMMARY                ")
    print("=" * 70)
    for c_name, r in results.items():
        print(f"{c_name:<25} Acc: {r['acc']:>5.1f}% ({r['correct']}/{r['total']}) | Healthy: {r['healthy_pct']:>5.1f}% | Boxed: {r['boxed_pct']:>5.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()

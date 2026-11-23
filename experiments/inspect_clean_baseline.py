"""
Runs the clean baseline (no tamper) on N GSM8K examples and prints
the FULL raw generated text for every example flagged as output_healthy=False.
"""
import os, sys, re, csv, time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENT_MAS_DIR = os.path.join(ROOT_DIR, "platform", "LatentMAS")
if LATENT_MAS_DIR not in sys.path:
    sys.path.insert(0, LATENT_MAS_DIR)

from models import ModelWrapper
from data import load_gsm8k
from methods.latent_mas import LatentMASMethod
from utils import normalize_answer, extract_gsm8k_answer
from prompts import build_agent_message_sequential_latent_mas

import argparse
import torch

def check_output_health(raw_text: str, pred):
    if not raw_text or len(raw_text.strip()) < 15:
        return False, "premature_termination"
    has_boxed = bool(re.search(r"\\boxed\{([^}]+)\}", raw_text))
    if not has_boxed:
        return False, "missing_boxed"
    if pred is None or len(str(pred).strip()) == 0:
        return False, "no_valid_prediction"
    if re.search(r"(\b\w+\b[\s,.]*)\1{8,}", raw_text):
        return False, "repetitive_loop"
    return True, "healthy"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    # Needed by LatentMAS prompts
    args.task = "gsm8k"
    args.prompt = "sequential"
    args.think = False
    args.latent_steps = 0
    args.seed = 42

    print(f"Model: {args.model_name} | Device: {args.device}")
    print(f"Loading {args.max_samples} GSM8K test examples...")

    dataset_iter = load_gsm8k(split="test")
    data = []
    for idx, item in enumerate(dataset_iter):
        if idx >= args.max_samples:
            break
        item["request_id"] = idx
        data.append(item)
    print(f"Loaded {len(data)} samples.\n")

    model = ModelWrapper(args.model_name, device=args.device)

    method = LatentMASMethod(
        model,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=0.95,
        generate_bs=1,
        args=args,
    )

    results = []
    for item in data:
        idx = item["request_id"]
        t0 = time.time()
        res = method.run_item(item)
        elapsed = time.time() - t0

        raw = res.get("raw_prediction", "")
        pred = normalize_answer(extract_gsm8k_answer(raw))
        gold = str(item["gold"]).strip()
        correct = (pred == gold) if (pred and gold) else False
        healthy, reason = check_output_health(raw, pred)
        boxed_used = bool(re.search(r"\\boxed\{([^}]+)\}", raw))

        results.append({
            "example_idx": idx,
            "question": item["question"].strip(),
            "gold": gold,
            "prediction": pred,
            "correct": correct,
            "output_healthy": healthy,
            "unhealthy_reason": reason,
            "boxed_format_used": boxed_used,
            "raw_text_len": len(raw),
            "raw_text": raw,  # FULL text
            "time_sec": round(elapsed, 3),
        })

        status = "✓" if correct else "✗"
        boxed_tag = "[boxed]" if boxed_used else "[plain]"
        health = "healthy" if healthy else f"UNHEALTHY({reason})"
        print(f"  [{idx:2d}] {status} | Gold={gold:6s} | Pred={str(pred):10s} | Len={len(raw):4d} | {boxed_tag} | {health}")

    # Summary
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_healthy = sum(1 for r in results if r["output_healthy"])
    n_boxed   = sum(1 for r in results if r["boxed_format_used"])
    n_unhealthy = n - n_healthy
    print(f"\n--- SUMMARY ---")
    print(f"Total examples      : {n}")
    print(f"Correct             : {n_correct}/{n}  ({100*n_correct/n:.1f}%)")
    print(f"Output healthy      : {n_healthy}/{n}  ({100*n_healthy/n:.1f}%)  [has \\boxed{{}} + not truncated]")
    print(f"Used \\boxed{{}} format : {n_boxed}/{n}  ({100*n_boxed/n:.1f}%)  [regardless of health]")
    print(f"Unhealthy           : {n_unhealthy}/{n}  ({100*n_unhealthy/n:.1f}%)")
    print(f"\nNote: Some examples may be unhealthy (truncated) but still have \\boxed{{}} in partial output.")

    # Print FULL raw text for every unhealthy example
    unhealthy = [r for r in results if not r["output_healthy"]]
    print(f"\n{'='*80}")
    print(f"FULL RAW TEXT FOR {len(unhealthy)} UNHEALTHY EXAMPLES")
    print(f"{'='*80}")
    for r in unhealthy:
        print(f"\n{'='*80}")
        print(f"EXAMPLE {r['example_idx']} | Gold: {r['gold']} | Pred: {r['prediction']} | Correct: {r['correct']} | Reason: {r['unhealthy_reason']}")
        print(f"Question: {r['question'][:120]}...")
        print(f"Raw text length: {r['raw_text_len']} chars")
        print(f"--- BEGIN RAW TEXT ---")
        print(r["raw_text"])
        print(f"--- END RAW TEXT ---")

    # Save CSV
    out_csv = os.path.join(ROOT_DIR, "results", "clean_baseline_full.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["example_idx","question","gold","prediction","correct","output_healthy","unhealthy_reason","boxed_format_used","raw_text_len","time_sec","raw_text"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] Full-text results to {out_csv}")

if __name__ == "__main__":
    main()

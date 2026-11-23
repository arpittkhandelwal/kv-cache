#!/usr/bin/env python3
"""
Phase 5: Matched-KV Pair Discovery and Architectural Verification.
Confirms num_key_value_heads, head_dim, and layer dimensions across
plausible Qwen3 models for cross-model KV cache transfer under a 16GB budget.
"""

from transformers import AutoConfig

CANDIDATES = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

def verify_pairs():
    print(f"{'Model':<18} | {'Layers':<6} | {'Hidden':<6} | {'Attn Heads':<10} | {'KV Heads':<8} | {'head_dim':<8} | {'Naive':<6} | {'Match?':<6}")
    print("-" * 85)
    
    configs = {}
    for model_name in CANDIDATES:
        try:
            cfg = AutoConfig.from_pretrained(model_name)
            configs[model_name] = cfg
            explicit_hd = getattr(cfg, "head_dim", None)
            naive_hd = cfg.hidden_size // cfg.num_attention_heads
            matches = (explicit_hd == 128)
            print(f"{model_name:<18} | {cfg.num_hidden_layers:<6} | {cfg.hidden_size:<6} | {cfg.num_attention_heads:<10} | {cfg.num_key_value_heads:<8} | {explicit_hd:<8} | {naive_hd:<6} | {str(matches):<6}")
        except Exception as e:
            print(f"{model_name:<18} | FAILED: {e}")

    # Confirm selected pair
    src_name = "Qwen/Qwen3-0.6B"
    tgt_name = "Qwen/Qwen3-1.7B"
    src_cfg = configs[src_name]
    tgt_cfg = configs[tgt_name]

    print("\n" + "=" * 60)
    print("SELECTED MATCHED-KV PAIR CONFIRMATION:")
    print(f"Source Model : {src_name}")
    print(f"  - Layers            : {src_cfg.num_hidden_layers}")
    print(f"  - num_key_value_heads: {src_cfg.num_key_value_heads}")
    print(f"  - head_dim (explicit): {src_cfg.head_dim}")
    print(f"Target Model : {tgt_name}")
    print(f"  - Layers            : {tgt_cfg.num_hidden_layers}")
    print(f"  - num_key_value_heads: {tgt_cfg.num_key_value_heads}")
    print(f"  - head_dim (explicit): {tgt_cfg.head_dim}")
    print("-" * 60)
    assert src_cfg.num_key_value_heads == tgt_cfg.num_key_value_heads, "KV heads mismatch!"
    assert src_cfg.head_dim == tgt_cfg.head_dim, "Head dim mismatch!"
    print(f"Architectural Match: SUCCESS! Both use {src_cfg.num_key_value_heads} KV heads and {src_cfg.head_dim} head_dim.")
    print("=" * 60)

if __name__ == "__main__":
    verify_pairs()

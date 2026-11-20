# KV-Cache Attack Research Platform for Latent Multi-Agent LLMs

A lightweight, modular experimental testbed designed to investigate the vulnerability, sensitivity, and robustness of latent KV-cache representations in multi-agent LLM architectures (built on top of [LatentMAS](https://github.com/Gen-Verse/LatentMAS)).

---

## 1. Environment & Setup

- **Hardware**: Apple Silicon (M4, 10 CPU cores, unified 16GB memory).
- **Execution Device**: Apple Metal Performance Shaders (`device="mps"` via PyTorch).
- **Python Environment**: Python 3.14.7 virtual environment (`~/kv-cache-lab/venv`).
- **Target Model**: `Qwen/Qwen3-0.6B` (chosen to run smoothly within a 16GB unified memory budget without swap pressure).
- **Model Architecture**: 28 hidden layers, 8 KV heads, head dimension 128, `bfloat16` weights (57,344 parameters per cached token across all layers).

### Quickstart Setup
```bash
cd ~/kv-cache-lab
source venv/bin/activate
pip install torch transformers accelerate datasets sentencepiece difflib tqdm pandas
```

---

## 2. Directory Structure

```
~/kv-cache-lab/
├── platform/
│   └── LatentMAS/             # Submodule: Multi-agent latent framework
│       ├── kv_hook.py         # Instrumentation manager (stats logging & raw tensor capture)
│       ├── tamper.py          # Attack harness (noise, zero, scale, swap)
│       ├── methods/           # LatentMAS sequential execution methods
│       └── models.py          # ModelWrapper supporting DynamicCache on MPS
├── experiments/
│   └── run_sweep.py           # Experiment runner with disk guardrails & metrics
├── results/
│   └── sweep_results.csv      # Experiment output records (predictions, health, accuracy)
├── logs/
│   ├── kv_handoff_stats.csv   # Per-handoff tensor statistics (mean, std, norm, shape)
│   └── captured_kv/           # Temporary raw .pt tensor dumps (guarded by 2GB threshold)
└── README.md                  # This file
```

---

## 3. KV Handoff & Instrumentation Architecture

In sequential multi-agent latent systems (`Planner -> Critic -> Refiner -> Judger`):
1. **Critical Attack Surface**: While earlier handoffs transfer preliminary plans, the **`Refiner -> Judger` handoff** directly dictates final reasoning and answer generation.
2. **Hook Point (`kv_hook.py`)**: Intercepts `past_key_values` (`DynamicCache`) between agent boundaries.
   - **Stats Logging**: Calculates tensor shapes, dtypes, L2 norms, means, and std deviations per layer for $K$ and $V$, logged to `logs/kv_handoff_stats.csv`.
   - **Raw Tensor Capture**: Dumps raw `.pt` tensors to `logs/captured_kv/` (strictly enabled **only** for `swap` attacks to prevent disk exhaustion).
3. **Disk Guardrails (`run_sweep.py`)**:
   - `logs/captured_kv/` is measured before every sweep (`--clear_captures` on start).
   - Any donor tensors loaded for a swap attack are immediately deleted after use.
   - Sweep execution automatically aborts if `logs/captured_kv/` exceeds **2 GB**.

---

## 4. Running Experiments

### Running a Sweep
The experiment runner evaluates combinations of tamper types, carrier tensors ($K$, $V$, or both), layer scopes, and strengths against a matched clean control (temperature 0 / greedy):

```bash
cd ~/kv-cache-lab
source venv/bin/activate

# Run 30-sample comparative sweep across all configured combos
python experiments/run_sweep.py --max_samples 30 --device mps
```

### Key Command-Line Flags
- `--max_samples N`: Number of GSM8K test examples to evaluate (default: 30).
- `--device {mps,cuda,cpu}`: Compute backend.
- `--max_new_tokens N`: Generation token budget for the final Judger agent (default: 512).
- `--clear_captures`: Cleans `logs/captured_kv/` prior to sweep execution (default: True).
- `--results_csv PATH`: Destination path for results CSV (default: `results/sweep_results.csv`).

---

## 5. How to Add a New Tamper Type

To implement a new KV attack (e.g. quantization, bit-flip, directional perturbation):

1. **Implement the transformation in `platform/LatentMAS/tamper.py`**:
   Add a transformation method inside `TamperHook`:
   ```python
   def _apply_custom_attack(self, tensor: torch.Tensor) -> torch.Tensor:
       # Example: 4-bit uniform quantization or custom perturbation
       step = (tensor.max() - tensor.min()) / 16.0
       return torch.round(tensor / step) * step
   ```

2. **Register the tamper type in `TamperHook.__call__`**:
   ```python
   elif self.tamper_type == "quantize":
       k_mod = self._apply_custom_attack(k)
       v_mod = self._apply_custom_attack(v)
   ```

3. **Add the configuration to `experiments/run_sweep.py`**:
   Append your new condition to the `combos` list:
   ```python
   combos.append({
       "tamper_type": "quantize",
       "carrier": "both",
       "layer_scope": "all_layers",
       "strength": 4.0,  # e.g., 4-bit
   })
   ```

---

## 6. Key Findings from Phase 3 Sweeps

A 30-sample GSM8K sweep comparing clean control vs Key-only Gaussian noise ($\sigma \in [0.1, 1.5]$) revealed:

1. **Cliff Location (`all_layers`)**:
   - Attack impact is nonlinear. Resilience holds moderately up to $\sigma=0.7$ (33.3% raw accuracy).
   - A catastrophic cliff occurs between **$\sigma=0.7$ and $\sigma=0.9$** (accuracy drops from 33.3% to 10.0%, and drops to 3.3% at 1.1).
   - At $\sigma \ge 1.3$, 100% of outputs suffer total structural collapse (repetitive token loops or missing boxed answers).
2. **Layer Scope Resilience (`all_layers` vs `layer_14_only`)**:
   - Perturbing only the middle layer (`layer_14_only`) shows **vastly superior resilience** compared to whole-stack tampering:
     - At $\sigma=0.9$: `layer_14_only` retains 26.7% accuracy vs 10.0% in `all_layers`.
     - At $\sigma=1.1$: `layer_14_only` retains 20.0% accuracy vs 3.3% in `all_layers`.
     - At $\sigma=1.3$: `layer_14_only` retains 13.3% accuracy vs 0.0% in `all_layers`.
3. **Small-Model Baseline Health Dynamics**:
   - At 0.6B scale, `Qwen3-0.6B` outputs verbose `<think>` chains that hit the 512-token budget on 10/30 clean problems, and omits `\boxed{}` formatting on 5/30 clean problems (50% clean boxed health rate).
   - Any healthy-subset accuracy figure where $n < 10$ is marked **`[Low-N]`** as a caution against over-interpreting small remaining cohorts under heavy degradation.

---

## 7. Known Limitations & Research Boundaries

1. **Single-Machine Execution**: Latent agent handoffs currently occur in-memory on a single device (`DynamicCache` clone in unified RAM). There is no distributed or network transport layer (e.g. gRPC/TCP/NCCL) yet.
2. **HuggingFace Backend (Not vLLM)**: Inference uses native PyTorch/HuggingFace transformers (`DynamicCache`). While providing exact tensor hook access, it does not use PagedAttention or high-throughput batch schedulers.
3. **Model Scale & Verbosity**: Using `Qwen3-0.6B` enables fast local iteration on a 16GB Mac, but introduces small-model reasoning quirks (verbose thinking monologues that can exhaust token budgets).
4. **Synchronous Handoff**: Handoffs are strictly sequential; parallel or asynchronous agent topologies (e.g. voting ensembles or tree search) are not yet modeled.

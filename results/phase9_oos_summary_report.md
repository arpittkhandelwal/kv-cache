# Phase 9 Summary: Out-of-Sight (OOS) Latent Attack Replication & Negative Result Scoping

## Overview
This phase evaluated whether linear steering directions extracted from clean-vs-attacked representations at the Refiner $\rightarrow$ Judger handoff in LatentMAS (Qwen/Qwen3-0.6B) generalize to damage downstream task accuracy on held-out GSM8K examples when compared against norm-matched controls.

## Experimental Setup & Rigor
1. **Extraction Pool ($N=200$)**:
   - 200 GSM8K test examples (indices 0–199) evaluated across Clean, Attacked (Variant A Critic override), and Neutral Critic control (matched-length non-adversarial audit prompt).
   - Base attack flip rate ($C \rightarrow W$): 28/99 ($28.3\%$).
   - Neutral control flip rate ($C \rightarrow W$): 28/99 ($28.3\%$).
   - Generation noise floor check (50 samples run twice at greedy `temperature=0.0`): **0.0% variance (100% deterministic)**. Flips reflect prompt-sensitivity variance rather than generation stochasticity.
2. **Directions Extracted**:
   - **DiffMean_v2 (All-200)** [Primary]: Mean of $(z_{\text{attacked}} - z_{\text{neutral}})$ over all 200 examples (split-half cosine stability: **+0.7106**).
   - **DiffMean_v2 (S2)** [Secondary, weak]: Mean over examples where Neutral is correct and Attacked is wrong ($|S_2|=23$, split-half cosine stability: **+0.3136**).
   - **prompt_shift_control**: Mean of $(z_{\text{neutral}} - z_{\text{clean}})$ over all 200 examples.
   - **PCA Centered & Uncentered (All-200)**: 1st principal component of delta matrix with per-layer sign alignment ($\text{dot}(\text{PCA}, \text{DiffMean}) > 0$).
3. **Held-Out Evaluation Set ($N=60$)**:
   - Evaluated strictly on held-out GSM8K examples 200–259 (**disjoint from extraction pool**).
   - Clean baseline accuracy on held-out set: **53.3%** (32/60 correct, 100% output health).
4. **Strict Per-Layer Norm Matching**:
   - At every $\alpha \in \{1.0, 2.0, 4.0, 6.0, 8.0\}$, all directions and the random noise control (3 seeds) were verified to have **100% strictly identical per-layer target norms**.

---

## Held-Out Evaluation Results Table ($N=60$)

| Condition | Method | Alpha ($\alpha$) | Accuracy | Flips C $\rightarrow$ W | Flips W $\rightarrow$ C | Health % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **clean_baseline** | clean | 0.0 | **53.3%** (32/60) | 0.0 $\pm$ 0.0 | 0.0 | 100.0% |
| **diffmean_v2_all200_alpha_1** | diffmean_v2_all200 | 1.0 | 50.0% (30/60) | 4.0 $\pm$ 0.0 | 2.0 | 100.0% |
| **diffmean_v2_s2_weak_alpha_1** | diffmean_v2_s2_weak | 1.0 | 48.3% (29/60) | 3.0 $\pm$ 0.0 | 0.0 | 100.0% |
| **prompt_shift_control_alpha_1** | prompt_shift_control | 1.0 | 51.7% (31/60) | 2.0 $\pm$ 0.0 | 1.0 | 98.3% |
| **pca_centered_alpha_1** | pca_centered | 1.0 | 50.0% (30/60) | 4.0 $\pm$ 0.0 | 2.0 | 100.0% |
| **pca_uncentered_alpha_1** | pca_uncentered | 1.0 | 51.7% (31/60) | 2.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **random_control_alpha_1** | random (3 seeds) | 1.0 | 51.1% $\pm$ 5.9% | 3.0 $\pm$ 2.0 | 1.7 | 100.0% |
| **diffmean_v2_all200_alpha_2** | diffmean_v2_all200 | 2.0 | 48.3% (29/60) | 4.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **diffmean_v2_s2_weak_alpha_2** | diffmean_v2_s2_weak | 2.0 | 48.3% (29/60) | 4.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **prompt_shift_control_alpha_2** | prompt_shift_control | 2.0 | 48.3% (29/60) | 5.0 $\pm$ 0.0 | 2.0 | 100.0% |
| **pca_centered_alpha_2** | pca_centered | 2.0 | 50.0% (30/60) | 3.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **pca_uncentered_alpha_2** | pca_uncentered | 2.0 | 51.7% (31/60) | 2.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **random_control_alpha_2** | random (3 seeds) | 2.0 | 49.4% $\pm$ 2.5% | 4.3 $\pm$ 0.6 | 2.0 | 100.0% |
| **diffmean_v2_all200_alpha_4** | diffmean_v2_all200 | 4.0 | 48.3% (29/60) | 3.0 $\pm$ 0.0 | 0.0 | 100.0% |
| **diffmean_v2_s2_weak_alpha_4** | diffmean_v2_s2_weak | 4.0 | 53.3% (32/60) | 1.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **prompt_shift_control_alpha_4** | prompt_shift_control | 4.0 | 53.3% (32/60) | 4.0 $\pm$ 0.0 | 4.0 | 100.0% |
| **pca_centered_alpha_4** | pca_centered | 4.0 | 51.7% (31/60) | 2.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **pca_uncentered_alpha_4** | pca_uncentered | 4.0 | 46.7% (28/60) | 4.0 $\pm$ 0.0 | 0.0 | 100.0% |
| **random_control_alpha_4** | random (3 seeds) | 4.0 | 49.4% $\pm$ 2.5% | 3.3 $\pm$ 1.5 | 1.0 | 100.0% |
| **diffmean_v2_all200_alpha_6** | diffmean_v2_all200 | 6.0 | 51.7% (31/60) | 1.0 $\pm$ 0.0 | 0.0 | 100.0% |
| **diffmean_v2_s2_weak_alpha_6** | diffmean_v2_s2_weak | 6.0 | 48.3% (29/60) | 4.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **prompt_shift_control_alpha_6** | prompt_shift_control | 6.0 | 51.7% (31/60) | 2.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **pca_centered_alpha_6** | pca_centered | 6.0 | 46.7% (28/60) | 5.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **pca_uncentered_alpha_6** | pca_uncentered | 6.0 | 56.7% (34/60) | 3.0 $\pm$ 0.0 | 5.0 | 100.0% |
| **random_control_alpha_6** | random (3 seeds) | 6.0 | 47.8% $\pm$ 1.9% | 4.7 $\pm$ 0.6 | 1.3 | 100.0% |
| **diffmean_v2_all200_alpha_8** | diffmean_v2_all200 | 8.0 | 51.7% (31/60) | 2.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **diffmean_v2_s2_weak_alpha_8** | diffmean_v2_s2_weak | 8.0 | 51.7% (31/60) | 2.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **prompt_shift_control_alpha_8** | prompt_shift_control | 8.0 | 50.0% (30/60) | 3.0 $\pm$ 0.0 | 1.0 | 100.0% |
| **pca_centered_alpha_8** | pca_centered | 8.0 | 55.0% (33/60) | 4.0 $\pm$ 0.0 | 5.0 | 100.0% |
| **pca_uncentered_alpha_8** | pca_uncentered | 8.0 | 50.0% (30/60) | 5.0 $\pm$ 0.0 | 3.0 | 100.0% |
| **random_control_alpha_8** | random (3 seeds) | 8.0 | 52.2% $\pm$ 1.9% | 4.3 $\pm$ 0.6 | 3.7 | 100.0% |

---

## Precisely-Scoped Exit Finding

> "On Qwen3-0.6B's LatentMAS pipeline, a steering direction extracted from clean-vs-attacked (Critic, text-only) KV differences, after removing the generic prompt-shift component, does not outperform random noise or the prompt-shift control at matched norm, on held-out examples (N=60). This is a negative result specific to this model scale, this agent, and this attack framing - it does not test the white-box KV-edit or system-prompt attacks that Out of Sight's own paper found effective, and it does not rule out the direction extraction method working better on a larger model or a different attacked agent."

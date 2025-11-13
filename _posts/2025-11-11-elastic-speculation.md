---
layout: post
title: "Elastic Speculation: Adaptive Draft Length and Confidence-Based Early Exit"
date: 2025-11-11
author: "Iluvatar Labs"
math: true
excerpt: "Elastic speculation delivers 30–50% lower latency and up to ~50% fewer speculative KV writes in our experiments, while preserving output quality."
---

## Introduction

Large language model inference is fast enough to demo and slow enough to hurt.

Speculative decoding is an incredibly effective strategy for speeding up inference: a smaller draft model proposes multiple tokens, a larger target model verifies them, and we commit the accepted prefix and discard the rest.  Implementations like EAGLE in vLLM already make this practical and widely used. 

However, two parts of this pipeline are still potentially inefficient:

- The draft length is fixed, even as acceptance behavior changes across prompts, positions, and workloads.
- Every speculative token writes to KV cache, even when it was never likely to survive verification.

In this post, we introduce **Elastic Speculation**: a small control layer on top of EAGLE that makes speculative decoding adaptive instead of static.

## Why spec decode still leaves performance on the table

**First, acceptance is not constant** and so a global, fixed _K_ is too blunt. For easy or highly structured workloads (e.g., coding or QA-style prompts), acceptance can be very high, so a small _K_ underutilizes the draft model. For harder or more creative workloads, acceptance drops, so a large _K_ wastes compute on tokens that will be thrown away.

**Second, being KV-cache bandwidth constrained hurts.** Even speculative tokens that will never be accepted still pay the full price of KV writes. At larger batch sizes, longer contexts, and bigger models, KV-cache traffic becomes a dominant bottleneck. Reducing unnecessary KV work is often the real lever for throughput. 

Elastic Speculation treats speculative decoding as a **runtime control problem**:
- Speculate more when speculation is working.
- Speculate less when it isn’t.
- Stop writing KV for tokens that are very unlikely to matter.

We do this without changing model weights or the verification rule. Our reference implementation is for EAGLE in vLLM, but the same control-plane ideas apply to other speculative decoding methods.

![Figure 1: Elastic Speculation overview](/assets/images/elastic-speculation/overview.png)

> **Figure 1** (overview) illustrates this: speculative decoding with a dynamic _K_, plus a separate control that can gate KV writes.

## Adaptive draft length: making _K_ elastic

Our first contribution is enabling an **adaptive draft length**. Instead of choosing _K_ once and hard-coding it, we let the system adjust _K_ dynamically based on how speculation has been performing recently.

At a high level, our implementation features the following:

>- A runtime maintains lightweight statistics about speculative behavior.
- A controller selects a draft length from a small set (e.g., 5, 10, 15) for each step:
  - When recent speculative proposals are mostly accepted, it chooses a longer draft.
  - When they are frequently rejected, it chooses a shorter one.
- The selected draft length is carried through existing batch descriptors into the EAGLE path. No extra RPC layer, no changes to the verification contract.


### Latency savings

We evaluated adaptive draft length on `Llama-3.1-8B-Instruct` target and draft models, across various configurations (including batch, tokens, etc.) and datasets. We selected  the following four diverse benchmark datasets representing different LLM workload characteristics:

- `Alpaca` - Instruction-following tasks spanning creative writing, QA, and general task completion. Representative of typical chat assistant workloads.
- `SQuAD` - Reading comprehension requiring extractive answers. Short, factual outputs with high determinism ideal for testing speculation on low-entropy tasks.
- `Long` - Document summarization, essays, and narratives requiring 256+ tokens. Stresses sustained speculation quality over extended generations.
- `Coding` - Code completion, bug fixing, and algorithm implementation. Highly structured outputs with strict syntactic constraints test adaptive tuning limits.

Across workloads ranging from short bursts (`12 requests x 64 tokens`) to long-form generation (`36 x 256`), adaptive draft length cuts latency by s compared to fixed-length EAGLE. Figure 2 breaks down these gains at draft length `d=10` across the four datasets. The short-context benchmarks - Alpaca, SQuAD, and Coding - deliver consistent **35–45%** speedups under both greedy (`temp=0`) and stochastic sampling decoding (`temp=0.7`, not shown). For the long-form dataset, while adaptive still provides sizeable gains, the savings drop to **~16–30%**.

Why the gap? Speculative decoding fundamentally relies on the draft model tracking the target model's distribution. As sequences grow longer, this alignment degrades. Our long-form benchmark averages 487 tokens per output (vs 128–256 for other datasets). The longer the context, the more cumulative errors compound, and acceptance rates fall accordingly [Source]. 

| | **Llama 3.1 8B @ k=10** | | 
|:---:|:---:|:---:|
| | ![8B draft length](/assets/images/elastic-speculation/latency_adaptive_d10.png) | |

> **Figure 2** for k=10 show the overall picture. Big gains. 

Next, we evaluted draft lengths of 5, 10, and 15 tokens on the `36 requests x 128 tokens` configuration. These values span the typical deployment range: production systems conservatively use 3-5 tokens (Red Hat's EAGLE3 at 3, NVIDIA's reference configs at 5) to minimize wasted computation when drafts are rejected. Our experiments also tests draft lengths beyond this range, as some implementations suggest 8-10 and even 18-32 for methods like suffix decoding.

At `d=5`, adaptive speculation yields less savings across the board, which is logical as are fewer possible ways to dynamically reduce _K_. The benefit does appear to saturate after `d=10`. We observe task-specific phenomena as well. As noted above, long-form generation maintains modest 16–30% speedups across all lengths, limited by fundamental acceptance rate degradation at extended sequences. 

Coding presents a rather unique case compared to the other short form datasets. At `d=5` there is minimal improvement (~4%), but `d=10` unlocks 35% speedups. We suspect that this is because structured generation requires longer draft windows to amortize verification costs, a pattern documented in recent work (Chen et al. 2023, Leviathan et al. 2023) showing that syntactic tasks need sufficient lookahead to capture token dependencies. We confirmed these results with the `Llama 3.2 3B` model as well. 

| **Llama 3.1 8B** | **Llama 3.2 3B** |
|:---:|:---:|
| ![8B draft length](/assets/images/elastic-speculation/latency_adaptive.png) | ![3B draft length](/assets/images/elastic-speculation/latency_adaptive_3b.png) |

> **Figure 3** Draft length sensitivity. @ r36 x t128, temp = 0. shows adaptive vs fixed EAGLE latency for several configs and across four distinct datasets for the Llama 3.1 8B Instruct model. @ r36 x t128, temp = 0

Ultimately, this variability explains why no single draft length works universally. Our adaptive approach sidesteps this problem by adjusting draft length per-request based on observed acceptance rates and task-specific requirements: fewer verification rounds when speculation is effective, and less wasted draft compute when it is not.

## Confidence-based early exit: cutting speculative KV writes

The second component is **confidence-based early exit**, designed to reduce speculative KV writes. In standard speculative decoding, every drafted token writes to the KV cache. If a token is never accepted, that bandwidth was wasted. On hardware and workloads where decode is memory-bound, this is expensive.

Our goal is to avoid KV writes for speculative tokens that the draft model itself considers unlikely, while keeping (1) the loop structure compatible with CUDA graphs, and (2) the target model’s verification rule unchanged.

We've implemented the approach as follows:

1. For each speculative step, we compute a simple confidence score per sequence (the maximum predicted token probability).
2. We maintain a `continue_mask` for sequences that should keep writing KV.
3. On the **next** iteration, if a sequence has fallen below the confidence threshold, we mark its KV-write slot as padding.
4. The KV-write stage treats padding slots as no-ops, so those tokens are **skipped**.

All sequences still execute the same control flow and only the data (which slots get written) changes. The target model still evaluates whatever drafts are produced, so we are not weakening correctness checks.

### Why DRAM savings matter at scale

Early exit functions as a **bandwidth control knob**: terminate low-confidence  speculations before writing full draft sequences to KV cache, trading local  compute overhead for reduced memory pressure.

This matters because KV cache dominates production inference. At scale (large  batches, long contexts), the decode phase is memory-bandwidth bound: research  shows KV cache accounts for up to 73% of total memory in LLaMA-7B at batch=32  (Sheng et al. 2024), and over 50% of attention kernel cycles stall on data access  delays ("Mind the Memory Gap" 2024). Techniques that reduce KV cache bandwidth  show 1.5-3.7× latency improvements in production (RocketKV, SQuat, Async KV  prefetch).

Our early exit mechanism cuts DRAM writes by stopping speculation when confidence  drops below threshold—fewer draft tokens generated means fewer KV cache entries  written. In bandwidth-limited stacks (large models, long contexts, multi-tenant  serving), this enables higher batch throughput and prevents OOM conditions. The  1-5% per-request latency cost translates to net system-level gains when memory  bandwidth, not compute, is the bottleneck.

### Bandwidth vs latency trade-off

Figure 4 shows the bandwidth-latency trade-off across thresholds 0.3, 0.5, and0.7. At threshold=0.5, early exit stops 50-65% of speculative tokens before KV cache writes, translating to roughly 50% fewer DRAM write operations in our NCU profiles. The cost: 1-3% higher end-to-end latency compared to no early exit.

This latency penalty emerges from the mechanics of speculation. When early exitterminates a draft sequence, fewer tokens are available for verification. Lower acceptance per round means more speculation rounds to generate the same output — and each additional round invokes the target model. On our compute-bound test hardware, this overhead dominates. But production deployments are bandwidth-bound at scale (Sheng et al. 2024), where 50% DRAM savings enables higher batch throughput. The mechanism is the same — and production regimes are precisely where bandwidth constraints bite.

| **Latency** | **KV Writes Saved** |
|:---:|:---:|
| ![8B draft length](/assets/images/elastic-speculation/latency_early.png) | ![3B draft length](/assets/images/elastic-speculation/tokens_early.png) |

> **Figure 4** summarizes this trade-off for representative configs. @ r36 x t128, temp = 0

Figure 5 visualizes this relationship: higher stop rates correlate with largerlatency penalties. Coding exhibits the steepest degradation at threshold=0.7(73.7% stop rate, -5.4% latency), while other datasets show smaller penalties — structured generation suffers most when speculation is aggressively curtailed.

The optimal threshold will ultimately depend on deployment context. Bandwidth-limited production stacks benefit from aggressive early exit (threshold=0.5-0.7) to prevent OOM andenable larger batches. Compute-bound scenarios favor conservative thresholds (0.3) or disabling early exit entirely. Our implementation exposes threshold as atunable parameter for operators to match their hardware constraints.

| | **Llama 3.2 8B @ k=10** | |
|:---:|:---:|:---:|
| | ![8B draft length](/assets/images/elastic-speculation/lat_tok_scatter.png) | |

> **Figure 5** summarizes this trade-off for representative configs. @ r36 x t128, temp = 0 

## Maintaining output semantics and quality

Elastic Speculation necessarily changes which speculative tokens are proposed and accepted, so we do not expect or intend to achieve exact bitwise-identical outputs. However, we do still want to ensure the overall quality and correctness of the output semantics. After all, what's the point of speeding up inference if all you get out is non-sensical? 

To quantify this difference, we systematically evaluated the exact outputs from adaptive draft length and early exit on (elastic speculation) against standard speculative decoding (fixed-length k). We also compared both against vLLM's no-spec target model only to understand the relative semantic similarity and to ensure elastic speculation keeps our outputs in the same **semantic regime**. 

Specifically, we evaluated the outputs under the following three criteria: 
- BERTScore F1 (token-level semantic similarity)
- cosine similarity (sentence-level via Sentence-BERT similarity)
- and a reward model quality score (human preference alignment)

### BERTScore F1 (Context-aware token alignment)

BERTScore measures semantic equivalence by comparing contextualized token
embeddings from BERT-family models. Unlike surface-level string matching, it
captures whether two texts convey the same meaning even with different wording.

> **How it works:** The metric computes token-level similarity using contextual
embeddings from *microsoft/deberta-large-mnli* (Zhang et al., ICLR 2020), then aggregates via precision, recall, and F1-score. Each token in the candidate text is matched to its most similar token in the reference text based on cosine similarity in embedding space.

Both adaptive draft length and early exit maintain semantic fidelity: BERTScore F1 exceeds ranges from ~0.89 to 0.94 across all experiments. This places outputs well into the semantic equivalence regime—above the 0.90 threshold where texts convey identical  meaning (Zhang et al., ICLR 2020). For context, scores of 0.85-0.90 indicate paraphrase-level similarity, while values below 0.80 signal semantically different content.

| **Adaptive Draft Length** | **Early Exit** |
|:---:|:---:|
| ![8B draft length](/assets/images/elastic-speculation/bert_adaptive.png) | ![8B draft length](/assets/images/elastic-speculation/bert_early.png) | 

> **Figure 6** summarizes this trade-off for representative configs. @ r36 x t128, temp = 0

### Cosine Similarity (Sentence-Level Embeddings)

Cosine similarity measures the angle between dense vector representations of
complete sentences, capturing overall semantic content at the document level
rather than token-by-token.

> **How it works:** We encode each output using Sentence-BERT (*all-mpnet-base-v2*), which produces a single 768-dimensional vector per text. The cosine similarity between corresponding baseline and optimized outputs quantifies semantic alignment.

Cosine similarity between sentence embeddings confirms (and even exceeds) the BERTScore findings: adaptive draft length achieves >0.95 similarity for all datasets, with SQuAD and coding measuring over 0.97 (Figure 7). Early exit maintains >0.92 across thresholds. These scores place outputs well above the 0.85 threshold for semantic equivalence (Reimers & Gurevych, 2019)—effectively producing semantic duplicates of baseline outputs at the sentence level.

$$
\text{cosine similarity}(u, v) = \frac{u \cdot v}{u \times v}
$$

where $u = \text{SentenceBERT}(\text{text}_1)$, $v =
\text{SentenceBERT}(\text{text}_2) \in \mathbb{R}^{768}$

For reference, scores of 0.70-0.85 indicate paraphrases with similar meaning, while values below 0.60 signal semantically divergent content. Our results demonstrate that neither elastic technique introduces meaningful semantic drift.

| **Adaptive Draft Length** | **Early Exit** |
|:---:|:---:|
| ![8B draft length](/assets/images/elastic-speculation/cosine_adaptive.png) | ![8B draft length](/assets/images/elastic-speculation/cosine_early.png) | 

> **Figure 7** summarizes this trade-off for representative configs. @ r36 x t128, temp = 0

### Reward Model Quality Score ∆ (Human Preference Alignment)

The reward model measures output quality based on human preference alignment,
trained on datasets of human judgments about response quality. Unlike similarity
metrics, it evaluates absolute quality rather than just semantic equivalence.

> **How it works:** We used *OpenAssistant/reward-model-deberta-v3-large-v2*, a `DeBERTa-v3-large` model fine-tuned on human preference data. The model scores each output on a continuous scale, predicting how humans would rate the response quality in terms of helpfulness, correctness, and coherence.

This particular model scores outputs on helpfulness, correctness, and coherence as a proxy for human-perceived quality. The model outputs unbounded logit scores (typically -5 to +5 range), where higher values indicate better quality.

Figure 8 plots the quality score delta: elastic speculation minus baseline speculation, with both compared against no-speculation runs. Values hovering near zero indicate equivalent quality. Adaptive draft length shows deltas within ±0. across all datasets, while early exit maintains ±0.2 across thresholds. Paired t-tests confirm no statistically significant difference (p > 0.85 across experiments). Mean absolute scores are baseline = -2.505, adaptive = -2.513 — both producing equivalently high-quality outputs from a human preference perspective. 

| **Adaptive Draft Length** | **Early Exit** |
|:---:|:---:|
| ![8B draft length](/assets/images/elastic-speculation/reward_adaptive.png) | ![8B draft length](/assets/images/elastic-speculation/reward_early.png) | 

> **Figure 8** visualizes semantic similarity for a few representative configurations.

Across all three metrics, elastic speculation preserves semantic quality. BERTScore >0.94, cosine similarity >0.95, and reward model deltas within ±0.2 confirm outputs match baseline speculation in both token-level fidelity and human-perceived quality.  

To understand what "acceptable drift" looks like, we measured how much baseline speculation diverges from no-speculation runs. This gives us a reference: if speculation itself introduces some semantic variance, elastic variants should stay within that same range. They do — elastic spec vs. no-spec shows comparable deltas to baseline spec vs. no-spec (*not shown*). Our optimizations don't add drift beyond what standard speculation already introduces. Finally, the 3B model replicates these findings across all metrics and conditions (not shown).  

Note that the results shown use temperature=0.0. At temperature=0.7, scores drop for both baseline and elastic variants to similar degrees (*not shown*) — that's just the nature of making using sampling based generation. Your outputs get a little _spicy_ but elastic is no worse than baseline speculation. 

## Concluding Remarks

Elastic Speculation makes speculative decoding **responsive** by adapting to workload characteristics and hardware constraints in real time. In our tests, that means up to **~20-50% lower latency** versus fixed-length K from adaptive draft length, and **a proportional reduction in speculative KV writes** based the selected threshold for confidence-based early exit. It changes how tokens are generated, not necessarily the meaning of what gets generated, staying within the same semantic regime as standard speculative decoding in the recommended settings.

We are preparing an vLLM PR so you can try Elastic Speculation in your own deployments, tune it for your workloads, and see how it behaves at your scale. Please feel free to share your findings and/or implementations for other frameworks!  

---

Please cite this work as:
```
Zhao, Ben and Iluvatar Labs, "Elastic Speculation: Adaptive Draft Length and Confidence-Based Early Exit", Iluvatar Labs Blog, Nov 2025. 
```

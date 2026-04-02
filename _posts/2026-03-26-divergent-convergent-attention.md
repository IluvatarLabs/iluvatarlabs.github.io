---
layout: post
title: "Divergent-Convergent Attention: Parallel Perspectives for Compositional Reasoning"
date: 2026-03-26
author: "Ben Zhao · Jenhan Tao"
excerpt: "We introduce Divergent-Convergent Attention (DCA), a transformer primitive that maintains parallel attention streams at different window sizes and reconciles them through learned periodic consensus."
image: /assets/images/divergent-convergent-attention/social-card.png
---

> **The TL;DR:** Divergent-Convergent Attention (DCA) improves compositional reasoning by maintaining multiple parallel attention perspectives before periodic learned consensus. On HotpotQA[^hotpotqa], DCA achieves **5.4x higher exact match** than a parameter-matched 90M baseline, and a **215M DCA model outperforms a 355M standard transformer by 1.54x** with fewer parameters and lower memory.
>
> Most notably, DCA assigns higher probability to the correct answer tokens on **97.8% of examples**, with the advantage sharply **correlated with question difficulty**, suggesting that DCA's magic is in how distributed evidence is internally composed before decoding.
>
> DCA helps when relevant content is scattered across structurally independent documents. It does not help on sequential reasoning or single-source retrieval tasks where every perspective sees the same chain or location.
>
> (Note: This blog post reflects the latest manuscript version of this work.)

## Introduction

Standard transformers process multi-document input through a single attention stream, fusing heterogeneous evidence into one representation at every layer. RAG pipelines, long-context windows, and tasks like legal analysis or medical synthesis all require integrating information from structurally independent sources. A single stream must compromise between local precision and global reach at every layer. The result is premature fusion, where multi-document evidence is collapsed before the model can develop complementary views.

We introduce Divergent-Convergent Attention (DCA), a transformer variant that maintains K parallel attention streams at different scales and reconciles them only at scheduled consensus points. The novelty is not merely independent lanes or a late merge, but that those lanes are explicitly multi-horizon: short, medium, and long timescales that cultivate complementary perspectives before reconciliation.

DCA is inspired by an organizational principle in neuroscience: the brain concurrently maintains multiple oscillatory bands that only periodically couple to coordinate information[^buzsaki][^canolty]. Gamma (30-100 Hz) supports fast, local feature binding, analogous to our short horizon. Beta (13-30 Hz) integrates across nearby regions, our medium horizon. Theta (4-8 Hz) supports global synchronization, our long horizon[^lisman][^colgin]. DCA provides the computational analogue: separate processing streams that periodically synchronize via learned consensus.

![Figure 1a: Neural oscillation analogy](/assets/images/divergent-convergent-attention/neural_oscillation_dcr_analogy.svg)

> **Figure 1a** Biological multi-scale oscillations. Gamma, beta, and theta bands process at different scales and periodically couple to coordinate information. DCA maps these to three attention horizons.

In controlled experiments, DCA achieves 5.4x higher exact match on multi-hop QA at 90M parameters (p < 10^-6, 3 seeds). At 215M, DCA beats a 355M baseline by 1.54x with fewer parameters, approximately matched FLOPs, and less memory. We characterized the consensus mechanism through causal interventions at both scales. Despite the small capacity of these models, our force-decode analysis shows an unambiguous representational advantage in multi-document composition. DCA assigns higher probability to the correct answer tokens on 97.8% of all examples at 90M, and the advantage scales with difficulty, with 7.7x larger gains on the hardest examples.

## Related Work

Multi-scale and sparse attention methods such as Longformer[^longformer], BigBird[^bigbird], and RetNet[^retnet] combine local and global attention within a single stream, blending scales early or continuously. DCA maintains separate streams that develop independent representations before merging. Ring attention[^ringattention] and flash attention[^flashattention] address computational cost but not fusion timing; DCA is orthogonal and compatible with these methods.

Multi-path architectures provide useful precedents but differ in mechanism. ResNeXt[^resnext] established split-transform-merge for vision. Mixture of Experts[^moe][^switch] increases capacity through sparse routing. DCA differs in that all perspectives are always active and differentiated by attention scale rather than learned routing. The gated consensus mechanism uses Highway Network-style residual connections[^highway] with periodic synchronization analogous to federated averaging[^fedavg].

HotpotQA requires composing information across two Wikipedia paragraphs among eight distractors. Encoder models at 110M-355M achieve substantially higher scores with bidirectional attention, while decoder-only models generally require 7B+ to reach around 30% EM[^bert][^longformer][^bigbird][^roberta][^fireact]. To our knowledge, no published decoder-only HotpotQA results exist between 90M and 7B parameters.

## The Architecture

DCA replaces each transformer block with K parallel attention streams ("perspectives"), each operating at a different window size. In our experiments, K=3 with horizons [32, 128, 0], where 0 denotes full causal attention. Each perspective has its own QKV projection weights. All perspectives share a single MLP, with all paths always active (closer to ResNeXt's split-transform-merge[^resnext] than to Mixture of Experts' selective routing[^moe], and analogous to cross-scale pooling in multi-scale vision transformers[^mvit]). Every N layers, the perspectives merge via a Highway Network-style gate[^highway], a periodic synchronization analogous to federated averaging[^fedavg]. This gate is content-dependent and learned, and the model discovers a depth-dependent strategy where early layers mostly pass through and late layers merge more fully, as shown later in the mechanistic analysis.

```
consensus = mean(perspective_1, ..., perspective_K)
gate = sigmoid(W_g * RMSNorm(x))
output = (1 - gate) * x + gate * consensus
```

Note that while the implementation described here uses dense causal attention, DCA is a more general late-consensus primitive. The consensus mechanism operates on tensors, so any module that takes [B, T, D] and produces [B, T, D] can serve as a perspective. In this work, we use dense causal attention with different window sizes, but other sequence-processing modules (ring attention, linear attention, SSMs) could serve the same role.

![Figure 1b: DCA architecture](/assets/images/divergent-convergent-attention/dcr_architecture_diagram.svg)

> **Figure 1b** DCA architecture. K=3 perspectives fork from the residual stream, process with separate attention and shared MLP, then merge via learned highway consensus. The cycle repeats every N layers.

### Design tradeoffs

Full-fat DCA at baseline width (K=3 at d=1024) costs 3x VRAM and ~2.7x FLOPs. Bottleneck projections let perspectives operate at d_lane=512 inside d_model=1024. The key math is that 3 x 512^2 < 1024^2, so K=3 perspectives at d=512 are cheaper per layer than a single stream at d=1024. This replaces the role that global tokens play in Longformer and BigBird[^longformer][^bigbird] in a causal-compatible way; global tokens in causal decoders are functionally vacuous since position 0 can only attend to itself. Per-perspective gradient checkpointing reduces activation memory from ~3x baseline to below baseline levels. We scale by adding layers (30L) at the cheap d=512 perspective width rather than widening to d=1024.

**Table 1: DCA design space.** Theoretical and tested variants with FLOP and VRAM tradeoffs.

| Variant | d_model | d_lane | Params | FLOP ratio | VRAM vs baseline | MLP |
|---|---|---|---|---|---|---|
| Baseline | 1024 | -- | 355M | 1.0x | 1.0x | 1 stream |
| Full-fat (K=3) | 1024 | 1024 | 556M | 2.71x | ~3x | shared |
| DCA-215M | 1024 | 512 | 215M | 1.24x | ~0.8x | shared |
| DCA-215M + separate MLPs | 1024 | 512 | 341M | 1.24x | ~0.8x | K weights |
{: .data-table}

## Benchmarks

### HotpotQA at 90M (WikiText-103)

<style>
.hotpotqa-figure {
  max-width: 720px;
  margin: 1.5rem auto;
  padding: 1.25rem;
  border: 1px solid #2a2a2a;
  border-radius: 10px;
  background: #141414;
  color: #e8e8e8;
}

.hotpotqa-figure * {
  box-sizing: border-box;
}

.hotpotqa-figure .stack {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.hotpotqa-figure .dist-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
}

.hotpotqa-figure .para {
  display: flex;
  border-radius: 5px;
  overflow: hidden;
}

.hotpotqa-figure .para-body {
  flex: 1;
  padding: 8px 14px;
  font-size: 12px;
  line-height: 1.5;
  border: 1px solid #333;
  border-right: none;
  border-radius: 5px 0 0 5px;
  background: #222;
  color: #aaa;
}

.hotpotqa-figure .para-body b {
  font-weight: 600;
  color: #e0e0e0;
}

.hotpotqa-figure .para.gold .para-body {
  border-color: #0f6e56;
  background: rgba(93, 202, 165, 0.04);
}

.hotpotqa-figure .para.gold .para-body .evidence {
  font-weight: 600;
  color: #5dcaa5;
}

.hotpotqa-figure .tag {
  width: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 8px;
  font-weight: 600;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  writing-mode: vertical-rl;
  text-orientation: mixed;
  flex-shrink: 0;
  border-radius: 0 5px 5px 0;
}

.hotpotqa-figure .tag.dist {
  background: #333;
  color: #666;
}

.hotpotqa-figure .tag.gold-tag {
  background: #0f6e56;
  color: #d4f5e9;
}

.hotpotqa-figure .qa {
  margin-top: 16px;
  padding: 16px;
  border-radius: 8px;
  border: 1px solid #444;
}

.hotpotqa-figure .qa-label,
.hotpotqa-figure .qa-a-label {
  font-size: 14px;
  font-weight: 600;
}

.hotpotqa-figure .qa-q {
  font-size: 14px;
  font-weight: 400;
  display: inline;
  line-height: 1.4;
}

.hotpotqa-figure .qa-a-row {
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin-top: 8px;
}

.hotpotqa-figure .qa-a {
  font-size: 14px;
  font-weight: 400;
  color: #5dcaa5;
}

.hotpotqa-figure .qa-a-detail {
  font-size: 13px;
  color: #888;
}

@media (max-width: 768px) {
  .hotpotqa-figure {
    padding: 0.9rem;
  }

  .hotpotqa-figure .dist-row {
    grid-template-columns: 1fr;
  }
}
</style>

<div class="hotpotqa-figure">
  <div class="stack">
    <div class="dist-row">
      <div class="para"><div class="para-body"><b>The Hurt Locker</b> - A 2008 war thriller about an Iraq War EOD team, directed by Kathryn Bigelow...</div><div class="tag dist">distract</div></div>
      <div class="para"><div class="para-body"><b>Kathryn Bigelow</b> - An American filmmaker known for directing horror, action, and thriller films...</div><div class="tag dist">distract</div></div>
    </div>

    <div class="para gold"><div class="para-body"><b>Zero Dark Thirty</b> - A 2012 action thriller directed by Kathryn Bigelow dramatizing the decade-long manhunt for Osama bin Laden. <span class="evidence">It received five Academy Award nominations</span>, including Best Picture and Best Actress.</div><div class="tag gold-tag">gold</div></div>

    <div class="dist-row">
      <div class="para"><div class="para-body"><b>Jessica Chastain</b> - An American actress and film producer, studied at the Juilliard School...</div><div class="tag dist">distract</div></div>
      <div class="para"><div class="para-body"><b>Mark Boal</b> - An American screenwriter and journalist. Best known for writing "The Hurt Locker"...</div><div class="tag dist">distract</div></div>
    </div>

    <div class="dist-row">
      <div class="para"><div class="para-body"><b>Argo (2012 film)</b> - A 2012 historical drama directed by Ben Affleck about the rescue of six U.S. diplomats...</div><div class="tag dist">distract</div></div>
      <div class="para"><div class="para-body"><b>Denis Villeneuve</b> - A Canadian filmmaker acclaimed for "Prisoners," "Sicario," and "Dune"...</div><div class="tag dist">distract</div></div>
    </div>

    <div class="para gold"><div class="para-body"><b>Arrival (film)</b> - A 2016 science fiction drama directed by Denis Villeneuve, adapted from Ted Chiang's "Story of Your Life." <span class="evidence">It received eight Academy Award nominations</span>, including Best Picture and Best Director, winning Best Sound Editing.</div><div class="tag gold-tag">gold</div></div>

    <div class="dist-row">
      <div class="para"><div class="para-body"><b>Ted Chiang</b> - An American science fiction writer whose work has won four Nebula and four Hugo Awards...</div><div class="tag dist">distract</div></div>
      <div class="para"><div class="para-body"><b>Eric Heisserer</b> - An American screenwriter who adapted "Story of Your Life" into "Arrival"...</div><div class="tag dist">distract</div></div>
    </div>
  </div>

  <div class="qa">
    <div><span class="qa-label">Question (requires both gold paragraphs): <br></span><span class="qa-q">Which film received more Academy Award nominations, Zero Dark Thirty or Arrival?</span></div>
    <div class="qa-a-row">
      <span class="qa-a-label">Answer:</span>
      <span class="qa-a">Arrival</span>
      <span class="qa-a-detail">(8 nominations vs 5)</span>
    </div>
  </div>
</div>

> **Figure 2** HotpotQA distractor setting. 10 paragraphs per question: 2 supporting (gold), 8 distractors (gray). The answer requires composing information from both gold paragraphs scattered among topically similar distractors.

We pretrained DCA (89M params) and a parameter-matched baseline (90M params) on WikiText-103[^wikitext] for 50K steps, then finetuned both on HotpotQA across three seeds. Though DCA is modestly worse on WikiText-103 validation perplexity (21.48 vs 20.79, ~3%), the benefit on long reasoning is asymmetric. DCA achieves 5.4x higher exact match on HotpotQA (1.56% vs 0.29%, Table 2), with p < 10^-6 and odds ratio 5.49 (Fisher exact, pooled across seeds). DCA outperformed every baseline variant we tested, across both 50K and 30K pretrain budgets (Appendix H).

### Scaling to PG-19 and architectural exploration

The 90M result raises a natural question: does the advantage hold at larger scale? While the relative advantage is clear, absolute performance of both models is low (1.56% and 0.29% EM). WT103 is too small for 350M-class models, so we switched to PG-19[^pg19] (3B tokens) following standard conventions.

To calibrate the effect of pretraining domain, we also trained DCA 90M on PG-19 (EM=0.38%, compared to 1.56% on WT103). The 350M standard baseline on PG-19 achieves 0.93% EM, indicating that even at 4x the parameters, standard decoders remain poor at multi-hop QA. A FLOP-comparable DCA-215M on PG-19 achieves 1.43% EM vs the baseline's 0.93% (1.54x), with 39% fewer parameters and less VRAM (~35 vs ~45 GB). Within PG-19, scaling DCA from 90M to 215M improves EM from 0.38% to 1.43%, surpassing the 350M baseline by 1.54x.

**Table 2: HotpotQA results across scales and pretraining domains.** All models finetuned and evaluated on HotpotQA.

**WT103 pretraining (90M):**

| Model | Params | FLOP ratio | VRAM | EM% | F1% |
|---|---|---|---|---|---|
| Baseline 90M | 90M | ~1.0x | ~4 GB | 0.29 | 7.77 |
| DCA 90M | 89M | ~1.56x | ~8 GB | 1.56 | 14.40 |
{: .data-table}

**PG-19 pretraining (up to 350M):**

| Model | Params | FLOP ratio | VRAM | EM% | F1% |
|---|---|---|---|---|---|
| DCA 90M | 89M | ~1.56x | ~8 GB | 0.38 | 7.78 |
| Baseline 350M | 355M | 1.0x | ~45 GB | 0.93 | 10.81 |
| DCA-215M | 215M | 1.24x | ~35 GB | 1.43 | 11.32 |
{: .data-table}

### Architectural exploration

Scaling provided an opportunity to test which components of DCA are essential. Our 90M baseline uses per-head window assignment (heads 0-2 at w=32, heads 3-5 at w=128, heads 6-7 full causal), achieving the best perplexity among baseline variants (20.79) but only 0.29% EM on HotpotQA. A factorial experiment confirmed parallel streams are the primary mechanism; multi-scale windows are secondary. Two other architectural properties proved essential. Shared QKV weights collapse perspective diversity (cosine similarity >0.9 vs 0.2-0.4 with separate weights). Consensus at every layer (k=1) drops EM to 1.05% vs 1.59% with consensus every 6 layers (k=6).

These results motivated the DCA-215M design. A narrower variant at d=768 (323M params) achieved only EM=0.57%, undertrained at 6.2 tokens per parameter. A variant without per-perspective MLP (302M params, 1.07x FLOPs) achieved EM=1.21% (1.30x over baseline). Separate MLP weights (139M params, same FLOPs) achieved PPL=20.45 but EM=0.80%, confirming the shared MLP acts as a regularizer.

**Table 3: Architectural variant results.** All evaluated on HotpotQA.

| Model | Params | d_model | d_lane | EM% | VRAM |
|---|---|---|---|---|---|
| DCA 90M (full-fat) | 89M | 512 | 512 | 1.56 | ~8 GB |
| DCA-d768 | 323M | 768 | 768 | 0.57 | ~45 GB |
| DCA-noMLP | 302M | 1024 | 768 | 1.21 | ~35 GB |
| DCA-215M (bottleneck) | 215M | 1024 | 512 | 1.43 | ~35 GB |
| DCA 90M (separate MLPs) | 139M | 512 | 512 | 0.80 | ~10 GB |
{: .data-table}

The DCA-215M results (Tables 2 and 3) confirm this design is practical and competitive.

### What is DCA well suited for?

We selected benchmarks to test where DCA should help, not to maximize wins. Few existing tasks isolate distributed-source composition while remaining tractable for sub-billion-parameter decoder-only models, so HotpotQA serves as the primary stress test, 2Wiki as secondary corroboration, and the remaining tasks as negative controls.

![Figure 3: Information topology](/assets/images/divergent-convergent-attention/information_topology_panel_b.svg)

> **Figure 3** Information topology. DCA helps when relevant content is distributed across independent documents (left). It does not help when information forms a single chain (right).

Sequential reasoning tasks (bAbI[^babi], Tree pathfinding[^treepath], PrOntoQA[^prontoqa], LEGO[^lego]) show no advantage; all facts lie in a single flat sequence and every attention scale sees the same chain. Single-source tasks (TriviaQA[^triviaqa], LAMBADA[^lambada], MQAR[^mqar]) show no advantage; all perspectives see the same content. Tasks beyond model capacity (MuSiQue[^musique]) show both models at floor. 2WikiMultiHopQA[^twowiki] provides weak corroboration (Soft EM p = 0.004, EM ns).

DCA helps when relevant information is distributed across structurally independent segments, what we refer to as the information topology of the input, and does not help when information forms a single chain or resides at a single location. Within HotpotQA, the advantage is uniform across bridge questions (sequential logic, OR=4.65) and comparison questions (parallel logic, OR=4.51), indicating that multi-document context, not reasoning pattern, is the key factor.

## Mechanistic Analysis

### Force-decode: the representation advantage

To separate representation quality from generation dynamics, we feed the context to both models and force-decode the gold answer tokens (teacher-forcing), recording each model's log-probability of the correct token at each position. For each of 6,359 validation examples, we compare which model assigns higher probability to the gold answer.

**Table 4: Force-decode results (90M, WT103).** Top: paired comparison across all 6,359 validation examples (Wilcoxon signed-rank p < 10^-300). Bottom: advantage by baseline difficulty quintile.

| Slice | DCA advantage | DCA win rate |
|---|---|---|
| Overall | +6.25 nats (~520x) | 97.8% (6,217/6,359) |
| 0-20% (easiest) | +1.84 nats | 96.5% |
| 20-40% | +3.20 nats | 96.2% |
| 40-60% | +4.75 nats | 96.5% |
| 60-80% | +7.25 nats | 99.6% |
| 80-100% (hardest) | +14.23 nats | 100% |
{: .data-table}

The representation advantage is near-universal: DCA produces better internal representations on 97.8% of all examples, not just the 1.6% where EM=1. The advantage correlates with baseline difficulty (r=-0.888): 7.7x larger on the hardest examples than the easiest (Table 4, Figure 4). The harder an example is for a standard transformer, the more DCA's multi-perspective consensus improves the representation.

![Figure 4: Force-decode advantage by baseline difficulty quintile](/assets/images/divergent-convergent-attention/fig5_difficulty_scaling.svg)

> **Figure 4** DCA 90M vs Baseline 90M (both WT103). DCA's representational advantage scales with example difficulty. On the hardest quintile (where the baseline assigns the lowest probability to the correct answer), DCA's advantage is +14.23 nats. On the easiest, +1.84 nats. r=-0.888.

Recent work on latent multi-hop reasoning finds that while bridge-entity recall scales smoothly with model size, the compositional second hop does not, suggesting composition is a structural bottleneck rather than a capacity problem[^yanglatent]. That work studies parametric knowledge recall; DCA's setting differs in that all relevant information is provided in context. Nevertheless, our force-decode result is consistent with the broader view that end-to-end exact match may understate the gradual development of multi-hop structure in model representations: at 90M, the correct answer is already encoded with substantially higher probability under the right architecture, even where end-to-end EM remains near floor. Confirming this connection would require targeted compositional probes, such as entity-recall scores and causal interventions on bridge entities, applied directly to DCA's internal representations.

### Same retrieval, better composition

We computed mean token recall and derived an approximate token precision from aggregate F1 and recall on generated predictions (90M, WT103, pooled across seeds 137 and 2024). Token Recall is essentially identical (~57.5% in both models). Token Precision, derived via P = F1·R / (2R - F1), shows the full advantage: ~8.2% vs ~4.2% (~2.0x). The advantage appears to come primarily from composition rather than token-level recall.

> **Figure 5** DCA 90M vs Baseline 90M (both WT103). Token Recall is essentially identical (~57.5%). Token Precision (derived from aggregate F1 and recall) shows the full advantage (~8.2% vs ~4.2%).

![Figure 5: Token Recall/Precision](/assets/images/divergent-convergent-attention/fig3_token_recall_precision.svg)

First-sentence extraction approximately decomposes this into two components. The advantage that survives extraction (~1.2-1.5x) reflects compositional integration at the representation level. The remaining multiplier (~2-3x) reflects generation coherence, scaling with answer length (3x at 1 token, 12.8x at 4+ tokens). These ranges are inferred from comparing first-sentence and full-output EM ratios, not independently measured.

### Gate ablation: consensus is essential and precisely tuned

We force the consensus gate to fixed values during full QA evaluation using forward hooks. Gate=0 clamps the sigmoid to 0.001 (bypass consensus). Gate=1 clamps to 0.999 (force full consensus).

**Table 5: Gate ablation and learned gate values.** Top: forcing gate to fixed values during QA evaluation. Bottom: learned gate values at consensus layers.

| Condition | 90M | 215M |
|---|---|---|
| Learned gates | 101 | 91 |
| Gate=1 (force full) | 18 | 0 |
| Baseline | 12 | 59 |
| Gate=0 (bypass) | 2 | 2 |
{: .data-table}

| Consensus layer | 90M gate | 215M gate |
|---|---|---|
| Layer 5 (1st) | 0.29 | 0.31 |
| Layer 11 (final @ 90M) | 0.99 | 0.21 |
| Layer 17 (3rd) | -- | 0.28 |
| Layer 23 (4th) | -- | 0.35 |
| Layer 29 (final) | -- | 0.37 |
{: .data-table}

Bypassing consensus collapses performance from 101 to 2 correct at 90M and 91 to 2 at 215M (Table 5). Forcing full consensus drops 101 to 18 at 90M and 91 to 0 at 215M, worse than the 350M baseline (59 correct), indicating that forced consensus is actively destructive to the representations DCA has learned to build through gradual integration.

At 90M the learned strategy is binary: passthrough early (0.29), full commit at the final layer (0.99). At 215M it is gradual and never exceeds 0.37. Both strategies are load-bearing, and disrupting either destroys performance.

### Perspective divergence and attention patterns

Perspectives develop genuinely distinct representations (cosine similarity 0.21 between local and medium at layer 5), complementary rather than redundant (Appendix Table 8).

> **Figure 6** DCA 90M vs Baseline 90M (both WT103), consensus layer 5. Each DCA perspective specializes at a different scale, while the baseline compromises at 0.34.

![Figure 6: Cross-document attention fraction](/assets/images/divergent-convergent-attention/fig4_cross_doc_attention.svg)

Attention measurements (computed on EM=1 examples, n=101) confirm the specialization. The local perspective keeps 96% of attention within paragraphs (cross-document fraction 0.04), while the global perspective distributes 68% across documents. The baseline sits at 0.34. With DCA, local perspectives extract precise within-document content, global perspectives maintain cross-document context, and consensus integrates both. The baseline attends at multiple scales within a single residual stream, but must reconcile those scales within one shared representation.

## Methods

### Model architecture

All models use the GPT-2 architecture as a base[^radford]: learned positional embeddings, RMSNorm[^rmsnorm], GELU activations, no biases, and a weight-tied LM head.

**Standard transformer baseline.** At 90M: 12 layers, d=632, 8 heads (head dim=79), 90M parameters. At 350M: 24 layers, d=1024, 16 heads (head dim=64), 355M parameters. The 90M baseline (`baseline_mixed`) uses per-head window assignment (heads 0-2 at w=32, heads 3-5 at w=128, heads 6-7 full causal), which achieves the best perplexity among the baselines we tested (20.79). Additional baseline variants are compared in Appendix H.

**DCA.** At 90M: 12 layers, d=512, 8 heads (head dim=64), K=3 perspectives with horizons [32, 128, 0], consensus every=6, 89M parameters. At 215M: 30 layers, d_model=1024, d_lane=512 (lane width ratio=0.5), 8 heads, same horizons and consensus frequency, `memory_efficient=True` (per-perspective gradient checkpointing), 215M parameters. All DCA models use shared MLP (one module, called K times per layer), gated consensus policy, and `gate_lr_mult=10.0`.

Weight initialization uses std=0.02 for all parameters. Baseline TransformerBlocks use depth-scaled residual projections following GPT-2/GPT-3 convention. DCA blocks use flat initialization at all layers. Depth-scaled initialization was empirically shown to collapse DCA perspective diversity (cosine similarity 0.999 by step 5).

### Pretraining

**WikiText-103 (90M models).** 103M tokens of Wikipedia text. Tokenized with tiktoken GPT-2 BPE (vocab=50,257). Both DCA and baseline were trained for 50K steps with lr=6e-4, cosine decay to 10% of peak, warmup=2000 steps, microbatch=8 per device, grad accum=2, effective batch=16, seq len=1024, dropout=0.1, AdamW (beta1=0.9, beta2=0.95), weight decay=0.1, grad clip=1.0, seed=1337.

**PG-19 (215M and 350M models).** 3.07B tokens from Project Gutenberg books[^pg19]. Same tokenizer. The 350M baseline used 27K steps, lr=3e-4, microbatch=32 per device, grad accum=4, effective batch=128, dropout=0.0. DCA-215M used 21,700 steps (FLOP-matched; see Appendix F), with the same lr, batch, and dropout. Seed=1337.

**FLOP matching (DCA-215M vs 350M baseline).** We match total training FLOPs including gradient-checkpointing recompute overhead. Checkpointed regions (K=3 perspective attention+MLP) incur 4x forward FLOPs (1x forward, 1x recompute, 2x backward). Non-checkpointed regions (embedding, bottleneck projections) incur standard 3x. DCA-215M total training FLOPs per token: 2,636.6M. Baseline: 2,121M. Ratio: 1.243x. FLOP-matched steps: 27,000 / 1.243 = 21,700. Verified by an independent FLOP auditor to <1% error.

### HotpotQA finetuning

All models were finetuned on HotpotQA distractor setting[^hotpotqa]. The dataset contains 77,968 training examples and 6,359 validation examples. Each example contains 10 titled paragraphs (2 supporting, 8 distractors) concatenated with a question. Answers are short extractive spans.

We used a fixed protocol: identical finetuning recipe for all models, varying only the pretrained checkpoint. lr=5e-5, 3 epochs (14,619 steps), microbatch=4 per device, grad accum=4, effective batch=16, warmup=200 steps, cosine decay, masked loss (only answer tokens contribute to loss). For headline comparisons, we report results across seeds 42, 137, and 2024.

### Evaluation

**Exact Match (EM) and Token F1.** Computed with SQuAD-standard normalization[^rajpurkar]: lowercase, remove articles and punctuation, collapse whitespace. Decoding is greedy (`temperature=0`, deterministic).

**Force-decode.** For each validation example, we feed the context and teacher-force the gold answer tokens, recording per-token log-probabilities under both models. We compare mean log-probability per example in a paired setup. Statistical test: Wilcoxon signed-rank (non-parametric, no distributional assumptions). For difficulty-quintile analysis, examples are binned by baseline log-probability.

**Gate ablation.** Forward hooks on consensus gate layers clamp sigmoid output to 0.001 (`gate=0`, bypass consensus) or 0.999 (`gate=1`, force full consensus) during full greedy QA evaluation. Results are deterministic given the checkpoint.

**Token Recall and Token Precision.** These are the recall and precision components of SQuAD-standard token F1[^rajpurkar] (bag-of-words with counts, after normalization). Token Recall is reported as mean per-example token recall across pooled seeds 137 and 2024. Token Precision is derived from aggregate mean F1 and mean recall via P = F1 * R / (2R - F1). This is an approximate decomposition and is reported with approximate notation in the main text.

### Reproducibility

Software: PyTorch 2.9.0, CUDA 12.8, tiktoken, Python 3.13. All random seeds are fixed and reported. Greedy decoding is deterministic. Training uses fixed seeds for data shuffling, weight initialization, and dropout. Code for the DCA block will be released as a standalone module.

## LLM Disclosure

This work used Claude (Opus 4.5 and 4.6, Anthropic) as a research assistant for generating figures and plots, helping to write and evaluate benchmark scripts, and editing the manuscript. All experimental results were verified independently against saved artifacts. All scientific claims, experimental design decisions, and interpretations are the authors' own.

## Conclusion

Multi-document composition is a documented bottleneck for production LLMs. RAG pipelines retrieve relevant documents but fail to synthesize across them[^rag]. Models fail to use information in the middle of long contexts[^lostmiddle]. Multi-hop reasoning may require 30-70B parameters to emerge in standard transformers[^steelekatz]. With DCA, we sought to demonstrate that parallel multi-scale perspectives with periodic late consensus can improve on these deficiencies.

Despite the limited capacity of our models, extensive benchmarking demonstrated a consistent advantage on distributed-source tasks (5.4x EM at 90M, 1.54x at 215M FLOP-comparable) and no advantage on sequential, single-source, or capacity-limited tasks. At 90M, the resulting representations encode multi-document relationships better than a parameter-matched standard transformer on 97.8% of examples, with 7.7x larger gains on the hardest examples, reflecting an advantage in composition rather than retrieval.

Consensus frequency, horizon widths, and gate training dynamics were fixed throughout our experiments, leaving substantial room for task-specific tuning. Because DCA is fundamentally a primitive that operates on tensors, other sequence-processing modules (ring attention, linear attention, SSMs) could serve as perspectives, opening a combinatorial design space we have only begun to explore. The force-decode diagnostic is itself useful beyond DCA, offering a general tool for determining whether the bottleneck in a given architecture is understanding or expression. We will be sharing the base code for DCA on GitHub.

---

## Appendix

### HotpotQA task illustration

Figure 2 above illustrates the HotpotQA distractor setting used throughout the paper: 10 paragraphs per question, with 2 supporting paragraphs embedded among 8 distractors.

### Multi-seed raw data

**Table 6: Per-seed HotpotQA results (90M, WT103).** DCA mean EM: 1.562% (std 0.120%). Baseline mean EM: 0.288% (std 0.122%). Fisher exact (pooled): OR=5.49, p < 10^-6.

| Model | Seed | EM | F1 | n |
|---|---|---|---|---|
| DCA 90M | 42 | 0.01588 | 0.14508 | 6359 |
| DCA 90M | 137 | 0.01431 | 0.14439 | 6359 |
| DCA 90M | 2024 | 0.01667 | 0.14265 | 6359 |
| Baseline 90M | 42 | 0.00189 | 0.07625 | 6359 |
| Baseline 90M | 137 | 0.00425 | 0.08143 | 6359 |
| Baseline 90M | 2024 | 0.00252 | 0.07528 | 6359 |
{: .data-table}

### Additional benchmark evaluations

All results in this section come from 90M models pretrained on WT103.

Sequential reasoning tasks show no DCA advantage: bAbI 2-hop hits 100% for both models at all distractor counts, and PrOntoQA also hits 100% at all hop counts. Tree pathfinding favors the baseline by 2-7 points at depths 4-6. LEGO is roughly even, with the baseline at ~31% and DCA at ~30%.

Single-source tasks also show no advantage. TriviaQA and LAMBADA show no consistent lift. MQAR (fixed protocol, vocab=8192) remains at exact chance across all key-value counts and learning rates for both models.

Capacity-limited tasks stay at floor. MuSiQue shows DCA at 0.21% and baseline at 0.10% (p = 0.687, not significant). Additional synthetic compositional probes were similarly uninformative at this scale: Entity Comparison remained at chance (50%, loss near ln 2), and MQAR2 also remained at chance (50%). We treat these as floor-effect results for small models trained from scratch rather than meaningful tests of DCA's inductive bias.

2WikiMultiHopQA provides weak corroboration: EM is even (0.31% vs 0.33%, p = 1.0), but Soft EM (F1 >= 0.5) favors DCA at 2.31% vs 1.47% (p = 0.004).

### HotpotQA stratifications

Bridge vs comparison is nearly uniform: bridge OR=4.65, comparison OR=4.51. Multi-document context, not reasoning pattern, is the key factor.

Error taxonomy (chi-squared p < 10^-6) shows DCA shifting away from vague overlap (60.6% vs 65.9%) and hallucination (25.0% vs 28.3%) toward partial match (5.1% vs 1.4%) and wrong extraction (3.2% vs 0.4%).

Answer length scaling is substantial: 3.0x at 1 token, 7.1x at 2 tokens, and 12.8x at 4+ tokens.

Single-support filtering shows that 75% of examples need one paragraph. DCA is stronger on single-support examples (OR=4.95) than multi-support examples (OR=3.02).

### Force-decode difficulty scaling

![Appendix Figure: Force-decode advantage vs baseline difficulty](/assets/images/divergent-convergent-attention/fig5_scatter_reference.png)

> **Appendix Figure** Per-example force-decode advantage (DCA log-prob minus baseline log-prob) plotted against baseline log-prob (90M DCA vs 90M baseline, both WT103). Each point is one of 6,359 HotpotQA validation examples. r=-0.888. The harder an example is for the baseline (more negative log-prob), the larger DCA's representational advantage.

---

### Literature gap

To our knowledge, no published decoder-only HotpotQA results exist between 90M and 7B parameters.

**Table 7: Published HotpotQA results.** Encoder models dominate at 110M-355M due to bidirectional attention and span extraction heads. Decoder-only models need 7B+ for ~30% EM.

| Architecture | Params | HotpotQA | Notes |
|---|---|---|---|
| DCA 90M | 89M | 1.56% EM | Decoder, WT103 |
| Baseline 90M | 90M | 0.29% EM | Decoder, WT103 |
| Baseline 350M | 355M | 0.93% EM | Decoder, PG-19 |
| DCA-215M | 215M | 1.43% EM | Decoder, PG-19 |
| BERT-base-era systems | ~110M | ~54% EM | Encoder, bidirectional |
| Longformer-base | ~149M | 64% F1 | Encoder, local+global |
| Longformer-large | ~435M | 73% F1 | Encoder, local+global |
| BigBird-ETC | ~131M | 76% F1 | Encoder, sparse |
| RoBERTa-large-based systems | ~355M | ~70% EM | Encoder |
| Llama-2-7B | 7B | ~30% EM | Decoder (FireAct) |
| GPT-3.5 | proprietary | ~31% EM | Few-shot ReAct |
| Human | -- | ~91% F1 | Leaderboard |
{: .data-table}

Encoder models such as BERT[^bert], Longformer[^longformer], BigBird-ETC[^bigbird], and RoBERTa[^roberta] dominate at 110M-355M because HotpotQA was designed for BERT-era extractive QA with bidirectional attention and span extraction heads. Decoder-only models need 7B+ for ~30% EM (FireAct with Llama-2-7B[^fireact]). Steele & Katz[^steelekatz] identify a phase transition at 30-70B for emergent multi-hop reasoning.

### FLOP methodology

All calculations count linear-layer operations (2 x d_in x d_out per layer, per token). Attention-matrix computation is excluded as comparable across models at the same sequence length.

90M DCA: 12 layers x K=3 x (8d^2 + 16d^2) at d=512 ~= 279M FLOPs/token. Total training ~= 683 TFLOPs (50K steps, effective batch=16).

350M baseline: 24 layers x 24d^2 at d=1024 plus LM head ~= 707M FLOPs/token. Total ~= 7,507 TFLOPs (27K steps, effective batch=128).

DCA-215M: 30 layers x K=3 x 24d^2 at d_lane=512 plus projections and LM head ~= 690M FLOPs/token. Training ratio with gradient-checkpointing recompute: 2,636.6M / 2,121M = 1.243x. FLOP-matched steps: 21,700. Auditor-verified to <1% error.

### Architectural variant details

**DCA-d768:** 323M params, 1.77x FLOPs, FLOP-matched to 15,272 steps. EM=0.57%. Undertrained at 6.2 tokens/parameter. This motivated the bottleneck approach.

**DCA-noMLP:** Removed per-perspective MLP so the MLP runs once on the consensus output. 302M params, 1.07x FLOPs. EM=1.21% (1.30x). Per-perspective MLP matters.

**DCA 90M with separate MLPs:** Same FLOPs, +56% params (139M). PPL=20.45, beating the baseline's 20.79, but EM=0.80%. Shared MLP acts as a regularizer.

**DCA-215M (bottleneck):** 215M params, 1.24x FLOPs, ~35 GB VRAM. EM=1.43% (1.54x). Gate pattern is gradual (0.21-0.37 vs 0.99 at 90M).

### Additional WT103 variants

In addition to the headline comparison (DCA vs `baseline_mixed`, 50K steps, 3 seeds), we trained six additional 90M WT103 variants: three DCA variants (consensus every 1, 3, or 6 layers, plus uniform-horizon settings) and three baseline variants (full causal, layerwise windows, sliding window w=256) at 50K or 30K steps. In all cases, every DCA variant outperformed every baseline on HotpotQA EM, including cross-budget comparisons where DCA at 30K steps with 1-epoch finetuning exceeded baselines at 50K steps with 3-epoch finetuning. The factorial decomposition confirmed that parallel streams are the primary mechanism; multi-scale windows are secondary.

### Perspective divergence

Pairwise cosine similarity between K=3 perspectives at consensus layers, measured on HotpotQA inputs (90M, WT103). No EM stratification: correct and incorrect examples show nearly identical divergence, confirming divergence is an architectural property rather than a predictor of success.

**Table 8: Perspective divergence on QA data.** Local and medium are most dissimilar at layer 5 (0.21); by layer 11 they partially reconverge (0.62) while local-global remains distinct (0.34).

| Layer | Pair | Overall | EM=1 (n=101) | EM=0 (n=6,258) |
|---|---|---|---|---|
| 5 | local vs medium | 0.207 | 0.209 | 0.207 |
| 5 | local vs global | 0.435 | 0.437 | 0.435 |
| 5 | medium vs global | 0.405 | 0.407 | 0.405 |
| 11 | local vs medium | 0.621 | 0.625 | 0.621 |
| 11 | local vs global | 0.336 | 0.337 | 0.336 |
| 11 | medium vs global | 0.437 | 0.436 | 0.437 |
{: .data-table}

[^hotpotqa]: Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering". *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP 2018)*. arXiv:1809.09600

[^buzsaki]: Buzsaki, G. (2006). *Rhythms of the Brain*. Oxford University Press.

[^canolty]: Canolty, R. T., & Knight, R. T. (2010). "The Functional Role of Cross-Frequency Coupling". *Trends in Cognitive Sciences*, 14(11), 506-515.

[^lisman]: Lisman, J. E., & Jensen, O. (2013). "The Theta-Gamma Neural Code". *Neuron*, 77(6), 1002-1016.

[^colgin]: Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, T., Jensen, O., Moser, M.-B., & Moser, E. I. (2009). "Frequency of Gamma Oscillations Routes Flow of Information in the Hippocampus". *Nature*, 462, 353-357.

[^retnet]: Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2023). "Retentive Network: A Successor to Transformer for Large Language Models". arXiv:2307.08621

[^ringattention]: Liu, H., Zaharia, M., & Abbeel, P. (2023). "Ring Attention with Blockwise Transformers for Near-Infinite Context". arXiv:2310.01889

[^flashattention]: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Re, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". *Advances in Neural Information Processing Systems*, 35.

[^resnext]: Xie, S., Girshick, R., Dollar, P., Tu, Z., & He, K. (2017). "Aggregated Residual Transformations for Deep Neural Networks". *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017)*. arXiv:1611.05431

[^moe]: Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer". *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017)*. arXiv:1701.06538

[^switch]: Fedus, W., Zoph, B., & Shazeer, N. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". *Journal of Machine Learning Research*, 23(120), 1-39.

[^mvit]: Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., & Feichtenhofer, C. (2021). "Multiscale Vision Transformers". *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2021)*. arXiv:2104.11227

[^highway]: Srivastava, R. K., Greff, K., & Schmidhuber, J. (2015). "Highway Networks". arXiv:1505.00387

[^fedavg]: McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data". *Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017)*. arXiv:1602.05629

[^wikitext]: Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). "Pointer Sentinel Mixture Models". *Proceedings of the 5th International Conference on Learning Representations (ICLR 2017)*. arXiv:1609.07843

[^pg19]: Rae, J. W., Potapenko, A., Jayakumar, S. M., & Hillier, C. (2020). "Compressive Transformers for Long-Range Sequence Modelling". *Proceedings of the 8th International Conference on Learning Representations (ICLR 2020)*. arXiv:1911.05507

[^longformer]: Beltagy, I., Peters, M. E., & Cohan, A. (2020). "Longformer: The Long-Document Transformer". arXiv:2004.05150

[^bigbird]: Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., & Ahmed, A. (2020). "Big Bird: Transformers for Longer Sequences". *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. arXiv:2007.14062

[^babi]: Weston, J., Bordes, A., Chopra, S., & Mikolov, T. (2015). "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks". arXiv:1502.05698

[^treepath]: Brinkmann, J., Goswami, K., & Rajani, N. F. (2024). "A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task". *Findings of the Association for Computational Linguistics: ACL 2024*.

[^prontoqa]: Saparov, A., & He, H. (2023). "Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought". *Proceedings of the 11th International Conference on Learning Representations (ICLR 2023)*.

[^lego]: Zhang, Y., Yu, A. W., & Xu, W. (2022). "Unveiling Transformers with LEGO: A Synthetic Reasoning Task". arXiv:2206.04301

[^triviaqa]: Joshi, M., Choi, E., Weld, D. S., & Zettlemoyer, L. (2017). "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension". *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017)*. arXiv:1705.03551

[^lambada]: Paperno, D., Kruszewski, G., Lazaridou, A., Pham, N. Q., Bernardi, R., Pezzelle, S., Baroni, M., Boleda, G., & Fernandez, R. (2016). "The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context". *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)*. arXiv:1606.06031

[^mqar]: Arora, S., Eyuboglu, S., Timalsina, A., Johnson, I., Poli, M., Rudra, A., & Zou, J. (2023). "Zoology: Measuring and Improving Recall in Efficient Language Models". arXiv:2312.04927

[^musique]: Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). "MuSiQue: Multihop Questions via Single-hop Question Composition". *Transactions of the Association for Computational Linguistics*, 10, 539-554. arXiv:2108.00573

[^twowiki]: Ho, X., Nguyen, A.-K. D., Sugawara, S., & Aizawa, A. (2020). "Constructing a Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps". *Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)*. arXiv:2011.01060

[^bert]: Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2019)*. arXiv:1810.04805

[^roberta]: Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). "RoBERTa: A Robustly Optimized BERT Pretraining Approach". arXiv:1907.11692

[^yanglatent]: Yang, S., Gribovskaya, E., Kassner, N., Geva, M., & Riedel, S. (2024). "Do Large Language Models Latently Perform Multi-Hop Reasoning?" *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)*.

[^radford]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). "Language Models are Unsupervised Multitask Learners". OpenAI Blog.

[^rajpurkar]: Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). "SQuAD: 100,000+ Questions for Machine Comprehension of Text". *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016)*.

[^rmsnorm]: Zhang, B., & Sennrich, R. (2019). "Root Mean Square Layer Normalization". *Advances in Neural Information Processing Systems*, 32.

[^rag]: Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., Rocktaschel, T., Riedel, S., & Kiela, D. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. arXiv:2005.11401

[^lostmiddle]: Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). "Lost in the Middle: How Language Models Use Long Contexts". *Transactions of the Association for Computational Linguistics*, 12, 157-173. arXiv:2307.03172

[^steelekatz]: Steele, B., & Katz, M. (2026). "Scaling Trends for Multi-Hop Contextual Reasoning in Mid-Scale Language Models". arXiv:2601.04254

[^fireact]: Chen, B., Monajatipoor, M., Veen, D. V., Guo, Y., & Dubrawski, A. (2023). "FireAct: Toward Language Agent Fine-tuning". arXiv:2310.05915

## Citation

This blog post serves as the current preprint version of this work. Until an archival version is available, please cite it as:

```bibtex
@misc{zhao2026dca,
  author = {Ben Zhao and Jenhan Tao},
  title = {Divergent-Convergent Attention: Parallel Perspectives for Compositional Reasoning},
  year = {2026},
  howpublished = {\url{https://iluvatarlabs.github.io/blog/2026/03/divergent-convergent-attention/}},
  note = {Iluvatar Labs blog preprint}
}
```

### Acknowledgements

We thank Abel Chiao for helpful discussions and feedback on this work.

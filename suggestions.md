# Divergent-Convergent Attention Post Suggestions

## Recommended TL;DR

> **The TL;DR:** Divergent-Convergent Attention (DCA) improves multi-document compositional reasoning by maintaining multiple parallel attention perspectives before periodic learned consensus. On HotpotQA, DCA achieves **5.4x higher exact match** than a parameter-matched 90M baseline, and a **215M DCA model outperforms a 355M standard transformer by 1.54x** with fewer parameters and lower memory. The strongest result is representational: in force-decode analysis, DCA assigns higher probability to the correct answer tokens on **98% of examples**, with the advantage growing sharply on the hardest questions, suggesting that DCA improves how distributed evidence is internally composed before decoding.

## Shorter TL;DR Options

### Option A

> **The TL;DR:** DCA gives transformers parallel perspectives before fusion, and that changes what they represent. On HotpotQA, it delivers **5.4x higher exact match** at 90M, beats a **355M baseline with 39% fewer parameters** at 215M, and in force-decode assigns higher probability to the correct answer on **98% of examples**. The biggest gains appear on the hardest questions, pointing to a real advantage in **representing distributed evidence**, not just generating cleaner text.

### Option B

> **The TL;DR:** DCA helps transformers keep separate views of distributed evidence before merging them. The result is not just better answers, but better internal representations: DCA beats a parameter-matched baseline by **5.4x** on HotpotQA, beats a much larger 355M transformer at 215M, and wins force-decode on **98% of examples**, with the largest gains on the hardest cases.

### Option C

> **The TL;DR:** DCA improves how transformers internally compose evidence across documents. On HotpotQA, it yields **5.4x higher exact match** at 90M, outperforms a **355M standard transformer with a 215M model**, and in force-decode assigns higher probability to the gold answer on **98% of examples**. The representational advantage is strongest exactly where the baseline struggles most.

## Narrative Outline

### Recommended story

1. Problem
   Standard decoder-only transformers prematurely fuse evidence from structurally independent sources.

2. Hypothesis
   Multi-document composition benefits from keeping multiple perspectives separate before consensus.

3. Mechanism
   DCA implements that hypothesis with parallel horizons and periodic learned consensus.

4. Main empirical result
   DCA improves HotpotQA generation metrics at two scales, including against a larger baseline.

5. Core scientific result
   Force-decode shows that DCA represents the correct answer better on 98% of examples.

6. Strongest validation
   The representational advantage grows as baseline difficulty increases.

7. Scope condition
   The gain appears on distributed-source tasks, not on single-source retrieval or flat sequential reasoning.

8. Mechanistic support
   Gate ablations and perspective-divergence analyses align with the delayed-consensus hypothesis.

9. Conclusion
   The architectural benefit is representational first, generative second.

### Why this story is stronger

- It leads with the architecture’s claimed mechanism rather than just a benchmark delta.
- It uses EM as evidence, but not as the sole reason to believe the claim.
- It makes the difficulty-stratified force-decode result the central validation of the hypothesis.
- It keeps the task-selectivity results as a boundary condition, which makes the claim feel sharper and less overgeneralized.

## Proposed Structural Options

### Option 1: Language-first, minimal reorder

Keep the manuscript order mostly intact and change emphasis.

- Replace `Abstract` with a TL;DR that leads on representation.
- In the Introduction, explicitly tell the reader that the main evidence is force-decode, not just EM.
- After the first HotpotQA result, add a transition noting that low absolute EM motivates direct representation analysis.
- Rewrite the opening of Section 4 and the Discussion to treat force-decode as the core result.

This is the lowest-risk option. It preserves the current flow and mostly changes framing.

### Option 2: Moderate reorder

Move the representation story earlier conceptually, without rewriting the whole paper.

- Introduction
- Architecture
- Benchmarks
- Short bridge: why generation metrics are insufficient at this scale
- Force-decode / difficulty analysis
- Task-structure analysis
- Gate and divergence analysis
- Discussion

This still keeps the same material, but makes Section 4 feel like the payoff rather than a side analysis.

### Option 3: Stronger reorder

Make the paper explicitly about representational advantage under distributed evidence.

- Introduction
- Architecture
- Main result: force-decode and difficulty scaling
- Generation benchmarks
- Task-structure boundary conditions
- Mechanistic ablations
- Discussion

This is the clearest scientific narrative, but it is also the biggest intervention and changes the paper’s current rhythm.

## Recommendation

I would use Option 1 first.

- The current manuscript already has the right ingredients.
- The main issue is claim hierarchy, not missing content.
- You can make the force-decode result feel like the center of gravity without restructuring the paper aggressively.

If you want to push harder later, Option 2 is the best next step. I would not jump to Option 3 unless you want a more substantial rewrite.

## Suggested Emphasis Changes

### Introduction

Add one sentence that makes the core claim explicit:

> The central evidence for this claim is not only higher generation accuracy, but a force-decode analysis showing that DCA assigns higher probability to the correct answer on 98% of examples, with the largest gains on the questions hardest for a standard transformer.

### After the first HotpotQA result

Add one bridge sentence:

> Because both models remain in a low absolute-performance regime, we do not treat exact match alone as sufficient evidence; we therefore analyze the learned representations directly in Section 4.

### Section 4 framing

Current framing is accurate but too modest. The section should explicitly read as the paper’s main validation, not just a mechanistic supplement.

### Discussion opener

Lead with:

> While the absolute HotpotQA scores remain low at this scale, the core finding is representational rather than purely generative: DCA assigns higher probability to the correct answer tokens on 98% of examples, and the gain is largest on the examples hardest for a standard transformer.


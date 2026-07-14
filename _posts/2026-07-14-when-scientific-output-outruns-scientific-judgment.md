---
layout: post
title: "When Scientific Output Outruns Scientific Judgment"
date: 2026-07-14
category: Research
thumbnail: marvin
author: "Iluvatar Labs"
excerpt: "AI is scaling research output but risks creating an unsustainable wave of slop. Marvin is the industry-leading scientific agent that combats this quality gap by pairing frontier models with expert-built workflows, provenance, and persistent project state, scoring 93.7% across four LifeSciBench expert-reasoning tasks."
image: /assets/images/vibe-science/social-card.png
---

A recent *Science* article[^trinetx] on TriNetX documents what can happen when scientific output becomes easier to produce than to verify. The platform gives researchers access to anonymized electronic health records from more than 300 million patients. Publications mentioning TriNetX rose from 33 five years earlier to nearly 2,700 in 2025, then exceeded 2,100 by late June 2026. The article describes uncorrected bias, cherry-picking, and methods sections that claim analyses the platform cannot perform. Researchers also asked seven LLMs how to make a key correction for immortal-time bias in TriNetX; six proposed methods that were impossible to implement, and the researchers then found those impossible approaches described in published papers.

## AI is scaling output, not quality

The problem is not TriNetX itself but the combination of low-friction tooling, incentives that reward output, and too little scientific judgment between an analysis and a paper. That broader failure mode is what we mean by vibe science: *scientific-looking* work whose surface plausibility exceeds its evidentiary grounding. 

This might manifest as a literature review that misses the papers that matter, a hypothesis that cannot be operationalized, an analysis built on inappropriate assumptions, or a protocol missing the control that gives the experiment meaning. In many ways, vibe science is more dangerous than vibe coding because its errors are slower to detect and harder to contain. Software often offers relatively fast feedback meanwhile science rarely offers clean yes-or-no outcomes on these short timelines. AI is scaling this failure mode faster than ever, which makes scientific intuition and judgment increasingly indispensable. 

Meanwhile, because science is cumulative, low-quality slop of this kind has the potential to become foundational for future work. As scientists, we stand on the shoulders of giants and add small pieces to a shared knowledge corpus, so when one piece is unreliable, the weakness propagates. A striking example of this is in Alzheimer's research, where a 2006 *Nature* paper linking an amyloid-beta assembly to memory impairment drew more than 2,500 citations before it was retracted in 2024 over manipulated figures and unverifiable data. The cost? From 1995 to 2021, an $42.5 billion in private R&D with near universal focus on amyloid plaques was spent with a 95% failure rate, while the disease to this day remains largely untreatable.[^adcost]

That is not to say that AI has no future in science. In fact, quite the contrary, we believe it can be both beneficial and inevitable. However, while LLMs remain prone to hallucinations, poor synthesis of diverse or conflicting data, and limited long-horizon reasoning across real projects with real unknowns, it is essential that any scientific agent demonstrate the rigorous controls and transparency that science demands. Generating a plausible hypothesis is easier than deciding whether it is worth pursuing and what experiments should come next. Performing an analysis is easier than correctly interpreting it. 

## How Marvin closes the quality gap

This is why we've built Marvin the way we did. Marvin's workflows and skills library were built by and vetted by KOLs and scientists with decades of experience running and leading research labs in industry and academia. Across the system, Marvin preserves provenance and project state, supports reproducibility, and enables a thoughtful balance between full autonomy and keeping humans over the loop for the parts of science that benefit from judgment.

To demonstrate Marvin's world-class scientific reasoning, we evaluated several agentic platforms on the five publicly released questions from OpenAI's recently announced LifeSciBench[^lifescibench], four expert-reasoning tasks and a Visium spatial-transcriptomics analysis task. LifeSciBench is an exciting new benchmark that represents a more challenging benchmark for biology by moving beyond simple recall towards more realistic scientific workflows, as benchmarks like BixBench are now saturated at the frontier[^gpt55]. When driven by Claude Opus 4.8, Marvin averaged **93.7%** across the four expert-reasoning questions, compared with **86.1%** for the next-best system and **84.0%** for single-shot Claude Code. While both Biomni A1 and Claude Science were slightly better than Claude Code (**~2%**), Marvin achieved more than **4.5x** the performance differential (**9.7%**). 

![Horizontal bar chart showing Marvin leading four comparison systems on the pooled LifeSciBench expert-reasoning score.](/assets/images/vibe-science/lifescibench-expert-reasoning-opus.svg)

> **Figure 1** Pooled LifeSciBench score across four recently released expert-reasoning questions, using Claude Opus 4.8 as the underlying model.

On the spatial transcriptomics task, Marvin's (**81.1%** vs next best at **47.2%** and baseline at **43.4%)** performance gap is even higher, reflecting the quality and task-specificity of Marvin's curated skills library. Notably, these results were consistent and durable regardless of the underlying model, with similar deltas in performance seen when driven by GPT-5.5 and GLM-5.2 as well.

![Horizontal bar chart showing Marvin leading four comparison systems on the LifeSciBench D1 Visium analysis task.](/assets/images/vibe-science/lifescibench-d1-opus.svg)

> **Figure 2** LifeSciBench D1 score for a Visium spatial-transcriptomics analysis task, using Claude Opus 4.8 as the underlying model.

Curiously, a popular open-source alternative actually performed worse than the single-shot coding-agent baseline on average, scoring **68.0%** versus Claude Code's **84.0%** across the four reasoning questions and **24.5%** versus **43.4%** on the visual task across all models. In our view, this highlights exactly the risk that comes with vibe science. Something can look like the real deal and produce outputs that seem believable enough, but if it doesn't hold up under scrutiny, it's just creating more slop overloading peer review. 

To be clear, we are huge fans of open source, so this criticism isn't an attack on OSS, just a reminder that agentic science does benefit from genuine scientific expertise (and to this end, we plan to contribute a selected set of life science skills from Marvin's library shortly to the public domain.) We also are not trying to gate-keep science! On the contrary, that's exactly why we built Marvin and launched open science initiatives like IORI: so everyone can access and benefit from advances in AI and contribute to unmet scientific problems. We just believe that when it comes research, rigor is non-negotiable. The bottom line is that getting science right is hard, and frankly, it should be hard because satisfying a high standard for evidentiary proof is why the scientific method works.

## Supplemental

![Grouped horizontal bars comparing Marvin, the coding baseline, and an open-source alternative on pooled expert-reasoning scores with GPT-5.5 and GLM-5.2.](/assets/images/vibe-science/lifescibench-expert-reasoning-supplement.svg)

> **Figure 3** Supplemental pooled LifeSciBench score across the four expert-reasoning questions, using GPT-5.5 and GLM-5.2 as the underlying models.

![Grouped horizontal bars comparing Marvin, the coding baseline, and an open-source alternative on D1 with GPT-5.5 and GLM-5.2.](/assets/images/vibe-science/lifescibench-d1-supplement.svg)

> **Figure 4** Supplemental LifeSciBench D1 score, using GPT-5.5 and GLM-5.2 as the underlying models.

[^trinetx]: *Science*. (2026). ["Medical students are using a popular research tool to pump out misleading studies."](https://www.science.org/content/article/medical-students-are-using-popular-research-tool-pump-out-misleading-studies)

[^bixbench]: Mitchener, L. et al. (2025). ["BixBench: A Comprehensive Benchmark for LLM-Based Agents in Computational Biology."](https://arxiv.org/abs/2503.00096) arXiv:2503.00096.

[^gpt55]: OpenAI. (2026). ["Introducing GPT-5.5."](https://openai.com/index/introducing-gpt-5-5/)

[^lifescibench]: OpenAI. (2026). ["Introducing LifeSciBench."](https://openai.com/index/introducing-life-sci-bench/)

[^adcost]: Cummings, J. L., Goldman, D. P., Simmons-Stern, N. R., & Ponton, E. (2022). ["The Costs of Developing Treatments for Alzheimer's Disease: A Retrospective Exploration."](https://doi.org/10.1002/alz.12450) *Alzheimer's & Dementia*, 18(3), 469-477.

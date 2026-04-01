---
layout: product
title: "Meet Marvin: the Autonomous ML Research Agent"
date: 2026-03-01
category: Product
author: "Iluvatar Labs"
product_name: "Marvin"
product_subtitle: "The fully <u>autonomous</u> scientist agent."
product_tagline: "Define your research mission and Marvin will take care of the rest. Marvin automates the workflow end-to-end: reviewing literature, generating truly novel hypotheses, performing experiments, and rigorous analysis."
product_cta_text: "Read Marvin's latest research ↓"
product_cta_href: "#marvins-work"
hero_offset: 60
product_logo: '<svg width="130" height="90" viewBox="0 0 64 44" fill="none" xmlns="http://www.w3.org/2000/svg"><defs><mask id="visor-eyes"><rect x="0" y="0" width="64" height="44" fill="white"/><circle cx="22" cy="22" r="6.5" fill="black"/><circle cx="42" cy="22" r="6.5" fill="black"/></mask></defs><path d="M 14 2 L 8 2 Q 4 2 4 6 L 4 16 Q 4 22 0 22 Q 4 22 4 28 L 4 38 Q 4 42 8 42 L 14 42" stroke="#fff" stroke-width="2" stroke-linecap="round" fill="none"/><path d="M 50 2 L 56 2 Q 60 2 60 6 L 60 16 Q 60 22 64 22 Q 60 22 60 28 L 60 38 Q 60 42 56 42 L 50 42" stroke="#fff" stroke-width="2" stroke-linecap="round" fill="none"/><rect x="10" y="12" width="44" height="20" rx="10" fill="#fff" mask="url(#visor-eyes)"/></svg>'
backdrop: "constellation"
has_radar: true
has_knowledge_graph: true
excerpt: "Autonomous ML research agent that conducts rigorous scientific research end-to-end: literature review, hypothesis generation, experiment design, execution, analysis, and documentation."
---

## Why we built this

The bottleneck in ML research today isn't compute or data. It's the preparation. More research is being produced now than at any point in history, and the pace is only increasing. Researchers must ingest and synthesize growing volumes of information before they can actually start their research. And even once they start, a lot of the research cycle is still spent on logistics rather than the science itself.

We built Marvin because nothing out there worked well enough for our own research. The existing options were either too dumb (chasing red herrings down rabbit holes or proposing smart-sounding ideas that were anything but), too wasteful (channeling Ralph Wiggum on experiments that were never going to work), or too opaque (poor documentation, no reasoning traces or "logic trail" that forms the bedrock of scientific reproducibility.)

Marvin takes the information overload and busywork out of research. It does deep research, generates and tests novel and plausible hypotheses (with the help of **[Vera](#vera)**, our Visionary Exploration Research Agent), and documents every actionable step with a full logic trail with minimal to no human supervision. Context is kept fresh and up-to-date between all sessions and agents with our custom memory management system, **[Silmaril](#lorien)**.

## The research cycle, automated

<div class="timeline">
    <div class="timeline-line"></div>
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-title">Search</div>
        <div class="timeline-desc">Retrieve and cross-check across relevant scientific databases, prior art, and published methodologies.</div>
    </div>
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-title">Hypothesize</div>
        <div class="timeline-desc">Marvin synthesizes literature with experimental findings (and the help of special agent <strong>Vera</strong>) to generate truly novel but plausible hypotheses.</div>
    </div>
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-title">Design</div>
        <div class="timeline-desc">Experiment in batches with scenario trees by asking the right questions.</div>
    </div>
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-title">Execute</div>
        <div class="timeline-desc">Agents build, audit, review, and execute the plan. Scalable from local, hybrid, and cloud compute infrastructure.</div>
    </div>
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-title">Analyze</div>
        <div class="timeline-desc">Marvin analyzes the entire data corpus to extract actionable insights with the rigor expected of a PhD-level researcher.</div>
    </div>
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-title">Document</div>
        <div class="timeline-desc">Track not just the paper trail, but the <i>logic trail</i>. Every iteration, every interpretation, every decision.</div>
    </div>
</div>

## Dream big with Vera

**Vera** is Marvin's hypothesis engine: a tunable exploration agent that generates truly novel ideas and insights that conventional agents miss. Vera can operate as part of Marvin's standard research loop, dream independently while Marvin waits on experimental results, or be invoked manually when you need a brainstorm partner. Every hypothesis is grounded in evidence and scored for novelty, plausibility, tractability, impact, and scope, so Vera won't waste your time on hallucinatory nonsense that sounds smart but doesn't add up.

<div style="display: flex; justify-content: center; margin: 2.5rem 0;">
    <canvas id="radarCanvas" width="800" height="760" style="width: 100%; max-width: 400px; height: auto; aspect-ratio: 400/380;"></canvas>
</div>

## Silmaril persistent memory system

Research doesn't happen in a single session. **Silmaril** is Marvin's persistent memory system — a living knowledge base that spans all sessions to preserve the results, findings, decisions, and the connections between them. With intelligent depth, agents share context at whatever granularity their task demands, as efficiently as possible. Whether it's Marvin, Vera, or a swarm of sub-agents, they all pick up exactly where the collective left off.

<div style="display: flex; justify-content: center; margin: 2.5rem 0;">
    <canvas id="graphCanvas" width="1400" height="700" style="width: 100%; max-width: 700px; height: auto; aspect-ratio: 2/1;"></canvas>
</div>

## Your repo's "logic trail"

Designed for ML research teams by ML researchers.

<div class="output-tree">
    <div style="color: #dcdcaa;">my-project/</div>
    <div>&nbsp;&nbsp;├── <span style="color: #9cdcfe;">research_state.md</span> <span style="color: #555;">— goals, status, findings</span></div>
    <div>&nbsp;&nbsp;├── <span style="color: #dcdcaa;">docs/</span></div>
    <div>&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── <span style="color: #9cdcfe;">iteration_001.md</span> <span style="color: #555;">— scoreboard, analysis, lessons</span></div>
    <div>&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── <span style="color: #9cdcfe;">iteration_002.md</span></div>
    <div>&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── <span style="color: #ce9178; font-weight: 500;">handoff.md</span> <span style="color: #555;">— structured summary for the paper</span></div>
    <div>&nbsp;&nbsp;├── <span style="color: #dcdcaa;">experiments/</span></div>
    <div>&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── <span style="color: #888;">batch_001/</span></div>
    <div>&nbsp;&nbsp;└── <span style="color: #dcdcaa;">literature/</span></div>
</div>

Every iteration, every interpretation, and every decision is traceable and reproducible with an auditable git history.

From zero to autonomous research in just three steps.

<div class="code-block">
    <div><span style="color: #666;">$</span> <span style="color: #dcdcaa;">marvin</span> <span style="color: #9cdcfe;">init</span></div>
    <div><span style="color: #666;">$</span> <span style="color: #dcdcaa;">marvin</span> <span style="color: #9cdcfe;">plan</span> <span style="color: #555;"># scope your research interactively</span></div>
    <div><span style="color: #666;">$</span> <span style="color: #dcdcaa;">marvin</span> <span style="color: #9cdcfe;">run</span>  <span style="color: #555;"># autonomous research loop</span></div>
    <div><span style="color: #666;">$</span> <span style="color: #dcdcaa;">marvin</span> <span style="color: #9cdcfe;">write</span></div>
</div>

<div markdown="0">
<h2 id="marvins-work">Read Marvin and Vera's latest work</h2>
<div class="paper-card" onclick="window.open('/assets/marvin/marvin-1.pdf','_blank')" style="cursor: pointer;">
    <div class="paper-tag">Autonomous AI Research</div>
    <div class="paper-title">Why Partial Rename Invariance Fails in Transformers</div>
    <div class="paper-authors">Marvin · Vera</div>
    <div class="paper-abstract">Code LLMs break on 8–13% of all HumanEval problems when variables are renamed. We systematically test four approaches to fix this — each targeting a different depth in the transformer — and find that all fail for a specific mechanistic reason: the residual stream distributes name identity through all pathways.</div>
    <div class="paper-meta">
        <span>Read paper →</span>
    </div>
</div>
</div>

## Try Marvin

Marvin is in active development and designed for AI research by AI researchers. If you're an ML engineer, scientist, or even a hobbyist interested in trying it, please reach out. We'd love to hear about your project's needs and discuss how Marvin can help.

<a href="mailto:marvin@iluvatarlabs.com" class="cta-email">marvin@iluvatarlabs.com</a>

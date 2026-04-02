---
layout: product
title: "Introducing Actuator"
date: 2026-04-01
category: Product
author: "Iluvatar Labs"
product_name: "Actuator"
product_subtitle: "The end-to-end control layer for model transformation"
product_tagline: "Post-training is where differentiation and value are created &mdash; and quality is destroyed. By monitoring and adjusting training dynamics to course-correct when quality drifts, Actuator preserves what makes your models smart, before the damage is done."
product_logo: '<svg width="200" height="200" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><g><g><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 32 19.2 L 43.136 25.6 L 43.136 38.4 L 32 44.8 L 20.864 38.4 L 20.864 25.6 Z"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 32 19.2 L 32 32"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 43.136 25.6 L 32 32"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 43.136 38.4 L 32 32"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 32 44.8 L 32 32"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 20.864 38.4 L 32 32"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 20.864 25.6 L 32 32"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 32 19.2 L 32 11.52"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 43.136 25.6 L 48.64 22.4"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 43.136 38.4 L 48.64 41.6"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 32 44.8 L 32 52.48"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 20.864 38.4 L 15.36 41.6"/><path fill="none" stroke="#ffffff" stroke-width="1.024" stroke-linecap="round" d="M 20.864 25.6 L 15.36 22.4"/></g><circle cx="32" cy="19.2" r="2.048" fill="#fff"/><circle cx="43.136" cy="25.6" r="2.048" fill="#fff"/><circle cx="43.136" cy="38.4" r="2.048" fill="#fff"/><circle cx="32" cy="44.8" r="2.048" fill="#fff"/><circle cx="20.864" cy="38.4" r="2.048" fill="#fff"/><circle cx="20.864" cy="25.6" r="2.048" fill="#fff"/><circle cx="32" cy="11.52" r="1.536" fill="#fff"/><circle cx="48.64" cy="22.4" r="1.536" fill="#fff"/><circle cx="48.64" cy="41.6" r="1.536" fill="#fff"/><circle cx="32" cy="52.48" r="1.536" fill="#fff"/><circle cx="15.36" cy="41.6" r="1.536" fill="#fff"/><circle cx="15.36" cy="22.4" r="1.536" fill="#fff"/><circle cx="32" cy="32" r="4.48" fill="#fff"/><circle cx="32" cy="32" r="3.2" fill="#000"/><circle cx="32" cy="32" r="1.28" fill="#fff"/></g></svg>'
backdrop: "flowing"
hero_centered: true
content_fullwidth: true
excerpt: "A closed-loop control layer for model transformation. Actuator monitors, adjusts, and guardrails your post-training transformations in real time."
---

<style>
        /* ── Full-width editorial sections ── */
        .editorial-section {
            width: 100%;
            padding: 4rem 2rem;
        }

        .editorial-inner {
            max-width: 900px;
            margin: 0 auto;
            position: relative;
            padding-bottom: 1.65rem;
        }

        /* ── Divider ── */
        .section-divider {
            border: none;
            border-top: 1px solid #222;
            max-width: 900px;
            margin: 0 auto;
        }

        /* ── Section 1: Why ── */
        .why-section {
            padding: 5rem 2rem 4rem;
        }

        .why-section h2 {
            /* font-family: 'Space Grotesk', sans-serif; */
            font-size: 2.2rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 1.25rem;
        }

        .why-section p {
            font-size: 1rem;
            color: #bbb;
            line-height: 1.8;
            max-width: 750px;
        }

        .why-section strong { color: #fff; }

        /* ── Architecture diagram ── */
        .arch-section {
            padding: 3rem 2rem 4rem;
        }

        .arch-diagram {
            background: #080808;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 2rem 1.75rem;
            margin: 0;
        }

        .arch-stages {
            display: flex;
            align-items: flex-start;
            justify-content: center;
            gap: 0.95rem;
            margin-bottom: 0;
            flex-wrap: wrap;
        }

        .arch-stage-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0;
        }

        .arch-stage {
            background: #111;
            border: 1px solid #999;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: #fff;
            text-align: center;
            white-space: nowrap;
        }

        .arch-loop {
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #f5f5f5;
            font-size: 1.25rem;
            line-height: 1.05;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .arch-loop span { display: block; }

        .arch-arrow {
            color: #bbb;
            font-size: 1.45rem;
            font-weight: 600;
        }

        .arch-control {
            border: 1px solid #666;
            border-radius: 6px;
            padding: 1.25rem 1.5rem;
        }

        .arch-control-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #ddd;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 0.75rem;
            text-align: center;
        }

        .arch-control-flow {
            display: flex;
            justify-content: center;
            width: fit-content;
            max-width: 100%;
            margin: 0 auto;
        }

        .arch-control-grid {
            --arch-end-stage-width: 6.25rem;
            display: grid;
            grid-template-columns: var(--arch-end-stage-width) min-content max-content min-content max-content min-content var(--arch-end-stage-width);
            column-gap: 0.5rem;
            row-gap: 0.12rem;
            align-items: center;
            justify-content: center;
            width: fit-content;
        }

        .arch-control-stage {
            background: #111;
            border: 1px solid #999;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: #fff;
            text-align: center;
            white-space: nowrap;
        }

        .arch-control .arch-arrow {
            color: #eee;
        }

        .arch-control-stage-end {
            width: var(--arch-end-stage-width);
        }

        .arch-return-up {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            line-height: 1;
            color: #eee;
            text-align: center;
            position: relative;
            height: 0.9rem;
            width: 0.9rem;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            grid-column: 1;
            grid-row: 2;
            justify-self: center;
            z-index: 1;
        }

        .arch-return-up::before {
            content: "";
            position: absolute;
            left: 50%;
            top: 0.7rem;
            bottom: 0;
            border-left: 1.5px solid #eee;
            transform: translateX(-50%);
        }

        .arch-return-track {
            grid-column: 1 / 8;
            grid-row: 2;
            position: relative;
            height: 0.9rem;
            align-self: stretch;
            z-index: 0;
        }

        .arch-return-track::before {
            content: "";
            position: absolute;
            left: calc(var(--arch-end-stage-width) / 2);
            right: calc(var(--arch-end-stage-width) / 2);
            bottom: 0;
            border-top: 1.5px solid #eee;
        }

        .arch-return-stem {
            grid-column: 7;
            grid-row: 2;
            justify-self: center;
            position: relative;
            width: 0.9rem;
            height: 0.9rem;
            align-self: stretch;
            z-index: 1;
        }

        .arch-return-stem::before {
            content: "";
            position: absolute;
            left: 50%;
            top: 0;
            bottom: 0;
            border-left: 1.5px solid #eee;
            transform: translateX(-50%);
        }

        .arch-feedback {
            text-align: center;
            font-size: 0.9rem;
            color: #ccc;
            margin-top: 0.55rem;
            font-style: italic;
        }

        /* ── Result panels ── */
        .result-panel {
            padding: 2rem 2rem;
        }

        .result-panel-inner {
            max-width: 900px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 3rem;
            position: relative;
            padding-bottom: 1.65rem;
        }

        .result-panel-inner.reversed {
            flex-direction: row-reverse;
        }

        .result-chart {
            flex: 0 0 50%;
            max-width: 50%;
        }

        .result-text {
            flex: 1;
        }

        .result-category {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            line-height: 1.05;
            margin-bottom: 0.9rem;
        }

        .result-stat {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 6.2rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            line-height: 1;
            margin-bottom: 0.75rem;
        }

        .result-desc {
            font-size: 1.15rem;
            color: #999;
            margin-bottom: 1.25rem;
            line-height: 1.5;
            /* font-family: 'JetBrains Mono', monospace; */
        }

        .result-takeaway {
            font-size: 1.025rem;
            color: #777;
            line-height: 1.6;
            /* font-style: italic; */
        }

        /* ── SVG line chart ── */
        .line-chart-container {
            width: 100%;
        }

        .line-chart-container svg {
            width: 100%;
            height: auto;
            display: block;
        }

        .chart-legend {
            display: flex;
            gap: 1.25rem;
            margin-top: 0.75rem;
            flex-wrap: wrap;
        }

        .chart-legend-item {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
            color: #888;
            font-family: 'JetBrains Mono', monospace;
        }

        .chart-legend-swatch {
            width: 12px;
            height: 2px;
            border-radius: 1px;
        }

        .chart-legend-swatch.dashed {
            background: repeating-linear-gradient(to right, #777 0, #777 3px, transparent 3px, transparent 6px);
        }

        /* ── Horizontal bar charts ── */
        .bar-chart {
            width: 100%;
        }

        .bar-group {
            margin-bottom: 1.5rem;
        }

        .bar-group:last-child {
            margin-bottom: 0;
        }

        .bar-group-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: #777;
            margin-bottom: 0.6rem;
            letter-spacing: 0.02em;
        }

        .bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.4rem;
        }

        .bar-label {
            flex: 0 0 130px;
            font-size: 0.85rem;
            color: #888;
            font-family: 'JetBrains Mono', monospace;
            text-align: right;
            padding-right: 0.75rem;
            white-space: nowrap;
        }

        .bar-track {
            flex: 1;
            height: 20px;
            background: #151515;
            border-radius: 2px;
            position: relative;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 2px;
            transition: width 0.6s ease;
        }

        .bar-value {
            flex: 0 0 50px;
            font-size: 0.85rem;
            color: #aaa;
            font-family: 'JetBrains Mono', monospace;
            padding-left: 0.5rem;
        }

        /* ── Code block ── */
        .code-section {
            padding: 3rem 2rem 2rem;
        }

        .code-section h2 {
            /* font-family: 'Space Grotesk', sans-serif; */
            font-size: 2.2rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 1.25rem;
        }

        .code-block {
            background: #080808;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 1.5rem 1.75rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 2;
            color: #888;
            margin-bottom: 1.5rem;
        }

        .code-block .kw { color: #c586c0; }
        .code-block .fn { color: #dcdcaa; }
        .code-block .str { color: #ce9178; }
        .code-block .num { color: #b5cea8; }
        .code-block .cm { color: #555; }
        .code-block .var { color: #9cdcfe; }
        .code-block .op { color: #888; }

        .code-caption {
            font-size: 0.8rem;
            color: #333;
            margin-top: -0.75rem;
            margin-bottom: 0;
            font-style: italic;
        }

        /* ── CTA ── */
        .cta-section {
            padding: 2rem 2rem 6rem;
        }

        .cta-inner {
            max-width: 900px;
            margin: 0 auto;
            border-top: 1px solid #444;
            padding-top: 3rem;
            padding-bottom: 1.65rem;
            position: relative;
        }

        .section-footnote {
            position: absolute;
            right: 0;
            bottom: 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            color: #4d4d4d;
            text-transform: uppercase;
            text-align: right;
            white-space: nowrap;
        }

        .arch-footnote {
            position: static;
            margin-top: 0.85rem;
            font-size: 0.8rem;
            color: #999;
        }

        .cta-inner h2 {
            font-size: 2.2rem;
            font-weight: 700;
            color: #fff;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 1rem;
        }

        .cta-inner p {
            font-size: 1rem;
            color: #bbb;
            line-height: 1.8;
            margin-bottom: 1.5rem;
        }

        .cta-inner a {
            color: #fff;
            text-decoration: underline;
            text-underline-offset: 2px;
        }
        .cta-inner a:hover { opacity: 0.8; }

        .cta-email {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.1rem;
        }

        /* ── Responsive ── */
        @media (max-width: 768px) {
            .splash-hero-lockup h1 { font-size: 3.1rem; }
            .splash-hero-lockup img,
            .splash-hero-lockup > svg { width: 148px; height: 148px; margin-bottom: -0.9rem; }
            .splash-hero-lockup { gap: 0.35rem; margin-bottom: 0.85rem; }
            .splash-hero h2 { font-size: 1.15rem; margin-bottom: 1.25rem; }
            .splash-hero p { font-size: 0.95rem; line-height: 1.65; }

            .arch-stages { gap: 0.55rem; margin-bottom: 0; }
            .arch-stage-group { gap: 0; }
            .arch-stage { font-size: 0.85rem; }
            .arch-loop { font-size: 1rem; margin-top: 0.35rem; margin-bottom: 0.35rem; }
            .arch-control { padding: 1rem; overflow-x: auto; }
            .arch-control-label { font-size: 0.7rem; }
            .arch-control-flow { min-width: max-content; }
            .arch-control-grid { --arch-end-stage-width: 5.55rem; column-gap: 0.35rem; row-gap: 0.08rem; }
            .arch-control-stage { font-size: 0.78rem; padding: 0.4rem 0.65rem; }
            .arch-arrow { font-size: 1.2rem; }
            .arch-return-up { font-size: 0.85rem; height: 0.8rem; }
            .arch-return-up::before { top: 0.6rem; }
            .arch-return-track,
            .arch-return-stem { height: 0.8rem; }
            .why-section h2,
            .code-section h2,
            .cta-inner h2 { font-size: 1.8rem; }
            .result-category { font-size: 1.15rem; }
            .section-footnote { font-size: 0.62rem; }
            .arch-footnote { font-size: 0.72rem; margin-top: 0.7rem; }

            .result-panel-inner,
            .result-panel-inner.reversed {
                flex-direction: column;
                gap: 2rem;
            }

            .result-chart {
                flex: none;
                max-width: 100%;
                width: 100%;
            }

            .result-text {
                width: 100%;
            }

            .result-stat {
                font-size: 4rem;
            }

            .bar-label {
                flex: 0 0 90px;
                font-size: 0.65rem;
            }

            .editorial-section,
            .why-section,
            .arch-section,
            .result-panel,
            .code-section,
            .cta-section {
                padding-left: 1.5rem;
                padding-right: 1.5rem;
            }
        }
</style>

<!-- Architecture diagram -->
<div class="arch-section">
    <div class="editorial-inner">
        <h2 id="how-it-works" style="font-size: 2.2rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 1.25rem;">How it works</h2>
        <div class="arch-diagram">
            <div class="arch-stages">
                <div class="arch-stage-group">
                    <div class="arch-stage">Fine-tune</div>
                    <div class="arch-loop"><span>&uarr;</span><span>&darr;</span></div>
                </div>
                <div class="arch-stage-group">
                    <div class="arch-stage">Align</div>
                    <div class="arch-loop"><span>&uarr;</span><span>&darr;</span></div>
                </div>
                <div class="arch-stage-group">
                    <div class="arch-stage">Distill</div>
                    <div class="arch-loop"><span>&uarr;</span><span>&darr;</span></div>
                </div>
                <div class="arch-stage-group">
                    <div class="arch-stage">Compress</div>
                    <div class="arch-loop"><span>&uarr;</span><span>&darr;</span></div>
                </div>
            </div>
            <div class="arch-control">
                <div class="arch-control-label">Actuator Control Layer</div>
                <div class="arch-control-flow">
                    <div class="arch-control-grid">
                        <div class="arch-control-stage arch-control-stage-end">Monitor</div>
                        <div class="arch-arrow">&rarr;</div>
                        <div class="arch-control-stage">Adjust</div>
                        <div class="arch-arrow">&rarr;</div>
                        <div class="arch-control-stage">Guardrail</div>
                        <div class="arch-arrow">&rarr;</div>
                        <div class="arch-control-stage arch-control-stage-end">Certify</div>
                        <div class="arch-return-up">&uarr;</div>
                        <div class="arch-return-track" aria-hidden="true"></div>
                        <div class="arch-return-stem" aria-hidden="true"></div>
                    </div>
                </div>
                <div class="arch-feedback">&harr; continuous closed-loop feedback</div>
            </div>
        </div>
        <div class="section-footnote arch-footnote">MULTIPLE PATENTS PENDING</div>
    </div>
</div>

<!-- ═══════════════════════════════════════════════════ -->
<!-- Results                                             -->
<!-- ═══════════════════════════════════════════════════ -->

<!-- Results header -->
<hr class="section-divider">
<div class="result-panel" style="padding-bottom: 1rem;">
    <div class="editorial-inner" style="padding-bottom: 0;">
        <h2 id="results" style="font-size: 2.2rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; line-height: 1.1; margin-bottom: 1.25rem;">Better control, better models</h2>
        <p style="font-size: 1rem; color: #bbb; line-height: 1.8; max-width: 750px;">Today's post-training stack is a fragmented, open-loop affair. Teams pull together multiple tools, set knobs, run blind, eval after, and repeat. Actuator replaces that process with continuous live monitoring, automatic training-time adjustments, and guardrails to keep your model transformations on track. Quality in, quality out.</p>
    </div>
</div>

<!-- Panel 1 — Safety & alignment -->
<div class="result-panel">
    <div class="result-panel-inner">
        <div class="result-text">
            <div class="result-category" style="color: #22c55e;">Safety &amp; Alignment</div>
            <div class="result-stat">+56.9pp</div>
            <div class="result-desc">Worst-group accuracy @ 90% sparsity</div>
            <div class="result-takeaway">Actuator stays unbiased while post-hoc methods collapse under aggressive compression.</div>
        </div>
        <div class="result-chart">
            <div class="line-chart-container">
                <svg viewBox="0 0 400 260" xmlns="http://www.w3.org/2000/svg">
                    <!-- Grid -->
                    <line x1="60" y1="30" x2="60" y2="210" stroke="#111" stroke-width="1"/>
                    <line x1="60" y1="210" x2="370" y2="210" stroke="#222" stroke-width="1"/>
                    <line x1="60" y1="30" x2="370" y2="30" stroke="#111" stroke-width="0.5"/>
                    <line x1="60" y1="60" x2="370" y2="60" stroke="#111" stroke-width="0.5"/>
                    <line x1="60" y1="90" x2="370" y2="90" stroke="#111" stroke-width="0.5"/>
                    <line x1="60" y1="120" x2="370" y2="120" stroke="#111" stroke-width="0.5"/>
                    <line x1="60" y1="150" x2="370" y2="150" stroke="#111" stroke-width="0.5"/>
                    <line x1="60" y1="180" x2="370" y2="180" stroke="#111" stroke-width="0.5"/>

                    <!-- Y-axis labels -->
                    <text x="50" y="214" fill="#666" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="end">0.0</text>
                    <text x="50" y="154" fill="#666" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="end">0.2</text>
                    <text x="50" y="94" fill="#666" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="end">0.4</text>
                    <text x="50" y="34" fill="#666" font-family="JetBrains Mono, monospace" font-size="11" text-anchor="end">0.6</text>

                    <!-- X-axis labels -->
                    <text x="112" y="230" fill="#666" font-family="JetBrains Mono, monospace" font-size="12" text-anchor="middle">50%</text>
                    <text x="215" y="230" fill="#666" font-family="JetBrains Mono, monospace" font-size="12" text-anchor="middle">70%</text>
                    <text x="318" y="230" fill="#666" font-family="JetBrains Mono, monospace" font-size="12" text-anchor="middle">90%</text>

                    <text x="215" y="250" fill="#555" font-family="JetBrains Mono, monospace" font-size="12.5" text-anchor="middle">Sparsity</text>
                    <text x="14" y="120" fill="#555" font-family="JetBrains Mono, monospace" font-size="12.5" text-anchor="middle" transform="rotate(-90, 14, 120)">Worst-group accuracy</text>

                    <!-- Post-hoc ERM (dashed gray) -->
                    <polyline points="112,45 215,165 318,210" fill="none" stroke="#777" stroke-width="1.5" stroke-dasharray="5,4"/>
                    <circle cx="112" cy="45" r="3" fill="#777"/>
                    <circle cx="215" cy="165" r="3" fill="#777"/>
                    <circle cx="318" cy="210" r="3" fill="#777"/>

                    <!-- Post-hoc CDA (solid gray) -->
                    <polyline points="112,36 215,120 318,210" fill="none" stroke="#888" stroke-width="1.5"/>
                    <circle cx="112" cy="36" r="3" fill="#888"/>
                    <circle cx="215" cy="120" r="3" fill="#888"/>
                    <circle cx="318" cy="210" r="3" fill="#888"/>

                    <!-- Actuator (green, stays high) -->
                    <polyline points="112,33 215,36 318,39.3" fill="none" stroke="#22c55e" stroke-width="2"/>
                    <circle cx="112" cy="33" r="3.5" fill="#22c55e"/>
                    <circle cx="215" cy="36" r="3.5" fill="#22c55e"/>
                    <circle cx="318" cy="39.3" r="3.5" fill="#22c55e"/>

                    <!-- Value annotation -->
                    <!-- <text x="338" y="43" fill="#777" font-family="JetBrains Mono, monospace" font-size="11">0.569</text> -->
                </svg>
            </div>
            <div class="chart-legend">
                <div class="chart-legend-item">
                    <div class="chart-legend-swatch" style="background: #22c55e;"></div>
                    Actuator
                </div>
                <div class="chart-legend-item">
                    <div class="chart-legend-swatch" style="background: #666;"></div>
                    Post-hoc CDA
                </div>
                <div class="chart-legend-item">
                    <div class="chart-legend-swatch dashed"></div>
                    Post-hoc ERM
                </div>
            </div>
        </div>
        <div class="section-footnote">CivilComments Fairness - DistilBERT</div>
    </div>
</div>

<!-- Panel 2 — Robustness @ compression -->
<hr class="section-divider">
<div class="result-panel">
    <div class="result-panel-inner">
        <div class="result-text">
            <div class="result-category" style="color: #3b82f6;">Robustness @ Compression</div>
            <div class="result-stat">+13.1pp</div>
            <div class="result-desc">Math reasoning vs. Wanda @ 50% sparsity</div>
            <div class="result-takeaway">Co-scheduled compression recovers more quality across six benchmarks (GSM8K, MMLU shown).</div>
        </div>
        <div class="result-chart">
            <div class="bar-chart">
                <div class="bar-group">
                    <div class="bar-group-label">GSM8K (Math Reasoning)</div>
                    <div class="bar-row">
                        <div class="bar-label">Actuator</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 65%; background: #3b82f6;"></div>
                        </div>
                        <div class="bar-value">39.0%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">Wanda</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 43.2%; background: #3a3a3a;"></div>
                        </div>
                        <div class="bar-value">25.9%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">Post-hoc LoRA</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 21.3%; background: #3a3a3a;"></div>
                        </div>
                        <div class="bar-value">12.8%</div>
                    </div>
                </div>
                <div class="bar-group">
                    <div class="bar-group-label">MMLU</div>
                    <div class="bar-row">
                        <div class="bar-label">Actuator</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 86.2%; background: #3b82f6;"></div>
                        </div>
                        <div class="bar-value">51.7%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">Wanda</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 80.8%; background: #3a3a3a;"></div>
                        </div>
                        <div class="bar-value">48.5%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">Post-hoc LoRA</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 77.7%; background: #3a3a3a;"></div>
                        </div>
                        <div class="bar-value">46.6%</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="section-footnote">Pruning - Llama-3.2-3B</div>
    </div>
</div>

<!-- Panel 3 — Serving speed & latency -->
<hr class="section-divider">
<div class="result-panel">
    <div class="result-panel-inner">
        <div class="result-text">
            <div class="result-category" style="color: #f59e0b;">Serving Speed &amp; Latency</div>
            <div class="result-stat">+7.8%</div>
            <div class="result-desc">Acceptance rate vs. pure distillation</div>
            <div class="result-takeaway">Actuator-discovered teacher guidances outperform pure distillation and even hand-crafted rules.</div>
        </div>
        <div class="result-chart">
            <div class="bar-chart">
                <div class="bar-group">
                    <div class="bar-group-label">Draft Model Acceptance Rate</div>
                    <div class="bar-row">
                        <div class="bar-label" style="flex: 0 0 160px;">Actuator (aligned)</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 95%; background: #f59e0b;"></div>
                        </div>
                        <div class="bar-value">25.9%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label" style="flex: 0 0 160px;">Actuator (curated)</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 55%; background: rgba(245, 158, 11, 0.6);"></div>
                        </div>
                        <div class="bar-value">24.3%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label" style="flex: 0 0 160px;">Baseline</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 48%; background: #3a3a3a;"></div>
                        </div>
                        <div class="bar-value">24.0%</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="section-footnote">Distillation to 0.9B - DeepSeek-Coder-6.7B</div>
    </div>
</div>

<!-- Panel 4 — Reasoning & capability -->
<hr class="section-divider">
<div class="result-panel">
    <div class="result-panel-inner">
        <div class="result-text">
            <div class="result-category" style="color: #a855f7;">Reasoning &amp; Capability</div>
            <div class="result-stat">+15.8pp</div>
            <div class="result-desc">Alignment preservation vs. DPO alone</div>
            <div class="result-takeaway">Closed-loop stability prevents 2.3x more capability drift while maintaining reinforcement learning.</div>
        </div>
        <div class="result-chart">
            <div class="bar-chart">
                <div class="bar-group">
                    <div class="bar-group-label">GSM8K (Capability Preserved)</div>
                    <div class="bar-row">
                        <div class="bar-label">Actuator</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 72.2%; background: #a855f7;"></div>
                        </div>
                        <div class="bar-value">43.3%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">DPO</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 45.8%; background: #3a3a3a;"></div>
                        </div>
                        <div class="bar-value">27.5%</div>
                    </div>
                </div>
                <div class="bar-group">
                    <div class="bar-group-label">TruthfulQA (Output Quality)</div>
                    <div class="bar-row">
                        <div class="bar-label">Actuator</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 94.8%; background: #a855f7;"></div>
                        </div>
                        <div class="bar-value">56.9%</div>
                    </div>
                    <div class="bar-row">
                        <div class="bar-label">DPO</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width: 94.0%; background: #3a3a3a;"></div>
                        </div>
                        <div class="bar-value">56.4%</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="section-footnote">UltraFeedback RL - Qwen2.5-3B</div>
    </div>
</div>

<!-- Code snippet -->
<hr class="section-divider">
<div class="code-section">
    <div class="editorial-inner">
        <h2 id="get-started">Plug and play</h2>
        <p style="font-size: 1rem; color: #bbb; line-height: 1.8; max-width: 750px; margin-bottom: 1.5rem;">Actuator makes post-training <b>easy</b>. It drops right in to your existing stack and provides the unified end-to-end software layer you need to ship better models while skipping the pain.</p>

        <div class="code-block">
            <div><span class="kw">import</span> <span class="var">actuator</span></div>
            <div>&nbsp;</div>
            <div><span class="var">run</span> <span class="op">=</span> <span class="var">actuator</span>.<span class="fn">Controller</span>(</div>
            <div>&nbsp;&nbsp;&nbsp;&nbsp;<span class="var">protect</span>   <span class="op">=</span> {<span class="str">"safety"</span>: <span class="num">0.95</span>, <span class="str">"math"</span>: <span class="num">0.70</span>},</div>
            <div>&nbsp;&nbsp;&nbsp;&nbsp;<span class="var">on_drift</span>  <span class="op">=</span> <span class="str">"adjust"</span>,     <span class="cm"># auto adapt pressure</span></div>
            <div>&nbsp;&nbsp;&nbsp;&nbsp;<span class="var">on_breach</span> <span class="op">=</span> <span class="str">"halt"</span>,       <span class="cm"># stop and rollback</span></div>
            <div>&nbsp;&nbsp;&nbsp;&nbsp;<span class="var">certify</span>   <span class="op">=</span> <span class="num">True</span>,         <span class="cm"># produce audit trail</span></div>
            <div>)</div>
            <div>&nbsp;</div>
            <div><span class="var">run</span>.<span class="fn">start</span>(<span class="var">model</span>)  <span class="cm"># any framework</span></div>
        </div>
        <div class="section-footnote">ILLUSTRATIVE INTEGRATION</div>
    </div>
</div>

<!-- CTA -->
<div class="cta-section">
    <div class="cta-inner">
        <h2 id="get-access">Get early access</h2>
        <p>Actuator is currently in closed testing with our early design partners. If your team is running serious post-training and want to do it a better way, please reach out! We're excited to hear about what your team is working on and open to potential pilots or partnerships.</p>
        <p><a href="mailto:actuator@iluvatarlabs.com" class="cta-email">actuator@iluvatarlabs.com</a></p>
    </div>
</div>

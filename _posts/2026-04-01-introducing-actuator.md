---
layout: post
title: "Introducing Actuator"
subtitle: "A closed-loop control layer for model transformation that preserves quality before the damage is done."
date: 2026-04-01
category: Product
thumbnail: actuator
author: "Iluvatar Labs"
excerpt: "A closed-loop control layer for model transformation. Actuator monitors, adjusts, and guardrails your post-training transformations in real time."
---

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

# Iluvatar Open Research Initiative — Executive Summary

## What This Document Is

This is a comprehensive briefing on the Iluvatar Open Research Initiative. It is written to give a new reader full context with zero prior knowledge of Iluvatar Labs, Marvin, or the initiative itself. Every design decision is explained with its rationale.

---

## Company Context

**Iluvatar Labs** is an AI/ML research and commercialization company founded by Yuan Zhao. The company's core focus is post-training methods, inference efficiency, and closed-loop model optimization. Its two primary products are:

- **Actuator**: An end-to-end closed-loop control layer for model transformation (fine-tuning, alignment, distillation, compression). Actuator monitors training dynamics and auto-corrects when quality drifts. This is the company's commercial product, currently in closed testing with design partners. Target raise: $2–4M, enterprise-only GTM.

- **Marvin**: A fully autonomous AI scientist agent. Marvin automates the research workflow end-to-end: literature review, hypothesis generation (via a sub-agent called Vera), experiment design and execution, analysis, and documentation. Marvin produces a full "logic trail" — every iteration, interpretation, and decision is traceable and reproducible. Marvin includes a persistent memory system called Silmaril that maintains context across sessions and agents.

The company has core IP including the CCN (Causal Compression Networks) framework and provisional patents on SCV, NWOR, and adaptive speculative decoding methods. The team includes PhD scientists and advisors at Google Gemini and OpenAI Platform.

---

## What the Iluvatar Open Research Initiative Is

The Iluvatar Open Research Initiative is a public, open science program where Marvin performs computational biology research on behalf of the scientific community — for free, with all findings published openly.

The strategic logic is straightforward: Iluvatar Labs needs public, credible proof that Marvin produces scientifically meaningful output. A community that independently reviews, critiques, and validates Marvin's predictions is far more credible than any internal case study. The initiative is simultaneously genuine open science and a go-to-market strategy for Marvin.

### The Core Loop

1. A biological question is identified (either internally by Iluvatar or submitted by an external PI).
2. Marvin runs a multi-stage computational analysis: literature synthesis, public data integration, differential analysis, network modeling, hypothesis generation.
3. Internal PhD scientists vet every output for quality before anything is published.
4. Findings are published as preprints on bioRxiv with full reproducible pipelines on GitHub.
5. The scientific community reviews, critiques, suggests new data or literature, and ideally performs wet lab validation of the testable hypotheses Marvin produces.
6. Marvin periodically re-runs analysis incorporating new data, community feedback, and new literature. The project is "living" — it evolves over time.

### The Value Exchange

- **For PIs and researchers**: Free computational analysis on their research problems, co-authorship on resulting publications, access to Marvin's throughput and integration capabilities that would otherwise require a dedicated bioinformatics team.
- **For Iluvatar Labs**: Public demonstration of Marvin's capabilities, community-validated evidence of scientific value, brand recognition in the computational biology community, and inbound interest from potential enterprise customers and investors.
- **For the field**: Open, reproducible computational analyses on underserved problems (rare diseases, underfunded therapeutic areas), plus a new model for AI-assisted collaborative science.

---

## Structure: Two Tracks

### Track 1: Showcase Projects (Internally Initiated)

Iluvatar selects 2–3 high-impact computational biology problems, runs them through Marvin, vets the results internally, and publishes the full analysis. The community is invited to review, critique, and contribute.

**Purpose**: Proof of capability. These projects demonstrate what Marvin can do at its best, under conditions Iluvatar controls end-to-end. They provide the initial content that seeds community engagement.

**Selection criteria for showcase projects**:
- Rare disease or underserved therapeutic areas (higher community goodwill, lower competitive overlap with pharma)
- Problems requiring integration of fragmented public data across modalities (genomics, transcriptomics, proteomics, clinical)
- Available public data is substantial but not already well-synthesized
- Clear, achievable wet lab validation steps performable in a typical academic lab
- High probability of a non-obvious finding (not just confirming existing consensus)

### Track 2: Community-Submitted Projects (PI-Initiated)

External PIs submit biological questions they want Marvin to work on. Iluvatar selects a subset, runs the analysis, and delivers results to the PI and the public simultaneously.

**Purpose**: Network growth and word-of-mouth. A PI who gets their own problem analyzed becomes an evangelist in a way that a PI reviewing a showcase project never will. This is the growth engine.

**Why Track 2 matters more long-term**: The "pitch us your problem" framing inverts the value proposition — PIs use Marvin on their agenda, not Iluvatar's. This generates genuine engagement, sharing, and stories that spread organically through academic and industry networks.

---

## Submission and Selection Process (Track 2)

### Submission Format

A structured web form with required fields:
- Biological question (500 words max)
- Therapeutic or biological area
- Known relevant public datasets
- What computational analysis is needed
- What wet lab validation the PI could perform if findings warrant it
- Lab's relevant expertise and publications
- Why the problem matters (250 words max)
- Willingness to provide written feedback within 30 days (Y/N)
- Willingness to present at a community call (Y/N)

**Why structured form**: Standardizes triage, forces PI clarity, scales better than open-ended proposals, and levels the playing field (good problems matter more than good proposal writing).

### Selection Model: Quarterly Cohorts

Submissions open for 4 weeks, selection announced 2 weeks after close, work begins immediately. 3–5 projects per cohort initially.

**Why cohorts over rolling admission**:
- Creates urgency via deadlines (PIs respond to deadlines; open-ended calls get procrastinated)
- Enables batch evaluation and portfolio balancing across disease areas
- Generates regular cadence of announcements that sustains visibility
- Prevents the "submitted months ago, heard nothing" problem

### Selection Criteria (Weighted)

1. **Scientific tractability** (highest weight): Enough public data for meaningful analysis? Question well-defined enough for concrete, testable predictions?
2. **Impact and novelty**: Unmet need? Current literature inadequate?
3. **Validation feasibility**: Can the PI (or someone) realistically perform wet lab validation?
4. **Marvin's comparative advantage**: Does an automated system add clear value over a postdoc with standard tools?
5. **Portfolio diversity**: Balance across disease areas, data modalities, and institution types.

### Selection Committee

2–3 internal PhD scientists plus 1–2 rotating external advisors. External advisors provide credibility and domain breadth.

### Rejection Handling

Every rejected submission gets a brief, specific explanation and an invitation to resubmit. No ghosting. This is non-negotiable for community goodwill.

---

## Deliverable Tiers

### Tier 1: Literature Synthesis + Hypothesis Generation
- Structured literature review, dataset identification, gap analysis, generated hypotheses
- Timeline: 2–4 weeks
- For submissions where data availability is limited but the question is strong

### Tier 2: Computational Analysis on Public Data (Default)
- Everything in Tier 1, plus systematic analysis across public datasets, ML-based target/biomarker identification, reproducible pipeline published
- Delivered as preprint-ready manuscript draft + analysis notebooks
- Timeline: 6–10 weeks

### Tier 3: Full Pipeline with Living Updates
- Everything in Tier 2, plus ongoing monitoring for new data/publications, periodic re-analysis, living dashboard
- Timeline: 8–12 weeks initial, updates ongoing for 6 months
- Reserved for high-impact projects with strong PI engagement

---

## Openness Model

### Open (from day one)
- All analysis results and findings
- Computational pipelines, code, and environments (Apache 2.0 for code, CC-BY 4.0 for text/figures)
- Preprints on bioRxiv
- Raw intermediate outputs
- Selection criteria, cohort decisions, reasoning
- Project status dashboards

### Not Open
- Marvin's core architecture, training methodology, and proprietary systems
- Internal review deliberations (outputs of review are public; internal process is not)
- Submission details for unselected projects

### Why This Boundary
The line is at "outputs vs. engine." Everything Marvin produces is open. How Marvin works internally is commercial IP. This is defensible and standard (analogous to Google publishing research while keeping infrastructure proprietary). The key credibility signal: full computational pipelines are published, so anyone can reproduce the analysis with their own tools.

---

## IP and Attribution

### IP Stance
Iluvatar takes no IP on any findings produced through the initiative. The entire point is open science. If collaborators or third parties choose to pursue IP on downstream work, that is their prerogative.

### Collaboration Agreement
A one-page, plain-language document signed by selected PIs:

**Iluvatar commits to**: Delivering agreed analysis tier on stated timeline, publishing all findings openly, not claiming IP, crediting the PI as lead collaborator.

**PI commits to**: Providing written feedback within 30 days (best effort), allowing Iluvatar to publicly describe the collaboration, acknowledging Marvin/Iluvatar in resulting publications, sharing wet lab validation results if performed (strongly encouraged, not required).

**Why data sharing is encouraged but not required**: Mandatory sharing deters the best PIs — those with the most to lose and the most valuable data. Social incentive via public contributor status is more effective than contractual obligation.

### Authorship Framework
Uses CRediT (Contributor Roles Taxonomy):
- Submitting PI: Conceptualization, Supervision, plus any roles actively filled
- Iluvatar/Marvin team: Methodology, Software, Formal Analysis, Data Curation, Visualization
- Community contributors: Credited for specific roles based on actual contributions
- Corresponding author: Submitting PI (Track 2) or Iluvatar lead scientist (Track 1)
- Marvin: Acknowledged in methods section, not listed as an author (avoids the "AI authorship" distraction)

---

## Platform Architecture

### GitHub Organization: Dedicated Org (Not Under Iluvatar's Main Org)

Signals community project, not corporate product repo.

Structure:
```
iluvatar-open-research/
├── overview/              # Front door: what this is, how to submit, FAQ
├── community/             # Meta-discussions, cohort announcements, retrospectives
├── shared-pipelines/      # Reusable Marvin analysis modules
├── project-<slug>/        # One repo per accepted project
├── project-<slug>/
└── ...
```

### One Repo Per Project (Not a Monorepo)

Reasons: access control per-project, scoped issues/discussions, clean forking, per-project watch/notification settings. At 3–5 projects per quarter, manageable scale.

### Standardized Project Repo Layout

```
project-<slug>/
├── README.md                # Status, key findings, how to contribute
├── docs/
│   ├── question.md          # Original biological question
│   ├── status.md            # Current phase + what's needed
│   └── changelog.md         # Dated log of updates
├── literature/
│   ├── synthesis.md         # Marvin's literature review
│   └── references.bib
├── data/
│   ├── sources.md           # All public datasets used
│   ├── processed/
│   └── community-contributed/
├── analysis/
│   ├── 01-data-integration/
│   ├── 02-feature-analysis/
│   ├── 03-target-identification/
│   └── ...
├── findings/
│   ├── summary.md           # Plain-language summary
│   ├── hypotheses.md        # Specific testable predictions (MOST IMPORTANT FILE)
│   └── figures/
├── validation/
│   ├── plan.md              # Proposed wet lab experiments
│   └── results/
├── preprint/
│   ├── manuscript.md
│   └── supplementary/
└── environment/
    ├── Dockerfile           # Fully reproducible environment
    └── requirements.txt
```

### The Hypothesis File Format

Each hypothesis in `findings/hypotheses.md` is formatted as:
- **Prediction**: Specific, falsifiable statement
- **Supporting evidence**: With links to analysis outputs
- **Confidence level**: High / Moderate / Low with justification
- **Suggested validation experiment**: Specific protocol a wet lab PI can act on
- **Status**: Awaiting Validation / In Discussion / Validation In Progress / Validated / Refuted

This format is critical. It minimizes the gap between "computational finding" and "thing someone can act on in a lab." Vague findings don't get validated. Specific, experimentally-scoped hypotheses do.

### Marvin Update Flow

1. Marvin runs new analysis iteration (new data, refined parameters, or scheduled re-analysis)
2. Internal PhD team reviews and vets outputs
3. Pull request opened against project repo with structured summary: what changed, new datasets, new/revised hypotheses, retractions
4. PR left open for 7 days for community review
5. After review period, PR merged, README/changelog updated
6. Automated `marvin-update` issue opened for visibility

**Why PRs, not direct commits**: PRs create a visible review moment — community members get notified, diffs are reviewable, discussion attaches to specific changes. Also creates auditable record of how findings evolved over time.

### Issue Templates

Four standardized templates per project repo:
1. **Scientific Critique / Question** — concerns about methodology or results
2. **New Data Source** — suggested datasets with accession numbers and relevance
3. **Literature Suggestion** — relevant papers with citation and key finding
4. **Validation Offer** — which hypothesis, what experimental system, estimated timeline

### Community Repo

GitHub Discussions categories: Announcements, Project Ideas (low-friction idea pipeline for future cohorts), Methods & Tools, Introductions, Meta (feedback on the initiative itself).

---

## Website

The initiative has its own section on the Iluvatar Labs website (not a separate domain), designed to match the existing site's aesthetic (dark, minimal, generous spacing, arrow-link CTAs, monospace metadata).

### Two Layers

- **The website** is the reading layer: curated, narrative presentation of each project's current state. Designed for a PI who has 5 minutes to evaluate whether to engage.
- **GitHub** is the working layer: where contributions happen, code lives, issues get filed, PRs get reviewed.

Every action button on the website ("Discuss on GitHub," "View Pipeline," "Offer to Validate") routes to the appropriate GitHub location. The site never replicates GitHub's collaboration functionality.

### Site Structure

**Landing page**: Hero with tagline → How It Works (6-step interactive process) → Active Projects (clickable list with status, track, metrics) → Get Involved (submit / review / validate) → Submit CTA.

**Project detail page**: Project header with status + stats → Tabbed content:
- **Findings & Hypotheses**: Narrative summary, caveat, expandable hypothesis cards with prediction / evidence / validation / confidence / action buttons
- **Data Sources**: Table of all public datasets used, with "know a dataset we missed?" prompt
- **Timeline**: Color-coded chronological log of milestones, Marvin updates, internal reviews, community contributions, plus next scheduled iteration
- **How to Contribute**: Four contribution paths at different commitment levels (review / suggest data / validate / discuss), plus attribution explanation

### Key Design Decision

The hypothesis cards on the project detail page are the most important UI element. Each card is self-contained: a PI should be able to read one hypothesis and know within 60 seconds whether they can act on it. The "I can validate this →" link is the primary conversion event for the entire initiative.

---

## Rollout Plan

### Phase 0: Internal Preparation (Weeks 1–6)
- Select 2–3 showcase projects (Track 1), begin Marvin analysis
- Internal PhD team vets outputs to publication-quality bar
- Build submission form and website section
- Draft Collaboration Agreement
- Recruit 2 external advisors for selection committee
- Prepare communications materials

### Phase 1: Soft Launch (Weeks 7–12)
- Invite 5–8 trusted PIs from existing network to submit Track 2 problems
- Run their projects through the full pipeline
- Simultaneously publish first Track 1 showcase preprints on bioRxiv
- Identify and fix process failures: timeline realism, form design, output granularity
- Collect testimonials and case studies

**Why soft launch**: If the first public cohort has rough execution — slow turnaround, unclear deliverables, poor communication — the best first impression is burned. Soft launch allows private failure with forgiving participants.

### Phase 2: Public Launch (Weeks 13–16)
- Blog post leading with Phase 1 results (a finding, not a product pitch)
- Open Cohort 1 submissions
- Distribution: bioRxiv community, computational biology Bluesky/Twitter, relevant subreddits, direct outreach to PIs whose work overlaps showcase projects

### Phase 3: Steady State (Ongoing)
- Quarterly cohorts for Track 2
- Continuous updates on active projects
- Quarterly community calls (recorded, posted publicly)
- Annual retrospective: what worked, what didn't, what Marvin got wrong

---

## Marketing Strategy

### Core Principle: The Science Is the Marketing

Every marketing effort amplifies a scientific result, not a product claim. The most effective advertisement for Marvin is a bioRxiv preprint with a surprising finding that gets discussed organically.

### Tactics
- **Launch announcement**: Long-form blog post telling the story of one showcase project end-to-end. Narrative, not press release.
- **PI testimonials**: Short quotes or clips from soft-launch participants.
- **Conference presence**: Posters/talks at ISMB, RECOMB, ASHG, domain-specific meetings. Science on the poster, "Powered by Iluvatar" in the corner.
- **Direct outreach**: Personal emails from PhD scientists to targeted PIs whose work suggests they'd benefit. Not mass marketing.
- **Content cadence**: Weekly brief social updates, monthly longer blog posts, quarterly cohort announcements + community calls, annual retrospective.

### What Not to Do
- No press releases unless genuinely newsworthy (validated target, journal publication)
- No competitive comparisons with other AI scientist systems in initiative communications
- No over-designed marketing site — information density over polish for a scientist audience

---

## Success Metrics

### 6 Months
- 2–3 Track 1 preprints on bioRxiv
- First Track 2 cohort completed (3–5 projects)
- At least 1 project generating community discussion
- 20+ submissions for Cohort 2
- 2+ PI testimonials

### 12 Months
- At least 1 wet lab validation initiated
- At least 1 journal submission
- Organic inbound exceeds capacity (rejecting more than accepting)
- Identifiable word-of-mouth referrals
- External citation of a project preprint

### 18–24 Months
- A validated finding directly attributable to the initiative
- Self-sustaining community engagement
- Marvin recognized as credible in the computational biology community
- Inbound enterprise interest from organizations who found Marvin through the initiative

---

## Key Risks

| Risk | Mitigation |
|------|------------|
| Marvin produces a high-profile incorrect finding | Multi-stage PhD vetting. All findings framed as hypotheses, not claims. Publish uncertainty estimates. |
| Low submission volume | Soft-launch with known PIs. Direct targeted outreach. Don't rely on organic discovery alone. |
| High volume overwhelms capacity | Cohort model caps intake. Clear rejection with feedback. Scale gradually. |
| PI takes findings, publishes without acknowledgment | Collaboration Agreement requests acknowledgment. Preprints establish priority. Even if it happens, it demonstrates Marvin's value. |
| "Openwashing" accusations | Publish full pipelines, not just conclusions. Be explicit about what's open vs. proprietary. Engage with criticism directly. |
| Showcase findings are obvious or confirmatory | Choose problems carefully. Publish anyway with honest framing. |
| Scope creep per project | Tier system sets expectations upfront. Resist mid-project scope expansion. |

---

## Critical Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Name | Iluvatar Open Research Initiative | Centers the company brand, not a single product |
| Two tracks | Showcase + Community-submitted | Showcase controls quality narrative; community track drives growth |
| Cohort model | Quarterly | Urgency, portfolio balancing, regular announcement cadence |
| Primary collaboration platform | GitHub (dedicated org) | Target audience already there; persistent, public, version-controlled |
| One repo per project | Yes | Access control, scoped issues, clean forking |
| Website role | Reading/evaluation layer | Curated presentation for PIs; all action routes to GitHub |
| Marvin updates via PRs | Yes | Visible review moment, audit trail, community notification |
| Authorship model | CRediT taxonomy | Granular credit, avoids author-list politics |
| Marvin as author | No (methods acknowledgment) | Avoids distracting AI authorship debate |
| Data sharing by PIs | Encouraged, not required | Mandatory sharing deters top PIs; social incentive is more effective |
| IP stance | Iluvatar takes no IP | Removes participation barriers, builds trust |
| Preprint-first | Always | Establishes priority, enables community review, signals openness |
| Marketing tone | Science-first | Target audience is allergic to product marketing; credibility is the only currency |
| Rejection communication | Always with specific reasoning | Ghosting destroys community trust |
| Soft launch before public | Yes | Process debugging with forgiving audience before public exposure |
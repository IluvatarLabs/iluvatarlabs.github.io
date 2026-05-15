# IORI Spec v2

Supersedes `IORI_exec.md` (the original brainstorming doc, which was
bio-locked and partially hallucinated by a pre-compaction agent).
This document reflects decisions made in the May 2026 workshopping
sessions. It is the canonical reference for implementation.

---

## What IORI is

The Iluvatar Open Research Initiative is an open research program
where Marvin runs full research projects on public data, publishes
every iteration as it happens, and invites the field to critique,
extend, and validate. Anyone can contribute. Iluvatar takes no IP
on findings.

**IORI is domain-agnostic.** The first two projects happen to be
computational biology and psychiatric genetics. Future cohorts can
be any field where the model applies (public data exists, the
question is tractable, Marvin adds clear value over a person with
standard tools).

### Strategic logic (unchanged from v1)

Iluvatar needs public, credible proof that Marvin produces
scientifically meaningful output. A community that independently
reviews, critiques, and validates Marvin's findings is more
credible than any internal case study. IORI is simultaneously
genuine open science and a go-to-market strategy for Marvin.

---

## GitHub architecture

### Monorepo (changed from v1)

v1 proposed one repo per project plus overview/community/shared-pipelines
repos. v2 uses a **single monorepo** (`iluvatar-open-research`) to
avoid sprawl as the project count scales.

**Repo**: [`github.com/IluvatarLabs/iluvatar-open-research`](https://github.com/IluvatarLabs/iluvatar-open-research)

Single repo. Issues for structured contributions (5 templates).
Discussions enabled on the same repo for open-ended conversation.
No separate community repo — consolidate until volume demands a
split.

```
IluvatarLabs/iluvatar-open-research
├── README.md                        # front door + project table
├── CONTRIBUTING.md                  # contribution guide
├── LICENSE-CODE                     # Apache 2.0
├── LICENSE-TEXT                     # CC-BY 4.0
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── scientific-critique.yml
│   │   ├── new-data-source.yml
│   │   ├── literature-suggestion.yml
│   │   ├── validation-offer.yml
│   │   └── new-project-idea.yml
│   └── PULL_REQUEST_TEMPLATE.md
├── projects/
│   ├── junb-sasp-skeletal-muscle/
│   │   ├── README.md               # signpost: question, status, links
│   │   ├── hypotheses.md           # card deck with status tracking
│   │   ├── manifest.md             # per-cycle community inputs + changes
│   │   ├── data/
│   │   │   └── sources.md          # dataset provenance
│   │   ├── code/                   # scripts, notebooks, pipelines
│   │   ├── results/                # processed outputs mapped to findings
│   │   │   ├── 01-vascular-junb-sasp/
│   │   │   ├── 02-three-compartment-model/
│   │   │   └── ...
│   │   └── figures/                # supplementary figures
│   └── scz-neuronal-convergence/
│       └── (same structure)
└── templates/
    └── new-project/                 # scaffolding for adding projects
```

The repo is a **supplemental data archive** for the research, not a
narrative layer. The website (`/iori/<slug>/`) is the reading layer;
the repo is the working layer. Each project directory maps to what
a journal supplemental contains: what was claimed (hypotheses), how
it was produced (code), what it produced (results), what data went
in (sources), what it looks like (figures), and what changed and
why (manifest).

**Labels** (on the repo, 7 total):

| Label | Purpose |
|-------|---------|
| `project:muscle-aging` | Scopes to muscle aging project |
| `project:schizophrenia` | Scopes to schizophrenia project |
| `critique` | Scientific critique |
| `data` | Dataset suggestion |
| `literature` | Paper or direction suggestion |
| `validation` | Validation offer |
| `project-idea` | New project proposal |

Issue templates auto-apply type labels. Project labels applied
during triage (GitHub form dropdowns can't auto-label).

**GitHub Discussions** (enabled on the same repo):

| Category | What goes here |
|----------|---------------|
| Announcements | New projects, cycle releases, selections |
| Project Ideas | Low-friction brainstorming before formal submission |
| Methods & Tools | Cross-project analytical discussion |
| Introductions | Who you are, how you want to help |
| General | Everything else, feedback on IORI |

**GitHub Projects board** ("IORI Research Tracker"):

Custom fields: Project (single-select), Contribution (single-select),
Status (New / Under Review / Queued for Next Cycle / Incorporated /
Closed). Saved views per project and per contribution type. All
issues auto-added to the board. Navigation from project READMEs via
pre-filtered links.

**Dropped from v1**: separate org (consolidated under IluvatarLabs),
`overview/` repo, `community/` repo (Discussions on the main repo),
`shared-pipelines/` repo, per-project `question.md` (folded into
README), `findings/` directory (website handles narrative),
`environment/` (premature), `preprint/` (add when one exists).

### Issue templates (5)

Each template includes a **project selector dropdown** so issues
auto-label by project.

1. **Scientific critique** — challenge a methodology or finding
2. **New data source** — suggest a dataset with accession/URL
3. **Literature suggestion** — flag a relevant paper
4. **Validation offer** — which hypothesis, what system, timeline
5. **New project idea** — propose a question for a future cohort

### PR-based update flow (unchanged from v1)

1. Marvin runs new analysis iteration
2. Internal PhDs vet outputs
3. PR opened with structured summary: what changed, new datasets,
   new/revised hypotheses, retractions
4. PR sits open 7 days for community review
5. After review period, merged; README/changelog updated
6. Automated `marvin-update` issue opened for visibility

### Licensing

- Code, pipelines, environments: Apache 2.0
- Text, figures, manuscripts: CC-BY 4.0
- Iluvatar takes no IP on any findings

---

## Website architecture

### Two layers (unchanged from v1)

- **Website** (`iluvatarlabs.github.io/iori/`): the reading layer.
  Curated narrative for a PI with 5 minutes.
- **GitHub** (`github.com/iluvatar-open-research/iori`): the working
  layer. Where contributions happen, code lives, PRs get reviewed.

Every CTA on the website routes to the appropriate GitHub location.
The site never replicates GitHub's collaboration functionality.

### IORI landing page (`/iori/`)

Sections:
1. **Hero** — "Iluvatar Open Research Initiative." / tagline
2. **Thesis** — 4-pillar grid (Proposals, Feedback, Data, Research)
3. **Active projects** — D1 two-column cards showing project name,
   question, why-it-matters, latest iteration, tags, hypothesis count
4. **Contribute** — 5 contribution types with GitHub CTAs

The front page does NOT show full hypothesis cards. It shows a
per-project hypothesis count + confidence breakdown as a teaser.
The project page is the conversion surface.

### Project detail pages (`/iori/<slug>/`)

Rendered via `_layouts/iori-project.html` from `_projects/*.md`.

Sections:
1. **Hero** — project name, question, status, key stats
2. **Summary of discoveries** — the **hypothesis card deck**
3. **Detailed findings** — expanded evidence per hypothesis
4. **Open questions** — absorbed into hypothesis cards (each card
   includes its suggested validation, so this section is redundant
   as a standalone block; retain only if there are project-level
   open questions that don't map to a specific hypothesis)
5. **Sources, datasets, literature** — provenance

### Hypothesis cards (the primary conversion element)

Each card in the "Summary of discoveries" section contains:

| Field | Description |
|-------|-------------|
| **Prediction** | Specific, falsifiable statement (one sentence) |
| **Evidence** | Key statistic(s) with link to detailed section |
| **Confidence** | Strong / High / Moderate (with justification) |
| **Validation** | What experiment or analysis would confirm/refute |
| **Status** | Awaiting · In Discussion · In Progress · Validated · Refuted |
| **CTA** | "I can validate this →" (links to pre-filled GitHub issue) |

Design principle: a PI reads one card and knows within 60 seconds
whether they can act on it.

The current project pages have findings and open questions in
separate sections. The hypothesis card merges them: each finding
carries its own validation experiment and action link.

### Confidence tiers

Current implementation uses three tiers (diverged from v1's
High/Moderate/Low):

| Tier | Meaning |
|------|---------|
| **Strong** | Multiple independent lines of evidence, cross-validated, survives robustness checks |
| **High** | Solid evidence from primary analysis, directionally confirmed but formal power may be limited |
| **Moderate** | Consistent signal but based on inference, single-atlas, or literature synthesis rather than direct experimental test |

---

## Contribution model

### Four contribution types (as on the IORI website)

1. **Propose a project** — submit a research question for Marvin
2. **Review an active project** — challenge findings, suggest
   better analysis, engage with open questions
3. **Share a resource** — datasets, papers, benchmarks
4. **Run a validation** — test a hypothesis through independent
   analysis or experiment

These map to 5 GitHub issue templates (review + share split into
critique, data, and literature templates for structured input).

### Selection

Cohorts timed but triggered at our discretion (no fixed quarterly
cadence). Rolling submissions accepted via the `new-project-idea`
issue template. Selection criteria unchanged from v1.

Selection criteria (unchanged from v1, now field-neutral):
1. Scientific tractability (enough public data? well-defined question?)
2. Impact and novelty (unmet need? current literature inadequate?)
3. Validation feasibility (can someone test the findings?)
4. Marvin's comparative advantage (does an automated system add value?)
5. Portfolio diversity (across fields, data modalities, institutions)

### Attribution

Uses CRediT (Contributor Roles Taxonomy):
- Submitting researcher: Conceptualization, Supervision, plus roles filled
- Iluvatar/Marvin team: Methodology, Software, Formal Analysis, Data Curation
- Community contributors: credited for specific roles based on contributions
- Marvin: acknowledged in methods, not listed as an author

---

## Current project inventory

### 1. Skeletal Muscle Aging (`junb-sasp-skeletal-muscle`)
- **Question**: What drives aging muscle from regeneration toward
  fibrosis and inflammation, and can it be reversed?
- **Key finding**: Vascular endothelial cells, not fibroblasts, are
  the dominant inflammatory source, driven by JUNB/AP-1
- **Hypotheses**: 8 (3 Strong, 2 High, 3 Moderate)
- **Datasets**: 5 (387,000+ cells, 35 donors)
- **Status**: Active analysis, iteration 1

### 2. Mapping Schizophrenia Risk (`scz-neuronal-convergence`)
- **Question**: Can schizophrenia's hundreds of risk loci be resolved
  into targetable biology?
- **Key finding**: Risk converges on constrained synaptic genes in
  neurons, not microglia. EGR1 and CTCF are convergence regulators.
- **Hypotheses**: 7 (4 Strong, 1 High, 2 Moderate)
- **Datasets**: 12 (76,755 cases, 243,649 controls)
- **Status**: Active analysis, iteration 1

---

## Resolved decisions

- [x] GitHub org: under `IluvatarLabs` (not a separate org)
- [x] Repo name: `iluvatar-open-research`
- [x] Single repo (no separate community repo)
- [x] Cohorts timed, triggered at our discretion
- [x] Hypothesis cards on both website and GitHub
- [x] Submissions via GitHub issue template (no separate form)
- [x] Discussions enabled on the research repo
- [x] Community repo (`iori-community`) archived / dormant

## Open decisions

- [ ] Static "Marvin papers" archive section (separate from living
      projects — portfolio of completed work, not active collaboration)
- [ ] GitHub Projects board custom fields + saved views (pending
      auth scope for CLI, or set up manually in GitHub UI)

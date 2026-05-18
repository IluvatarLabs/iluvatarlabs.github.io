# IORI Cycle Publication SOP

Standard operating procedure for going from a Marvin project data
dump to a published cycle commit in the `iluvatar-open-research`
repo. This document must be complete enough that an agent with no
prior context can follow it and produce a correct commit.

---

## Prerequisites

You need access to two things:

1. **The Marvin project directory** (local workspace). The PI will
   tell you the path. Typical location: `~/Desktop/biomarvin_<name>/`
   or `~/Documents/GitHub/<name>/`.

2. **The published project page** on the IORI website. Located at
   `_projects/<slug>.md` in the website repo
   (`IluvatarLabs/iluvatarlabs.github.io`). This has the curated
   findings, confidence tiers, open questions, and sources that have
   already been reviewed and approved. If the project page doesn't
   exist yet, the PI must write it first. The repo commit mirrors
   and extends the website content — it does not replace or
   contradict it.

The target repo is `IluvatarLabs/iluvatar-open-research` (local
clone typically at `~/Documents/GitHub/iluvatar-open-research/`).

---

## Marvin workspace structure

A typical Marvin project directory looks like this:

```
<project_dir>/
├── docs/
│   ├── iteration_*.md          # internal iteration logs (NEVER publish)
│   ├── summary_*.md            # internal summaries (NEVER publish)
│   ├── wip_iteration_*.md      # work-in-progress (NEVER publish)
│   ├── preprint/
│   │   ├── manuscript.md       # manuscript source
│   │   ├── manuscript.pdf      # compiled PDF
│   │   └── figures/            # publication figures + render scripts + backing data
│   ├── vera/                   # sub-agent outputs (NEVER publish)
│   ├── review_*.md             # internal reviews (NEVER publish)
│   ├── PI_briefing_*.md        # internal briefings (NEVER publish)
│   └── (various internal docs) # strategic reviews, cover letters, etc. (NEVER publish)
├── experiments/
│   ├── batch_NNN/              # each batch = one analysis run
│   │   ├── brief.md            # what the batch does (DO NOT publish, but READ for triage)
│   │   ├── *.py                # analysis scripts (publish if batch is relevant)
│   │   ├── *.csv               # result tables (publish if batch is relevant)
│   │   ├── *.json              # result data (publish if batch is relevant)
│   │   ├── *.png               # output plots (publish if batch is relevant)
│   │   └── *.log               # execution logs (NEVER publish)
│   └── ...
├── literature/                 # ingested papers (may be empty)
├── research_state.md           # internal state tracker (NEVER publish)
├── soul.md                     # Marvin behavioral rules (NEVER publish — commercial IP)
└── marvin.yaml                 # project config (NEVER publish — commercial IP)
```

Not all Marvin workspaces look identical. Some may have additional
directories. The rule is the same: only curated analysis code,
result outputs, and publication figures go into the public repo.
Everything else stays private.

---

## Output: what the commit contains

```
projects/<slug>/
├── README.md           # project signpost (create if new project; update if existing)
├── hypotheses.md       # hypothesis cards with status tracking
├── manifest.md         # cycle record: what went in, what changed
├── sources.md          # data provenance organized by cycle
├── code/
│   ├── README.md       # analysis step index + finding mapping
│   ├── 01_<name>/      # renamed, numbered analysis steps
│   ├── 02_<name>/
│   └── ...
├── results/
│   ├── 01_<name>/      # output files, same numbering as code/
│   ├── 02_<name>/
│   └── ...
└── figures/
    ├── figure_1.png
    ├── ...
    └── supplementary_figure_s*.png
```

**Everything goes in ONE commit. Do not push partial content.
Prepare everything locally, get PI approval, then commit and push
in one shot.**

---

## Procedure

### 0. Read the findings first

Before touching any Marvin files, read the published project page
(`_projects/<slug>.md`) on the website repo. Understand:

- How many findings there are and what each claims
- The confidence tier for each finding
- What open questions exist
- What datasets are listed in the sources section

This is the ground truth. The repo commit must be consistent with
the website. The website is the curated narrative; the repo is the
supplemental archive backing it up.

### 1. Triage batches

Read `brief.md` in every experiment batch directory under
`experiments/`. Classify each batch as:

- **Publish**: produced evidence supporting a published finding
- **Exclude (dead end)**: exploratory, abandoned, or superseded
- **Exclude (internal only)**: manuscript writing, editorial review,
  deliverable review, or any batch with no new computation

**If a batch has no `brief.md`**: look at the Python filenames, CSV
output names, and any JSON result files to infer what it does. Check
if the CSV/JSON filenames match terminology from the published
findings (e.g., a file named `canonical_tf_sasp_table_final.csv`
clearly supports the TF-SASP findings). If you still can't
determine what a batch does, exclude it and note it as unclassified.

**If a batch produced intermediate results that were superseded by
a later batch** (e.g., an early TF screen that was later rerun with
corrected parameters): exclude the earlier version. Only publish
the final version that produced the results cited in the findings.

Only "Publish" batches go into the repo. Keep a local record of
which batches were excluded and why.

### 2. Map batches to findings

For each "Publish" batch, determine which of the published findings
it supports. A batch may support multiple findings. A finding may
draw from multiple batches. This is a many-to-many mapping, and
that is expected.

Create a mapping table (this becomes the basis for `code/README.md`):

```
| Batch | Description | Supports finding(s) |
|-------|-------------|---------------------|
| batch_022 | Donor-level TF-SASP correlations | 1, 2 |
| batch_054 | SCENIC regulon validation | 1, 2, 3 |
| ...   | ...         | ...                 |
```

**If a finding has no clean batch mapping** (e.g., it's primarily
a literature synthesis), note this. The finding still gets a
hypothesis card, but `code/README.md` will note that evidence comes
from literature synthesis rather than a computational experiment.

### 3. Group and rename batches into analysis steps

Group related batches into coherent analysis steps. The goal:
someone reading `code/` sees a logical analysis pipeline, not
arbitrary batch numbers.

Grouping criteria:
- Batches doing the same type of analysis across different
  compartments or datasets → one step
- A batch and its direct rerun/correction → one step (use the
  corrected version's files)
- Batches addressing unrelated questions → separate steps

Name each step descriptively. Prepend a two-digit number for
rough chronological order of the analysis pipeline:

```
01_data_qc_and_preprocessing
02_donor_level_tf_sasp_screen
03_scenic_regulon_validation
04_cross_atlas_replication
05_partial_correlation_decomposition
06_ligand_receptor_crosstalk
07_fap_growth_factor_analysis
08_country_confound_adjustment
```

Rules:
- No gaps in numbering
- No "batch" in the name
- Name describes what the analysis DOES
- Snake_case, lowercase
- If one source batch = one step, just rename it
- If multiple source batches merge into one step, combine their
  contents. If there are filename conflicts, prefix with the source
  batch number (e.g., `b054_correlations.csv`, `b055_correlations.csv`)

### 4. Copy code

For each analysis step, copy from the source batch(es) into
`code/<step_name>/`:

**Copy analysis scripts and any small input files they depend on**
(e.g., gene lists, parameter files). The goal is that someone
reading the code can understand what it does and what it reads.

**Do NOT copy:**
- Execution logs, compiled artifacts, or build outputs
- Internal Marvin planning documents (brief.md, challenge.md,
  preflight.md, design docs)
- Internal configuration files (Marvin's .yaml configs)
- Any documentation that reflects internal brainstorming, planning,
  or process rather than scientific analysis

The test: does this file contribute to understanding or reproducing
the analysis? If yes, include it. If it's internal process
documentation, exclude it.

**Do NOT modify the scripts.** Publish as-run. Do not fix hardcoded
paths, clean up imports, or refactor. The `code/README.md` will
note that scripts reference local data files.

### 5. Copy results

For each analysis step, copy from the source batch(es) into
`results/<step_name>/`:

**Copy:**
- `.csv` files (result tables)
- `.json` files (result data)
- `.png` files generated by analysis scripts (NOT preprint figures)

**Only copy FINAL outputs.** A file is "final" if it is directly
referenced by, or directly supports the interpretation of, a
published finding. A file is "intermediate" if it exists only as
input to another analysis step and is never itself cited in a
finding.

The test: could a reviewer point to this file and say "this is
the evidence behind finding N"? If yes, it's final. If it's a
stepping stone that was consumed to produce something else that
IS cited, it's intermediate.

Examples:
- Donor-level correlation tables → FINAL (cited in hypothesis cards)
- Canonical TF-SASP summary tables → FINAL
- Full differential expression tables → FINAL (gene-level results
  a reviewer might re-analyze)
- Cell-level AUCell matrices (hundreds of MB) → INTERMEDIATE
  (findings cite donor-level summaries derived from these)
- GRNBoost2 raw adjacency matrices → INTERMEDIATE
- Numpy intermediate arrays → INTERMEDIATE

**Do NOT copy:**
- Intermediate outputs (as defined above)
- Raw or large data files that aren't ours to redistribute
  (original datasets, compressed archives)
- Execution logs or error output
- Internal Marvin documentation (see exclusion table below)

**Size guideline:** files must be appropriate for a git repository.
Most final outputs are well under 10MB. Files over 50MB are almost
certainly intermediate and should not be committed. If you believe
a file over 50MB is genuinely final, flag it to the PI.

`code/` and `results/` MUST use the same step names and numbering.
A reviewer traces from `results/03_scenic_regulon_validation/` to
`code/03_scenic_regulon_validation/` to see the script that
produced the output.

### 6. Copy figures

From `docs/preprint/figures/` (or wherever the Marvin workspace
stores publication figures), copy into `figures/`:

**Copy:**
- `figure_*.png` (main publication figures)
- `supplementary_figure_*.png` (supplementary figures)
- Any other image format used (`.svg`, `.pdf` if present)

**Do NOT copy:**
- `render_figure_*.py` (figure generation scripts — internal)
- `_style.py` or similar shared styling modules (internal)
- `figure_*_data/` directories (backing data — internal)
- `__pycache__/`
- `README.md` from the source figures directory (internal)

### 7. Write code/README.md

Create `code/README.md` with:

1. **One-line note**: scripts reference local data files, see
   sources.md for download locations, scripts published as-run.
2. **Analysis step listing**: for each step, a heading with the
   step directory name, a description of what the analysis does,
   and a "Findings supported" line referencing hypothesis numbers.

```markdown
### 01_data_qc_and_preprocessing
HLMA atlas quality control and GRN viability assessment.
- **Findings supported:** Prerequisite for all findings

### 02_donor_level_tf_sasp_screen
Unbiased donor-level Spearman correlation screen of TFs against
SASP composite scores across vascular, FAP, and MuSC compartments.
- **Findings supported:** 1, 2, 3
```

3. **Notes on findings without computational steps** if any (e.g.,
   "Finding 8 is a literature synthesis. Evidence comes from the
   L-R data in step 06 combined with pharmacological literature.")

**No internal jargon in the README.** Do not reference batch
numbers, Marvin iteration numbers, or internal experiment IDs.
The README is public-facing. An external reader should understand
every word. Batch provenance is for internal records only — keep
it in a local note, not in the repo.

### 8. Write hypotheses.md

One card per published finding. Format:

```markdown
### N. [Finding title]

**Prediction:** [Specific, testable statement. One sentence.]

**Evidence:** [Key statistics. Reference analysis steps by number
if helpful, e.g., "step 02, step 03."]

**Confidence:** [Strong / High / Moderate]

**Validation needed:** [What experiment or analysis would confirm
or refute this.]

**Status:** [Awaiting validation / In Discussion / In Progress /
Validated / Refuted]

[Contribute →](https://github.com/IluvatarLabs/iluvatar-open-research/issues/new?template=validation-offer.yml)
```

Cards are separated by `---` horizontal rules.

**Content source**: pull from the website's `_projects/<slug>.md`
summary of discoveries section (for prediction + evidence +
confidence) and open questions section (for validation needed).
Merge each finding with its corresponding validation experiment
into one card.

**Kramdown warning**: if any text contains pipe characters `|`
(common in partial-correlation notation like `rho(X|Y|Z)`), escape
them as `\|` or kramdown will interpret them as table delimiters
and break the rendering.

**For Cycle 2+**: update existing cards (confidence may change,
status may change, evidence summary may strengthen). Add new cards
for new findings. NEVER delete a card. If a finding is retracted,
change status to "Refuted" and add a one-line explanation.

### 9. Write sources.md

One flat file. Organized by cycle. Format:

```markdown
## Cycle 1

| Dataset | Source | Accession / URL | Samples | Used for |
|---------|--------|----------------|---------|----------|
| ... | ... | ... | ... | ... |

### Software
- [List tools, versions]

### Key literature
- [Key cited papers]
```

**Content source**: pull from the website's sources/datasets
section and from any dataset inventory files in the Marvin
workspace (e.g., `docs/cellxgene_muscle_datasets.md`).

**For Cycle 2+**: add a new section at the top. New datasets get
a "Contributed by" column linking to the GitHub issue that
suggested them. Prior cycle datasets carry forward by reference.

### 10. Write manifest.md

One flat file. One section per cycle. Format:

```markdown
## Cycle N

**Date:** YYYY-MM-DD

### Community contributions incorporated
- Issue #N: [what was contributed and how it affected the analysis]
- (or "None" for Cycle 1)

### Datasets used
[Count] datasets. See [sources.md](sources.md) for full provenance.

### Hypotheses published
[Count] hypotheses ([breakdown by confidence]).
See [hypotheses.md](hypotheses.md) for full cards.

### Key limitations acknowledged
- [Bullet list of the most important caveats for this cycle]

### Open questions for the community
[Count] specific validation experiments. See the
[project page](https://iluvatarlabs.github.io/iori/<slug>/#open-questions-for-the-community).
```

**For Cycle 2+**: add a new section at the top. Document what
community input was incorporated, what hypotheses changed, and
what was retracted.

### 11. Update project README.md (if needed)

If this is a **new project**: create `README.md` with:
- Project name
- The research question (one line)
- Current status and cycle
- Link to the full project page on the website
- Contribute links (4 issue template URLs)
- Contact email

If this is an **existing project**: update the status line and
cycle number.

### 12. New project only: additional setup

If this is a new project being added to the repo for the first
time:

1. Create the project directory: `projects/<slug>/`
2. Create a GitHub label: `project:<short-name>`
3. Update the top-level `README.md` project table with the new
   project's name, directory link, question, and website link
4. Update `CONTRIBUTING.md` if the contribution types reference
   specific projects (check the issue template dropdowns)
5. Add the new project to the issue template dropdowns in
   `.github/ISSUE_TEMPLATE/*.yml` (each template has a project
   selector)

### 13. Review locally

Before committing, run these checks:

```bash
# Full file listing — does the structure look right?
find projects/<slug>/ -type f | sort

# No internal Marvin files leaked?
grep -rl "soul\|silmaril\|marvin\.yaml\|vera" projects/<slug>/

# No data files snuck in?
find projects/<slug>/ -name "*.h5ad" -o -name "*.gz" -o -name "*.npz"

# No large files?
find projects/<slug>/ -type f -size +50M

# code/ and results/ have matching step names?
diff <(ls code/) <(ls results/)

# Correct number of hypothesis cards?
grep -c "^### " projects/<slug>/hypotheses.md

# Correct number of figures?
ls projects/<slug>/figures/ | wc -l

# No .log, .pyc, .yaml, brief.md in code/?
find projects/<slug>/code/ -name "*.log" -o -name "*.pyc" -o -name "brief.md" -o -name "*.yaml"

# No __pycache__ anywhere?
find projects/<slug>/ -name "__pycache__"
```

Read `code/README.md` and verify the finding-to-step mapping makes
sense against the hypothesis cards.

**Show the file listing to the PI for approval before committing.**

### 14. Commit and push

One commit per cycle. Message format:

```
<Project Name>: Cycle N

- hypotheses.md: N hypothesis cards (X Strong, Y High, Z Moderate)
- sources.md: N datasets
- manifest.md: Cycle N record
- code/: N analysis steps (mapped to findings in code/README.md)
- results/: corresponding outputs
- figures/: N main + N supplementary
```

For Cycle 2+, the commit message also notes what changed.

**For Cycle 1**: commit directly to main (or to a branch for PR
review — PI's call).

**For Cycle 2+**: always use a PR. The PR template
(`.github/PULL_REQUEST_TEMPLATE.md`) has the structured format:
what changed, new datasets, revised hypotheses, retractions. The
PR sits open for 7 days for community review before merge.

### 15. Tag

After the commit is on main:

```bash
git tag <slug>/cycle-N
git push origin <slug>/cycle-N
```

Previous cycles remain accessible via tags.

---

## What NEVER goes in the repo

The repo is the equivalent of a supplemental data archive for a
scientific publication. The test for every file is semantic: does
this file help a reviewer or collaborator understand, verify, or
extend the published findings? If not, it stays private.

**Commercial IP and agent internals:**
Marvin's behavioral rules (`soul.md`), project configuration
(`marvin.yaml`), internal templates, and tools directories are
commercial IP. Never publish.

**Internal process documents:**
Anything that reflects how the work was managed rather than what
the work found. This includes: iteration logs, work-in-progress
docs, internal reviews, PI briefings, cover letters, strategic
docs, retrospectives, novelty queues, experiment planning docs
(brief.md, challenge.md, preflight.md), and deliverable reviews.
These are the sausage-making. The published findings, hypothesis
cards, and analysis code are the sausage.

Note: Vera (the sub-agent) and references to Marvin's
configuration in code comments are fine. Vera is public knowledge.
What stays private is Vera's internal prompts and deliberation
outputs, not the fact that she exists or that scripts followed
her recommendations.

**Intermediate analysis outputs:**
Files that exist only as input to another step and are never
themselves cited in a finding. The classic examples: cell-level
score matrices (hundreds of MB) where the finding cites the
donor-level summary, raw adjacency lists where the finding cites
module-level correlations, numpy intermediate arrays. If a
reviewer would never look at the file to evaluate a claim, it's
intermediate.

**Data we don't own:**
Original datasets (`.h5ad` files, large compressed archives) are
not ours to redistribute. `sources.md` provides accession numbers
and URLs for obtaining them from the original sources.

**Build artifacts and logs:**
Compiled Python files, `__pycache__/`, execution logs, LaTeX build
outputs, manuscript compilation artifacts. These are reproducible
from the source files and carry no scientific content.

When in doubt: ask "would this appear in a journal's supplemental
materials?" If no, it stays private.

---

## Cycle 2+ differences

Everything above applies to Cycle 1. Key differences for later
cycles:

- **hypotheses.md**: update existing cards, add new cards. Never
  delete. Retracted → status "Refuted" with explanation.
- **sources.md**: new Cycle section at top. New datasets include
  "Contributed by" column linking to the issue that suggested them.
- **manifest.md**: new section at top. Document community
  contributions incorporated (with issue numbers).
- **code/**: add new analysis steps with continuing numbering.
  Existing steps stay. Re-run with corrections → new step
  (e.g., `15_tf_screen_v2_corrected`), don't overwrite.
- **results/**: same as code. New step directories only.
- **figures/**: updated figures replace prior versions. Git history
  preserves the originals.
- **Commit**: always via PR for Cycle 2+. Use the PR template.
  7-day review window before merge.
- **Tag**: `<slug>/cycle-N` after merge.

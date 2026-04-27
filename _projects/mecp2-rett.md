---
title: "Drug repurposing candidates for MECP2-related disorders"
status: "Validation in progress"
iteration: 4
updated: 2026-04-15
started: 2026-02-08
summary: "Network pharmacology combining protein interaction graphs, gene expression, and known drug targets to identify repurposing candidates for Rett syndrome. Two candidates are in wet-lab validation at Emory: H1 preliminary data positive, H2 protocol starting."
tags:
  - Rare disease
  - Drug repurposing
  - Network pharmacology
github_url: "https://github.com/iluvatar-open-research/project-mecp2-rett"
preprint_url: "https://www.biorxiv.org/"
hypothesis_count: 2
dataset_count: 11
sample_count: "890"
---

## Research objective

Identify repurposing candidates — compounds with existing safety profiles in pediatric populations — that could rescue cellular phenotypes in MECP2 loss-of-function contexts. Rett syndrome has no approved disease-modifying therapy; repurposing has the shortest path to clinical testing if a candidate survives wet-lab validation.

We built a pediatric-specific protein-protein interaction graph anchored on MECP2 and the downstream targets it regulates in cortical neurons. Public expression data from 11 datasets (post-mortem brain tissue, iPSC-derived neurons, mouse models) was projected onto the network. Drug-target databases (DrugBank, ChEMBL, Open Targets) were intersected with high-centrality nodes to surface compounds with both **mechanistic plausibility** and **existing safety profiles**.

## Summary of discoveries

1. **[H1 · BDNF signaling potentiation via TrkB partial agonism](#h1--bdnf-signaling-potentiation-via-trkb-partial-agonism)** — high confidence, wet-lab validation in progress at Emory; preliminary data positive at 48 h.
2. **[H2 · mGluR5 negative allosteric modulation reduces hyperexcitability](#h2--mglur5-negative-allosteric-modulation-reduces-hyperexcitability)** — moderate confidence, validation protocol starting. Acute vs. chronic dose-response prediction addresses why prior mGluR5 attempts in Rett gave mixed results.

## Open questions for the community

- **Additional iPSC-derived MECP2-null cortical neuron lines** for H1 dose-response replication. Specifically lines with confirmed MECP2 loss-of-function (frameshift or large deletion) rather than missense.
- **Multi-electrode array (MEA) infrastructure** for H2 chronic-treatment phenotyping. We need at minimum 7-day continuous recordings with controlled drug dosing.
- **Patient samples or clinical data** correlating circulating BDNF or BDNF-pathway markers with Rett severity scores. The H1 hypothesis would predict an inverse correlation but no public dataset has the matched measurements.
- **Negative datasets** — repurposing predictions are noisy. We are particularly interested in datasets where MECP2-targeted compounds *failed* and the failure reason was characterized, so we can encode those failure modes into the next iteration's screen.

If you have any of the above, file an issue on the [project repository][repo] or email iori@iluvatarlabs.com.

[repo]: https://github.com/iluvatar-open-research/project-mecp2-rett

## Findings

The BDNF/TrkB pathway shows consistent suppression in MECP2-loss contexts across **8 of 11 datasets**. Network centrality places TrkB as a primary node downstream of MECP2-regulated gene programs, and MECP2 directly regulates *Bdnf* transcription via methyl-CpG binding at the BDNF promoter region. The pathway has decades of pediatric drug development behind it; partial agonists with safety data exist, lowering translational risk.

> **Caveat.** Repurposing predictions from network pharmacology are inherently noisy. The goal of this project is to surface candidates that *also* have plausible mechanistic stories, then let the wet lab decide. We expect most candidates to fail validation. The two listed here are the ones that survived internal review.

### H1 · BDNF signaling potentiation via TrkB partial agonism

**Confidence: High · Wet-lab validation in progress (Emory)**

**Prediction.** A TrkB partial agonist will rescue dendritic spine density in MECP2-null cortical neurons *in vitro*, measurable as **>25% increase in spine count** at 96 h vs. vehicle control, in lines confirmed for MECP2 loss-of-function.

**Supporting evidence.** BDNF/TrkB pathway suppression in 8 of 11 datasets. Network centrality places TrkB as a primary node downstream of MECP2-regulated gene programs. Preliminary data from Emory shows positive trend at the 48 h timepoint; full dose-response characterization underway.

**What would falsify.** (1) No measurable increase in spine count in MECP2-null lines at any concentration; (2) effect equivalent in MECP2-wild-type control lines (rules out general neurotrophic effect); (3) effect blocked by TrkB-selective antagonist with K562 dose-response control.

**Difficulty.** iPSC-derived cortical neuron culture + spine quantification (image analysis pipeline). 8-12 weeks. Currently underway; H1 preliminary data positive.

### H2 · mGluR5 negative allosteric modulation reduces hyperexcitability

mGluR5 dysregulation appears in **5 of 11 datasets**. The mGluR5 literature in Rett is mixed — multiple prior attempts have given inconsistent results. Marvin's contribution is the specific prediction of dose-response and **chronic vs. acute treatment** effects based on integrated expression-phenotype modeling, which proposes that prior failures may have been due to acute-only dosing missing chronic compensatory effects.

**Confidence: Moderate · Validation protocol starting**

**Prediction.** Selective mGluR5 NAM treatment will reduce hyperexcitability in MECP2-null cortical organoids, measurable by multi-electrode array as **>20% reduction in spike rate** after **7-day chronic exposure**.

**Supporting evidence.** mGluR5 dysregulation in 5 of 11 datasets; cross-references with published Rett syndrome literature show several attempts at this mechanism with mixed results. Marvin's specific contribution: the chronic vs. acute distinction, which prior studies typically did not test.

**What would falsify.** (1) No spike-rate reduction at chronic 7-day timepoint at any tolerated dose; (2) effect equivalent in MECP2-wild-type organoids; (3) acute-only dosing reproduces the prior mixed results, falsifying the chronic-is-different hypothesis.

**Difficulty.** Patient-derived or iPSC-derived cortical organoid culture + multi-electrode array recording. 10-14 weeks total (organoid maturation + 7-day chronic exposure + readout). Validation protocol starting at Emory.

## Sources, datasets, and literature

### Public datasets

- **Allen Brain Atlas Developmental** — reference for cell-type projection.
- **Mouse model RNA-seq panels** — 4 datasets covering Mecp2-null and conditional knockouts.
- **iPSC-derived cortical neuron RNA-seq** — 5 datasets with confirmed MECP2 loss-of-function lines.
- **Post-mortem human brain tissue** — 2 datasets, Rett patients vs. age-matched controls.
- **DrugBank, ChEMBL, Open Targets** — drug-target databases for repurposing intersection.

Total: 11 unique datasets, 890 samples after QC.

### Software

Network pharmacology pipeline: STRINGdb for PPIs, igraph for centrality, custom drug-target intersection scripts. Differential expression: DESeq2 + limma. Cross-dataset meta-analysis: random-effects via metafor.

### Open issues with reviewer concerns

- Internal review flagged that prior mGluR5 attempts in Rett are mixed; iteration 3 added the explicit acute-vs-chronic distinction to the H2 prediction.
- Iteration 4 added 2 newly-published iPSC datasets that strengthened the H1 BDNF/TrkB signal.

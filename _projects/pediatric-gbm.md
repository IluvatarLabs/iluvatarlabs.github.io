---
title: "Multi-omic target prioritization in pediatric high-grade glioma"
status: "Active analysis"
iteration: 2
updated: 2026-04-20
started: 2026-03-12
summary: "Integrating transcriptomic, proteomic, and methylation data across 14 public datasets (2,660 pediatric brain tumor samples) to identify therapeutic targets for a disease with a 5-year survival under 20% and no approved targeted therapies."
tags:
  - Oncology
  - Pediatric
  - Multi-omic
github_url: "https://github.com/iluvatar-open-research/project-pediatric-gbm"
preprint_url: "https://www.biorxiv.org/"
hypothesis_count: 3
dataset_count: 14
sample_count: "2,660"
---

## Research objective

Identify druggable molecular targets in pediatric high-grade glioma (pHGG) by integrating transcriptomic, genomic, proteomic, and epigenomic data across all available public pediatric brain tumor cohorts. The disease has a 5-year survival under 20% and **no approved targeted therapies**; many adult-GBM-derived targets fail in pediatric trials because pediatric biology differs.

We used:

- Bulk RNA-seq + microarray expression across 7 datasets
- Whole-genome sequencing across 2 datasets
- TMT mass-spectrometry proteomics across 2 datasets
- 450k methylation array across 3 datasets
- Single-cell RNA-seq across 8 datasets (cell-type deconvolution)

After QC, batch correction, and cross-platform normalization, the goal was to surface pediatric-specific candidates with multi-omic support and falsifiable validation paths.

## Summary of discoveries

1. **[H1 · PDGFRA-STAT3 axis as therapeutic vulnerability in PDGFRA-amplified pHGG](#h1--pdgfrastat3-axis-as-therapeutic-vulnerability-in-pdgfra-amplified-phgg)** — high confidence, awaiting validation.
2. **[H2 · CXCL12/CXCR4 chemokine signaling drives tumor-associated macrophage infiltration](#h2--cxcl12cxcr4-chemokine-signaling-drives-tumor-associated-macrophage-infiltration)** — moderate confidence, awaiting validation.
3. **[H3 · Targetable residual EZH2 activity in H3K27M-mutant pediatric GBM](#h3--targetable-residual-ezh2-activity-in-h3k27m-mutant-pediatric-gbm)** — moderate confidence, in discussion.
4. **[Pediatric-specific mechanistic contexts not captured in adult-GBM literature](#pediatric-specific-mechanistic-contexts)** — for all three targets, the proposed contexts differ meaningfully from how the same genes are described in adult GBM.

## Open questions for the community

The hypotheses below have specific, falsifiable predictions. We are looking for:

- **Wet-lab labs with pediatric GBM cell lines (SF188, KNS42, HSJD-DIPG007).** H1 in particular needs only standard 2D culture + Western blot + viability — 3-5 weeks of work. We have funding to support reagents for the first lab to commit.
- **Patient-derived organoid groups for H2.** Co-culture of pediatric GBM organoids with monocyte-derived macrophages plus AMD3100. Flow cytometry endpoints. 6-8 weeks.
- **ChIP-seq capacity for H3.** The exploratory hypothesis predicts 847 specific loci that retain H3K27me3 in H3K27M-mutant cells. Validating this requires careful locus-level annotation using updated reference (raised by Dr. Chen, Harvard, currently being incorporated into iteration 3).
- **Datasets we missed.** Particularly non-English repositories, recent preprint-adjacent deposits, and any pediatric-specific proteomics or single-nucleus data.

If you have any of the above, file an issue on the [project repository][repo] or email iori@iluvatarlabs.com.

[repo]: https://github.com/iluvatar-open-research/project-pediatric-gbm

## Findings

Marvin integrated transcriptomic (RNA-seq, microarray), genomic (whole-genome sequencing), proteomic (TMT mass spectrometry), and epigenomic (methylation array) data from 14 public datasets encompassing 2,660 pediatric brain tumor samples. After quality control, batch correction, and cross-platform normalization, differential analysis identified **2,340 genes with consistent dysregulation** across at least 3 independent datasets.

Network analysis using curated protein-protein interaction data narrowed this to **23 high-centrality candidates** with evidence of functional relevance in the pediatric-specific subnetwork. Three of these — *PDGFRA*, *CXCR4*, and *EZH2* — emerged with sufficient multi-omic support to formalize as testable hypotheses.

> **Caveat.** These are computational predictions derived from integrative analysis of public data. None have been experimentally validated in the specific contexts proposed here. The hypotheses below represent the strongest candidates from the current iteration, not confirmed findings. We actively invite scrutiny — particularly on dataset selection, batch correction choices, and the network centrality filtering threshold.

### H1 · PDGFRA–STAT3 axis as therapeutic vulnerability in PDGFRA-amplified pHGG

**Confidence: High · Awaiting validation**

**Prediction.** Pharmacological inhibition of PDGFRA in PDGFRA-amplified pediatric GBM cell lines will reduce STAT3 phosphorylation and suppress proliferation. Specifically: crenolanib IC₅₀ < 1 μM in SF188 and KNS42 lines; >60% reduction in pSTAT3 (Y705) at 24 h; >50% viability reduction at 72 h (CellTiter-Glo).

**Supporting evidence.** Consistent overexpression of PDGFRA across **11 of 14 datasets**. Network analysis places PDGFRA as a hub node upstream of STAT3 in the pediatric-specific subnetwork (centrality rank 4). Two independent proteomic datasets confirm elevated pSTAT3 in PDGFRA-amplified samples (log2FC 1.8, FDR < 0.01).

**What would falsify.** (1) Crenolanib IC₅₀ > 5 μM in both lines; (2) no measurable reduction in pSTAT3 at sub-cytotoxic concentrations; (3) effect equivalent in PDGFRA-wild-type control lines.

**Difficulty.** Standard mid-sized academic lab (2D culture, WB, viability). Estimated 3-5 weeks. No animal work required.

### H2 · CXCL12/CXCR4 chemokine signaling drives tumor-associated macrophage infiltration

**Confidence: Moderate · Awaiting validation**

**Prediction.** CXCR4 antagonism (AMD3100 / plerixafor) will reduce tumor-associated macrophage infiltration in pediatric GBM organoid models, measurable as >30% reduction in CD68⁺/CD163⁺ population by flow cytometry at 72 h co-culture.

**Supporting evidence.** scRNA-seq deconvolution identifies CXCL12-high tumor cells co-localized with M2-polarized macrophages in **5 of 8 single-cell datasets**. Bulk transcriptomic signature correlates with worse progression-free survival (HR 1.8, 95% CI 1.15-2.81, *p* = 0.03). *Limitation:* interaction is inferred from expression correlation, not direct functional evidence.

**Difficulty.** Lab with organoid culture + flow cytometry. 6-8 weeks. Patient-derived material preferred; immortalized lines acceptable fallback.

### H3 · Targetable residual EZH2 activity in H3K27M-mutant pediatric GBM

**Confidence: Moderate · In discussion**

**Prediction.** EZH2 inhibitor (tazemetostat) will selectively reduce viability in H3K27M-mutant pediatric GBM lines vs. H3-wild-type lines, with differential sensitivity >5-fold (IC₅₀ comparison).

**Supporting evidence.** Methylation array analysis shows global H3K27me3 loss in H3K27M samples — consistent with existing literature. However, Marvin's locus-specific analysis identified **847 regions that paradoxically retain or gain H3K27me3**, enriched for developmental transcription factors. This suggests a targetable dependency on residual EZH2 activity not visible at the global level.

This is the **most exploratory** of the three hypotheses. Dr. Chen (Harvard) raised concerns about locus annotation; iteration 3 is re-running with updated reference.

### Pediatric-specific mechanistic contexts

A repeated pattern across all three targets: each gene has well-established roles in **adult** GBM, but the **mechanistic context** identified here is pediatric-specific and not well-characterized in existing literature.

For PDGFRA, the adult literature emphasizes ligand-independent signaling in proneural GBM. The pediatric data instead points to a STAT3 axis activated downstream of PDGFRA amplification, with proteomic confirmation of pSTAT3(Y705) elevation in samples that wouldn't be flagged as PDGFRA-driven by transcriptomics alone.

For CXCR4, adult studies focus on tumor cell migration and invasion. The pediatric signal here is **immune-cell coupling**: CXCL12-high tumor cells co-locate with M2-polarized macrophages, suggesting CXCR4 antagonism could reshape the tumor microenvironment rather than directly target tumor cells.

For EZH2, the adult literature treats H3K27M as causing global PRC2 inhibition. The pediatric methylation data shows that **the global signal hides a locus-specific story** — 847 regions retain or gain H3K27me3, enriched for developmental TFs. This suggests the residual EZH2 activity is what's actually doing the work in maintaining the H3K27M phenotype.

The full analysis pipeline, intermediate outputs, and reasoning artifacts are available in the project repository. Marvin's logic trail — every filter decision, parameter choice, and reasoning step — is reproducible end-to-end via the included Dockerfile.

## Sources, datasets, and literature

### Public datasets

- **CBTTC Pediatric Brain Tumor Atlas** — RNA-seq + WGS, 943 samples. Primary integrated cohort.
- **HERBY Clinical Trial Transcriptomics** — Microarray, 120 samples (added in iteration 2 from community contribution).
- **St. Jude Cloud Pediatric Cancer** — RNA-seq + WGS + Methylation, 312 samples. Cross-modality within-sample integration.
- **Allen Brain Atlas (Developmental)** — scRNA-seq reference for cell-type deconvolution.
- **CPTAC Pediatric Brain Tumor Pilot** — Proteomics (TMT), 218 samples. Re-processed after batch correction.
- **Mackay et al. 2017 Methylation** — 450k methylation array, 1,067 samples. Used for H3K27me3 locus analysis.

### Software

Standard differential expression + network analysis pipeline using R/Bioconductor (DESeq2, limma, ComBat for batch correction). Network centrality via igraph on STRINGdb-curated PPIs.

### Open issues with reviewer concerns

- Issue #14 — Dr. Patel (UCL): missing HERBY data. **Resolved in iteration 2.**
- Internal review (Apr 11): batch effect in CPTAC proteomics. **Resolved with ComBat re-run.**
- Dr. Chen (Harvard): H3K27me3 locus annotation reference outdated. **In progress, iteration 3 will use updated reference.**

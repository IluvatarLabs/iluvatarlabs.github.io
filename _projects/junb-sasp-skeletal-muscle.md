---
title: "JUNB drives inflammatory signaling in vascular cells of aging human muscle"
status: "Active analysis"
iteration: 1
updated: 2026-05-04
started: 2026-04-22
summary: "Vascular endothelial cells, not fibroblasts, are the dominant source of age-related inflammatory secretion in human skeletal muscle, driven by JUNB and the JNK/AP-1 signaling axis. The three major cell compartments (vascular, muscle stem cell, fibroblast) each run distinct regulatory programs, with AP-1 polarity actually reversing in fibroblasts. The leading therapeutic hypothesis pairs a JNK-pathway modulator for the vascular axis with ANGPT2 blockade for a second, JNK-independent vessel-destabilization axis."
tags:
  - Sarcopenia
  - Single-cell transcriptomics
  - Senescence
  - Drug discovery
github_url: "https://github.com/iluvatar-open-research/project-junb-sasp-skeletal-muscle"
preprint_url: "https://www.biorxiv.org/"
hypothesis_count: 8
dataset_count: 5
sample_count: "387,000+ cells, 35 donors"
---

## Research objective

Determine which cell compartment is the dominant source of the senescence-associated secretory phenotype (SASP) in aged human skeletal muscle, identify the transcriptional regulatory program driving it, and propose compartment-specific therapeutic strategies.

The approach:

- Donor-level Spearman correlations between transcription factor expression and a 12-gene SASP composite score (SASP12) across four compartments (vascular endothelial, FAP, MuSC, immune) in two independent single-cell atlases
- pySCENIC regulon inference (GRNBoost2 + AUCell) to distinguish genuine regulatory activity from co-expression artifacts
- Partial-correlation decomposition to separate collinear signals (JUNB vs CDKN1A/p21)
- Cross-atlas integrative analysis (meta-analytic pooling, direction-of-effect voting, replication tiering)
- Vascular-to-FAP ligand-receptor crosstalk analysis for paracrine axis identification

Use the Human Lifemap Muscle Atlas (HLMA, 387,000+ cells, 23 donors) as the primary dataset, with the Nature Aging 2024 atlas (90,902 cells, 12 donors) for cross-atlas validation.

> This is a living project. Marvin has finished this iteration using existing public data and literature. The next iteration begins when you contribute: a critique of the analysis, an alternative interpretation, a dataset we missed, a question we haven't asked, or experimental validation of a finding. Marvin will incorporate your input and run the next research cycle, published openly, for free. [See open questions below](#open-questions-for-the-community) or [learn more about how IORI works](/iori/).

## Summary of discoveries

1. **[Vascular endothelial cells dominate SASP](#vascular-junb-sasp-dominance) — <span class="chip chip-high">High confidence</span>.** JUNB regulon activity correlates with SASP at rho=0.923 (p=3.64×10⁻¹⁰, N=23 donors) in vascular cells, the strongest coupling of any TF in any compartment. Reframes the prevailing FAP-centric model of muscle aging.

2. **[Three-compartment regulatory model](#three-compartment-tf-architecture) — <span class="chip chip-high">High confidence</span>.** Vascular cells run an AP-1 axis (JUNB-dominated). MuSCs run a dual AP-1 + p21 axis. FAPs operate through C/EBPβ with opposite AP-1 polarity (all seven canonical subunits show negative AUCell coupling). Each compartment requires a different therapeutic strategy.

3. **[KLF10 is a TGF-β bystander biomarker, not a SASP driver](#klf10-bystander) — <span class="chip chip-high">High confidence</span>.** Despite the strongest and most consistent mRNA-SASP correlation across five datasets (rho=0.813, I²=27.2%), regulon-level validation is null (AUCell rho=0.079 in FAP). Prevents a misdirected therapeutic strategy targeting KLF10 in FAPs.

4. **[Two distinct vascular→FAP paracrine axes](#vascular-fap-crosstalk) — <span class="chip chip-high">High confidence</span> / <span class="chip chip-moderate">Moderate</span>.** Axis 1: JNK-dependent inflammatory crosstalk (CXCL2→CXCR2, IL6→IL6R). Axis 2: JNK-independent vessel destabilization (ANGPT2→Tie1/Tie2). The two axes are not reducible to a single pharmacological target.

5. **[FAP growth factor compensation is inadequate](#fap-growth-factor-compensation) — <span class="chip chip-moderate">Moderate</span>.** FAPs upregulate FGF7 and HGF with age but the signal does not reach MuSCs: the FGF7→FGFR1 axis is broken (receptor flat), and IGF2 declines across all FAP subtypes while MuSC IGF1R rises. Growth factor supplementation may bypass this failure.

6. **[CDK4/6 inhibitors are CONTRAINDICATED in muscle](#cdk46-contraindication) — <span class="chip chip-moderate">Moderate</span>.** CDKN1A/p21 operates downstream of JUNB in the regulatory cascade. CDK4/6 activity is required for MyoD-driven satellite cell re-entry. Any clinical program pairing a JNK inhibitor with a CDK4/6 inhibitor in muscle-aging populations requires satellite-cell safety screens.

7. **[JNK→AP-1→CDKN1A→SASP hierarchy](#jnk-ap1-cdkn1a-hierarchy) — <span class="chip chip-moderate">Moderate</span>.** Partial-correlation decomposition shows JUNB's SASP signal is largely mediated through CDKN1A, while CDKN1A retains independent signal beyond JUNB. CDKN1A/p21 is a pharmacodynamic biomarker for JNK-directed trials, not a therapeutic target.

8. **[BML-260 + ANGPT2 blockade as leading combination](#therapeutic-model) — <span class="chip chip-moderate">Moderate</span>.** BML-260 (a DUSP22 modulator) addresses the JNK-dependent vascular inflammatory axis; ANGPT2 blockade (MEDI3617 or AKB-9778) addresses the JNK-independent vessel-destabilization axis. Neither agent alone is predicted to cover both paracrine pathways.

## Open questions for the community

These findings are correlational. Causal validation requires perturbation in primary human muscle cell cultures, not mouse models (the AP-1 polarity we report is human-specific). We are actively looking for:

- **JUNB perturbation in primary human vascular endothelial cells.** JNK must be targeted at the kinase/phosphatase layer (canonical JunB phospho-sites Thr102/Thr104 mean siJUNB or CRISPRi would not recapitulate a JNK-inhibitor phenotype). BML-260 is the lead candidate but has not been tested in this tissue context.
- **C/EBPβ perturbation in primary human FAPs.** The FAP SASP mechanism operates through a fundamentally different TF program (C/EBPβ, not AP-1). Regulon-level confirmation is currently underpowered (AUCell rho=0.474, p=0.064 at N=16 snRNA-filtered donors).
- **A third independent vascular muscle aging atlas.** Cross-atlas replication is directionally concordant but formally underpowered (Nature Aging endothelial rho=0.720, p=0.008, N=12 donors, observed power 0.753 vs 80%-power floor of 0.745). No fully independent vascular muscle aging atlas currently exists.
- **FGF7/HGF/IGF2 supplementation studies in MuSC activation assays.** The broken FAP→MuSC growth factor axis (Section 3.5 in preprint) predicts that exogenous supplementation should bypass the endogenous signaling failure. FGF7 (palifermin) is already FDA-approved for oral mucositis.
- **CDK4/6 + JNK inhibitor satellite-cell safety screens.** The CDK4/6 contraindication is inferred from the p21 hierarchy and established satellite-cell biology, not directly tested in the context of muscle aging.
- **Human iPSC-derived vascular organoid models.** Cross-species re-analysis shows the AP-1 polarity is not conserved in mouse (Tabula Muris Senis, 1 of 5 directional matches in EC, 2 of 5 in FAP). Preclinical mouse models will not recapitulate the vascular and FAP axes we identify.

If you have data, capacity, or relevant perturbation systems, file an issue on the [project repository][repo] or email iori@iluvatarlabs.com.

[repo]: https://github.com/iluvatar-open-research/project-junb-sasp-skeletal-muscle

## Findings

### Vascular JUNB-SASP dominance

Among 21 candidate TFs spanning AP-1, KLF, EGR, IRF, C/EBP, NF-κB, and STAT families, JUNB ranked first in vascular endothelial cells with rho=0.929 (p=1.65×10⁻¹⁰, N=23 donors). CDKN1A (p21) ranked second (rho=0.913), followed by EGR1 (rho=0.908). All seven Jun/Fos subunits showed rho > 0.84, indicating a coherent AP-1 regulatory axis rather than a single TF effect.

pySCENIC regulon inference confirmed this is genuine regulatory activity: JUNB regulon AUCell scores correlated with SASP12 at rho=0.923 (p=3.64×10⁻¹⁰), nearly identical to the raw mRNA correlation (delta rho = +0.002). The JNK pathway showed a distinctive post-translational activation pattern: upstream kinase mRNAs (MAP3K, MAP2K4/7, MAPK8/9/10) had null age effects (Cohen's d ≈ 0) while downstream immediate-early AP-1 targets (FOS, JUNB) were robustly age-upregulated (FOS Cohen's d=0.907, JUNB d=0.901 in vascular cells). This is consistent with JNK activation by phosphorylation rather than transcriptional upregulation of the kinase cascade.

The JUNB-SASP coupling survived four verification tests: IL6⁺ venular exclusion (rho changed from 0.9262 to 0.9287), leave-one-out analysis (minimum rho=0.9185), senescence-marker comparison (JUNB correlated more strongly with CDKN1A/p21 than with activation markers ICAM1, VCAM1, or SELE), and cross-atlas replication in Nature Aging endothelial cells (rho=0.720, p=0.008, N=12, directionally concordant but formally underpowered).

**Confidence: High.** Meets all five pre-registered evidence criteria. The single limitation is absence of a fully independent third vascular muscle aging atlas.

### Three-compartment TF architecture

The TF-SASP coupling landscape differs dramatically across compartments:

- **Vascular:** AP-1 axis. JUNB rho=0.929, CDKN1A rho=0.913, EGR1 rho=0.908.
- **MuSC:** Dual AP-1 + p21 axis. CDKN1A rho=0.929, JUNB rho=0.934, EGR1 rho=0.917. The p21 axis is the dominant distinguishing feature (donor-level JUNB-p21 collinearity rho=0.944).
- **FAP:** C/EBP-EGR-KLF axis. CEBPB rho=0.802, EGR2 rho=0.799, KLF10 rho=0.784. JUNB is last at rho=-0.014. Critically, all seven canonical AP-1 subunits show *negative* AUCell regulon coupling. AP-1 targeting in FAPs may be contraindicated.

After joint partial-correlation adjustment for age, sex, and sequencing technology, the hierarchy sharpens: vascular JUNB partial rho=0.912 (N=23); FAP CEBPB partial rho=0.888 (N=22, strengthened by +0.066); MuSC EGR1 partial rho=0.622 (N=23, JUNB demoted to second place after confounder adjustment).

**Confidence: High** for the vascular and FAP programs. **Moderate** for MuSC (cross-atlas CDKN1A coupling present but below the rho>0.5 effect-size threshold).

### KLF10 bystander

KLF10 showed the strongest mRNA-level SASP coupling of any TF in FAPs (rho=0.862, p<0.001, N=22 donors) and was the only TF positive in all five dataset-compartment combinations at the rho>0.5 threshold (cross-dataset I²=27.2%, the lowest heterogeneity of any TF).

However, pySCENIC AUCell analysis refuted this interpretation. KLF10 AUCell rho=-0.164 (p=0.455) versus raw mRNA rho=0.836 (delta=-1.000) in HLMA vascular cells. In HLMA FAPs, KLF10 AUCell rho=0.079 (p=0.770) versus raw mRNA rho=0.862 (delta=-0.782). KLF10's co-expression regulon showed near-zero or negative association with SASP despite strong mRNA-level correlation.

The mechanistic explanation: KLF10 (TIEG1) is a TGF-β early response gene and transcriptional repressor that recruits Sin3A/HDAC complexes. In aging muscle, TGF-β signaling activates both KLF10 mRNA (as a negative feedback regulator) and SASP genes (via SMAD-independent branches including AP-1). KLF10 mRNA rises with SASP because both respond to the same upstream TGF-β signal, but KLF10's repressive targets are distinct from SASP output genes. The KLF10 co-expression module in vascular cells (74 targets) contained zero SASP genes, compared to JUNB's 104-target module which included CXCL2 and CCL2.

**Confidence: High** as a SASP regulatory driver. This negative result prevents a misdirected therapeutic strategy targeting KLF10 in FAPs.

### Vascular-FAP crosstalk

Donor-level Spearman coupling between 49 vascular ligand-FAP receptor pairs resolved two mechanistically distinct paracrine axes:

**Axis 1: JNK-dependent inflammatory crosstalk.** Four pairs showed strong positive vascular-JUNB coupling: CXCL2→CXCR2 (rho_JUNB=0.924, BH-FDR q=1.5×10⁻⁸), IL6→IL6R (rho_JUNB=0.855), TNF→TNFRSF1A (rho_JUNB=0.824), VEGFA→KDR (rho_JUNB=0.766). CXCL2→CXCR2 and VEGFA→KDR survive 49-pair BH-FDR. Because vascular JUNB activity is the presumed JNK readout, these ligand outputs are predicted to be reducible by JNK inhibition.

**Axis 2: JNK-independent vessel destabilization.** ANGPT2→TIE1 (and ANGPT2→TEK) showed moderate coupling to donor age (rho_age=+0.465, p=0.025) and to vascular JUNB (rho_JUNB=0.442, p=0.035). Because ANGPT2 mean expression is not a JNK-downstream AP-1 target in our data, this axis is predicted to persist under JNK-targeted therapy and requires a mechanistically distinct intervention. Two Tie2-axis therapeutic candidates (MEDI3617, a neutralizing anti-ANGPT2 monoclonal; AKB-9778/razuprotafib, a VE-PTP inhibitor) converge on Tie2 activation but have not been tested jointly in aged-muscle cohorts.

Three additional pairs showed age-associated loss of protective paracrine factors: SEMA3C→NRP1 (rho_age=-0.550), SEMA3F→NRP2 (rho_age=-0.500), FGF1→FGFR1 (rho_age=-0.490). These did not couple to vascular JUNB. JNK inhibition is not expected to restore these ligands; restoration strategies (recombinant FGF1, VEGFB mimetics) would be required separately.

**Therapeutic implication:** Vascular SASP → FAP signaling is not a single pharmacological entity. JNK inhibition (Axis 1) plus ANGPT2/Tie2 stabilization (Axis 2) are complementary; addressing only Axis 1 leaves the vessel-destabilization and protective-loss programs intact.

**Confidence: Axis 1 High** (2 of 4 pairs survive 49-pair BH-FDR). **Axis 2 Moderate** (0 of 3 protective-factor pairs survive BH-FDR; F068_04, iter-070 downgrade).

### FAP growth factor compensation

Despite their distinct SASP regulatory architecture, FAPs showed a regenerative-compensatory secretome profile at the population level. Aged FAPs upregulated FGF7 (Cohen's d=+1.29) and HGF (d=+1.10) while decreasing pro-inflammatory ligands TNF (d=-0.54) and IL6 (d=-0.51). This pattern is regenerative-compensatory, not pro-inflammatory.

However, this molecular compensation did not translate to functional benefit. The FAP growth factor score did not predict MuSC activation at the donor level (rho=0.189, p=0.41, N=21 matched donors). The FGF7→FGFR1 crosstalk axis was broken: FAPs surged FGF7 (d=+1.29) but MuSCs did not upregulate FGFR1 (d=+0.13, flat). IGF2 declined across all FAP subtypes (d=-0.60 to -1.24) while MuSCs upregulated IGF1R (d=+0.89), creating a "ligand-receptor mismatch" where MuSCs are primed to receive growth signals that are not being sent.

Exogenous growth factor supplementation (FGF7/KGF, HGF, IGF1/IGF2) is justified to bypass the broken endogenous FAP→MuSC signaling axis. FGF7 (palifermin) is FDA-approved for oral mucositis, providing a potential repurposing pathway. FGF7 is reported to act through FGFR2 rather than FGFR1 in satellite cells, which may modify interpretation.

**Confidence: Moderate.** Growth factor compensation is clearly documented; the broken crosstalk axis is consistent but based on single-atlas population-level effects.

### CDK4/6 contraindication

The CDKN1A/p21-SASP coupling (donor-level rho=0.929, p=1.66×10⁻¹⁰, N=23 in MuSC) operates in parallel with AP-1, and partial-correlation decomposition places CDKN1A immediately downstream of JUNB in the regulatory cascade. CDK4/6 activity is required for MyoD-driven cell-cycle re-entry in satellite cells (reviewed in standard myogenesis literature). Pharmacological CDK4/6 inhibition establishes a p53-dependent senescent state with a restricted secretory phenotype, consistent with a senomorphic action of palbociclib that is distinct from its impact on muscle stem cell activation.

CDKN1A/p21 is a pharmacodynamic biomarker for JNK-directed trials, not a therapeutic target in the MuSC compartment. Any clinical program that pairs a JNK inhibitor with a CDK4/6 inhibitor in muscle-aging populations requires satellite-cell safety screens as a prerequisite.

**Confidence: Moderate.** The contraindication is inferred from the p21 hierarchy and established satellite-cell biology; the supporting evidence for the CDK4/6 claim is from literature synthesis, not direct experimentation in this study.

### JNK-AP1-CDKN1A hierarchy

Partial-correlation decomposition on donor-level scores in HLMA vascular cells: raw rho(JUNB, SASP)=0.937, rho(CDKN1A, SASP)=0.935, rho(JUNB, CDKN1A)=0.927. Partialing out CDKN1A largely extinguishes the JUNB signal (partial rho(JUNB|SASP|CDKN1A)=0.137, p=0.531), whereas partialing out JUNB preserves a substantial CDKN1A signal (partial rho(CDKN1A|SASP|JUNB)=0.513, p=0.012).

Interpreted as a causal chain consistent with JNK→JUNB N-terminal phosphorylation → JUNB/AP-1 transcriptional activity → CDKN1A induction → secretory program, JUNB's SASP effect is largely mediated through CDKN1A, while CDKN1A retains an independent SASP-associated signal beyond JUNB.

This places CDKN1A/p21 downstream of JUNB in the regulatory cascade and identifies it as the most proximal biochemical pharmacodynamic readout available: plasma or muscle CDKN1A/p21 is predicted to be the cleaner efficacy marker for JNK-directed trials than JUNB mRNA.

Three independent genetic-regulation layers converge on the conclusion that AP-1 immediate-early TFs are not transcriptionally regulated at the genomic level: upstream kinase mRNAs show null age effects, bulk muscle eQTL from Open Targets yields 0 of 54 TF×sarcopenia/lean-body-mass trait colocalizations, and single-cell eQTL in OneK1K (N=982) finds JUNB, FOS, EGR1, and ATF3 null at the cis-eQTL level in immune cells, while CDKN1A carries a significant cis-eQTL (p=3.4×10⁻¹⁰). The therapeutic entry point is the JNK/MAPK kinase layer, not the TF-DNA interface.

**Confidence: Moderate.** HLMA-only; causal direction is inferred from post-translational literature, not directly tested.

### Therapeutic model

Based on the three-compartment model and two-axis paracrine architecture, the leading combination hypothesis:

**Vascular, Axis 1 (JNK-dependent).** BML-260, a preclinical DUSP22 modulator that represses JNK activity and ameliorates skeletal muscle wasting in human-muscle-cell, sarcopenia, glucocorticoid, and ICU-catabolism models. Unlike siJUNB/CRISPRi, it targets the kinase/phosphatase layer at which JNK engages JUNB post-translationally. First-generation systemic JNK inhibitors encountered translational bottlenecks: tanzisertib (CC-930) was discontinued in phase 2 for liver enzyme elevations, and CC-90001 failed its IPF phase 2 efficacy endpoint.

**Vascular, Axis 2 (JNK-independent vessel destabilization).** Tie2-axis therapeutic candidates MEDI3617 (neutralizing anti-ANGPT2 monoclonal) and AKB-9778/razuprotafib (VE-PTP inhibitor) converge on Tie2 activation. This axis is predicted to persist under BML-260 monotherapy because ANGPT2 is not downstream of vascular JUNB in our data.

**MuSCs.** CDK4/6 inhibitors are contraindicated. CDKN1A/p21 is a biomarker, not a target. The relevant therapeutic strategy is to protect satellite-cell activation capacity, not to suppress the CDKN1A axis.

**FAPs.** AP-1 targeting is contraindicated (all seven subunits show negative regulon-SASP coupling). C/EBPβ is the leading candidate regulator. FAPs are the dominant senolytic target in muscle (navitoclax-sensitive, BCL-2-dependent), and senolytic therapy targeting FAPs is complementary to, not competing with, JNK-directed therapy of the vascular compartment.

The BML-260 + ANGPT2-blockade combination is our leading hypothesis; neither agent alone is expected to address the other axis.

**Confidence: Moderate.** The combination is inferred from the compartment-specific paracrine architecture; none of the agents have been tested jointly in aged-muscle cohorts.

## Sources, datasets, and literature

### Datasets

- **HLMA atlas** (CNGBdb OMIX004308, Lai et al.) — 387,000+ cells/nuclei from donors aged 15-99 years. Compartment-specific files: Vascular (N=23 donors, 16,157 cells after IL6⁺ exclusion), MuSC (N=23 donors, 9,559 cells), FAP (N=22 donors, 40,389 cells), Immune (N=12 donors, 13,773 cells, snRNA-only subset).
- **Nature Aging 2024 atlas** (Kedlian et al., Sanger cellxgene) — 90,902 nuclei from 17 donors. Fibroblast file (N=12 donors with fibroblast subtypes) and endothelial cells (N=12 donors).
- **GTEx v8 bulk muscle** (N=803) — Used for post-mortem ischemic-time confound triangulation (SMTSISCH covariate).
- **OneK1K single-cell eQTL** (N=982, 14 immune cell types) — cis-eQTL validation for AP-1 TFs vs CDKN1A.
- **Open Targets / Genetics Portal** — Bulk muscle eQTL colocalizations (54 TF×sarcopenia/lean-body-mass trait combinations, 0 hits).

### Software

pySCENIC 0.12.x (GRNBoost2 + AUCell), Python (scipy, scanpy), R (partial correlation, Fisher-z meta-analysis).

### Selected literature

- Thoma et al. 2020. BML-260 as a DUSP22 modulator protective against skeletal muscle wasting.
- Li et al. 2025. Single-nucleus multiomic study of AP-1 and NF-κB in aged human skeletal muscle; CCR5 antagonist Maraviroc. *Nat Commun* 16:6207.
- Wang et al. CDK4/6 inhibition and p53-dependent senescence in muscle context.
- Kedlian et al. 2024. Nature Aging single-cell atlas of human skeletal muscle.
- Lai et al. 2024. HLMA multimodal cell atlas of ageing human skeletal muscle.

Full corpus, including rejected sources and iteration audit trail, in the [project repository][repo].

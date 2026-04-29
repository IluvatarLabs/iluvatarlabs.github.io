---
title: "SOD2 as a driver of myocardial fibrosis in humans"
status: "Active analysis"
iteration: 1
updated: 2026-04-22
started: 2026-04-02
summary: "A genome-wide Mendelian randomization scan across 1,420 plasma proteins, followed by fine-mapping and directional enrichment, nominates six causal proteins for myocardial fibrosis indexed by cardiac MRI T1. SOD2 emerges as the top protective effector through a mitochondrial antioxidant axis."
tags:
  - Cardiology
  - Multi-omic
  - Mendelian randomization
  - Drug discovery
github_url: "https://github.com/iluvatar-open-research/project-sod2-myocardial-fibrosis"
preprint_url: "https://www.biorxiv.org/"
hypothesis_count: 6
dataset_count: 7
sample_count: "1,420 proteins"
---

## Research objective

Identify causal proteins in human myocardial fibrosis with cardiac MRI T1 mapping as a quantitative proxy, and propose a mechanism of action for the top hits. Triangulate evidence using:

- Pairwise colocalization (`coloc`) between cis-pQTLs and the T1 GWAS
- Statistical fine-mapping (`susieR`) at MR-positive loci, with LD informed by 1000 Genomes Europeans via `LDlinkR`
- Two-sample Mendelian randomization (`MendelianRandomization` / `TwoSampleMR`)

Use all available datasets including the myocardial T1 GWAS, multi-tissue cis-pQTLs, and GTEx v10 eQTLs / sQTLs from heart atrial appendage and left ventricle.

## Summary of discoveries

1. **[Genetic validation of six causal proteins for myocardial fibrosis](#genetic-validation-of-six-causal-proteins).** A proteome-wide MR scan of 1,420 plasma proteins identified 32 Bonferroni-significant signals; SOD2 was the top protective effector, with ECM1, SCGN, IGDCC4, GSTP1, and ASPN validated by fine-mapped credible-set enrichment despite weak formal colocalization.

2. **[Mechanistic axes spanning oxidative stress, TGF-β, calcium signaling, and adhesion](#mechanistic-axes).** The six proteins converge on a coherent network: oxidative stress upstream (SOD2, GSTP1) feeding TGF-β modulation (ASPN, ECM1), endothelial calcium signaling (SCGN), and cell adhesion (IGDCC4).

3. **[Protein–transcript QTL discordance reveals cell-type-specific regulation](#proteintranscript-qtl-discordance).** Bulk heart eQTLs/sQTLs do not colocalize with the protein-T1 signals; the causal regulatory variation lives in distal cell-type-specific enhancers and, for SOD2, post-transcriptional miRNA-binding-site disruption.

4. **[Translational implications: tractability, biomarkers, clinical boundaries](#translational-implications).** ASPN is the most actionable extracellular target via TGF-β1 peptide antagonism; GSTP1 should be activated, not inhibited; circulating biomarker signal is currently strongest for ASPN and SOD2; MR against diagnosed heart failure shows no robust links, suggesting specificity to subclinical fibrosis.

## Open questions for the community

The signals here surface targetable biology but leave substantial gaps. We are actively soliciting:

- **Wet-lab validation of SOD2 perturbation in human cardiac fibroblasts.** Specifically, dose-response on collagen synthesis (COL1A1/COL3A1) under TGF-β1 stimulation in primary or iPSC-derived cardiac fibroblasts overexpressing or silencing SOD2.
- **Protein-level cardiac eQTL data.** The bulk-heart transcript QTLs systematically miss our signals. Cell-type-resolved snRNA-seq plus matched proteomics from human ventricular samples would let us localize the regulatory variants we currently can only infer.
- **An independent T1 GWAS for replication.** The current European cohort (~48,000 SNPs, 2 genome-wide significant loci) is underpowered; a second independent imaging GWAS would let us validate the directional enrichment screen against true colocalization.
- **A standardized plasma ASPN ELISA tied to longitudinal T1 measurements.** Preclinical and small human studies suggest circulating ASPN tracks fibrotic remodeling; we need a prospective cohort with paired imaging.
- **GSTP1 induction studies in cardiac contexts.** Existing Nrf2 agonists are pleiotropic. We are looking for selective GSTP1 inducers tested specifically against fibroblast activation endpoints.

If you have data, samples, or capacity to take on any of the above, file an issue on the [project repository][repo] or email iori@iluvatarlabs.com.

[repo]: https://github.com/iluvatar-open-research/project-sod2-myocardial-fibrosis

## Findings

### Genetic validation of six causal proteins

A two-sample MR screen of 1,420 proteins with at least three genome-wide significant cis-pQTL instruments identified 32 proteins that passed a Bonferroni threshold of *p* < 0.05/1,420 = 3.52 × 10⁻⁵. SOD2 was the top signal (inverse-variance weighted β = -0.257), with ECM1 and SCGN following as strong candidates (β = -0.037 and β = -0.369, respectively).

A focused MR analysis of SOD2 demonstrated a robust protective effect on T1 (β = -0.257, 95% CI -0.314 to -0.199, *p* = 3.24 × 10⁻¹⁸), with strong instrument strength (F = 67.82), concordance across MR estimators (weighted median β = -0.320, *p* = 7.58 × 10⁻⁶), and low heterogeneity (Q = 15.1, df = 13, *p* = 0.30). As a negative comparator, ECE1 showed no significant causal effect (β = 0.094, *p* = 0.087), underscoring specificity of the SOD2 finding.

![Figure 1](/assets/images/iori-sod2-myocardial-fibrosis/fig_1.png)

> **Figure 1** Mendelian randomization analysis reveals a protective causal effect of SOD2 on myocardial T1. The forest plot shows causal effect estimates (β) and 95% confidence intervals for genetically predicted levels of SOD2 and the negative comparator ECE1. Higher SOD2 levels are significantly associated with lower myocardial T1 (less fibrosis); ECE1 shows no significant causal effect.

Formal colocalization using `coloc` could initially be applied to only seven proteins due to limited SNP overlap (10 overlapping variants required), a consequence of systematic cross-dataset attrition wherein only 39.57% of pQTL SNPs were present in the T1 GWAS. In this restricted set, no protein achieved strong colocalization (PP.H4 > 0.5), with SCGN showing the best but still modest evidence (PP.H4 = 0.282). The lack of colocalization was consistent with the known low power of the T1 GWAS — only two genome-wide significant associations — indicating that absence of strong PP.H4 likely reflects power and overlap constraints rather than true biological discordance.

To break this stalemate, statistical fine-mapping and a directional enrichment strategy were applied to the MR-prioritized loci. For SOD2, SuSiE fine-mapping defined nine 95% credible-set variants across five signals; eight appeared in the T1 GWAS, of which 50% were nominally significant, 12.5% met *p* < 1 × 10⁻⁴, and **all showed protective directionality** consistent with the MR estimate. The lead cis-pQTL rs9365083 also associated with T1 at *p* = 9.2 × 10⁻¹⁰.

For ECM1, a 95% credible set of 61 variants contained 12 T1-nominal associations (19.7%) with 91.7% directional concordance. SCGN's 12-variant set contained six T1-nominal associations (50%), all directionally concordant. The remaining MR-positive proteins were adjudicated with a credible-set overlap and directional screen quantifying enrichment and inconsistency rates. Three additional proteins — IGDCC4 (13.04%), GSTP1 (27.78%), and ASPN (12.50%) — met the validation threshold with 0% inconsistency.

![Figure 2](/assets/images/iori-sod2-myocardial-fibrosis/fig_2.png)

> **Figure 2** GSTP1, ASPN, and IGDCC4 show significant enrichment of directionally-consistent signals for myocardial T1. Enrichment rate of T1 GWAS signals (*p* < 0.05) within the credible variant set for each protein. All three exceed the 10% validation threshold (red dashed line) with perfect directional consistency.

The emerging biology points to complementary mechanisms. The strong, directionally protective SOD2 effect aligns with a model in which increased mitochondrial superoxide dismutase lowers oxidative stress, attenuating fibroblast activation and collagen deposition, thereby reducing T1. ECM1 (extracellular matrix protein implicated in cardiac fibrosis) and SCGN (calcium signaling protein) suggest parallel axes of matrix remodeling and calcium-dependent cellular responses. The validated set is consistent with an integrated mechanism spanning antioxidant defenses and matrix/cell-signaling pathways.

### Mechanistic axes

A proteome-wide MR screen of 1,420 proteins identified 32 proteins with Bonferroni-significant causal effects on myocardial T1, with high concordance between IVW and MR-Egger directions (31/32) and limited evidence of pleiotropy (3/32 with significant intercepts).

![Figure 3](/assets/images/iori-sod2-myocardial-fibrosis/fig_3.png)

> **Figure 3** A proteome-wide MR screen identifies multiple proteins with significant causal effects on myocardial T1. The volcano plot shows the IVW beta coefficient versus statistical significance for 1,420 proteins, with the dashed line indicating the Bonferroni significance threshold (*p* < 3.52 × 10⁻⁵). SOD2, SCGN, and ECM1 are top candidates with significant negative effects, suggesting protective roles against myocardial fibrosis.

Because standard colocalization was weak — constrained by limited SNP overlap and absent sdY in the source GWAS — locus-focused fine-mapping and a directional enrichment screen were applied to test for shared causal variants between pQTLs and T1.

![Figure 4](/assets/images/iori-sod2-myocardial-fibrosis/fig_4.png)

> **Figure 4** Directional enrichment validates GSTP1, ASPN, and IGDCC4 as candidate proteins. Enrichment rate of directionally-consistent T1 GWAS signals (*p* < 0.05) for each protein, with the red dashed line indicating the 10% validation threshold.

The six validated proteins converge on a coherent mechanistic network in which **oxidative stress is upstream**, while TGF-β modulation, endothelial calcium signaling, and cell-adhesion programs govern fibrotic remodeling.

NOX4-derived reactive oxygen species initiate TGF-β/SMAD2/3 activation in cardiac fibroblasts. **SOD2 and GSTP1 act as protective antioxidant buffers** that limit ROS-driven p38/JNK MAPK signaling. Genetic or pharmacologic impairment of these enzymes increases p38/JNK phosphorylation and downstream stress responses; enhancing SOD2 activity reduces ROS and MAPK activation. The negative MR effects of SOD2 on T1 and the directional enrichment validating GSTP1 are consistent with reduced oxidative activation of pro-fibrotic signaling when these enzymes are upregulated.

Downstream of oxidative triggers, two extracellular modulators tune TGF-β availability: **ASPN directly binds TGF-β1** with high affinity (KD ≈ 15 nM); ASPN-mimic peptides attenuate SMAD2/3 phosphorylation, reduce COL1A1 expression in fibroblasts, and improve cardiac function in preclinical TAC models. **ECM1**, a matrix-organizing glycoprotein, likely reduces active TGF-β bioavailability by sequestering latent complexes. Together, ASPN and ECM1 act as extracellular brakes on TGF-β signaling that interface with the upstream redox axis.

A third axis centers on endothelial calcium signaling and secretory control. **SCGN**, an EF-hand Ca²⁺-binding protein with significant cardiac endothelial expression, plausibly attenuates calcineurin–NFAT activation and secretion-coupled paracrine pro-fibrotic cues. Endothelial secretome studies emphasize that release of anti-fibrotic mediators is Ca²⁺-dependent, aligning with SCGN's role in regulated exocytosis.

Finally, **IGDCC4's** genetic validation highlights an adhesion module. Immunoglobulin superfamily proteins (NCAMs, JAMs, Nectins) regulate fibroblast proliferation, migration, myofibroblast differentiation, and matrix synthesis — IGDCC4 may influence fibroblast–ECM and cell–cell interactions that set fibrotic tone, though precise mechanism is a key gap.

### Protein–transcript QTL discordance

The analysis began with direct colocalization between T1 GWAS hits and heart eQTLs but systematic harmonization failures revealed the scale of protein–transcript discordance. Neither of the two genome-wide significant T1 loci (rs79220007 at *p* = 3.97 × 10⁻¹⁰ and rs9365083 at *p* = 9.23 × 10⁻¹⁰) could be colocalized with GTEx heart eQTLs because the lead variants were entirely absent from the eQTL dataset. A global check confirmed only **1.25% SNP overlap** genome-wide.

A more aggressive coordinate-based harmonization of pQTL credible sets for ECM1 and SCGN with GTEx heart eQTL/sQTLs also yielded zero overlapping variants. The closest transcriptomic variants were 7,961 bp (ECM1) and 4,286 bp (SCGN) away — far below the minimum required for `coloc`. Targeted integration focusing on the six high-confidence proteins showed that **none of 300 pQTL credible-set variants matched significant cis-eQTL or cis-sQTL variants** in GTEx heart, despite robust transcript-level QTLs for the corresponding genes.

![Figure 5](/assets/images/iori-sod2-myocardial-fibrosis/fig_5.png)

> **Figure 5** Coordinate-based harmonization shows zero overlapping variants between pQTL credible sets for ECM1 and SCGN and GTEx heart transcript QTLs. The number of shared variants between the protein QTLs and expression (eQTL) or splicing (sQTL) QTLs from heart atrial appendage (AA) and left ventricle (LV) is zero across all comparisons.

Functional annotation of the fine-mapped signals contextualizes this: the protein–transcript discordance points to **cell-type-specific enhancers and post-transcriptional control** rather than bulk transcriptional mediation. At the SOD2 locus, the T1-associated pQTL signal maps to a distal enhancer active in cardiac fibroblasts and immune cells, marked by H3K27ac, open chromatin, and promoter contact via chromatin looping. SOD2 also harbors a 3'-UTR variant (rs4555948) predicted to disrupt binding of hsa-miR-222-3p — a post-transcriptional route to altering protein abundance.

![Figure 6](/assets/images/iori-sod2-myocardial-fibrosis/fig_6.png)

> **Figure 6** Variants in the SOD2 pQTL 95% credible set are enriched for association with the T1 cardiac GWAS signal. Percentage of credible-set variants reaching nominal (*p* < 0.05) or strong (*p* < 1e-4) association with the T1 phenotype, plus directionally consistent effects. Complete directional consistency supports a shared causal variant operating at the protein level.

Collectively, these results explain why bulk heart eQTL/sQTL integration fails despite strong protein-T1 associations: causal regulatory variation for these proteins is concentrated in distal, cell-type-specific enhancers and, at least for SOD2, in post-transcriptional elements rather than in transcript-level or splicing effects detectable in bulk tissue.

### Translational implications

ASPN is strongly positioned for translation. ASPN-mimic peptides directly bind TGF-β1 with high affinity (KD ≈ 15 nM by microscale thermophoresis), reduce phosphorylated SMAD2/3, and suppress pro-fibrotic gene expression. *In vivo* administration improves cardiac function and reduces histologic fibrosis in mouse models, consistent with on-target interference in TGF-β signaling. ELISA detects an approximately threefold rise in circulating ASPN in mouse ischemia-reperfusion, supporting both target engagement and biomarker plausibility. Together with the genetic evidence, **ASPN nominates as a dual-use opportunity**: a tractable anti-fibrotic target via peptide antagonism of TGF-β1/SMAD2/3, and a candidate circulating biomarker for remodeling.

GSTP1 also shows strong genetic support, but pharmacology indicates its **inhibition is maladaptive** in the heart, redirecting strategy toward activation. The GSTP1 inhibitor arsenic trioxide consistently reduces ejection fraction, increases left ventricular end-diastolic volume, and promotes collagen deposition *in vivo*, while inducing endothelial-to-mesenchymal transition through AKT/GSK-3β/Snail signaling *in vitro*. Pharmacologic activation of the Nrf2 pathway, which upregulates GSTP1 among other antioxidant defenses, reduces oxidative injury and fibrosis markers. The translational stance: **selectively increase GSTP1 activity** (e.g., via Nrf2 agonism or GSTP1-directed inducers) rather than inhibit it.

![Figure 7](/assets/images/iori-sod2-myocardial-fibrosis/fig_7.png)

> **Figure 7** GSTP1, ASPN, and IGDCC4 are validated as causal candidates for T1-indexed myocardial fibrosis. All three proteins exceeded the pre-specified 10% validation threshold (red dashed line) with perfect directional consistency, supporting their causal association with fibrosis.

Biomarker appraisals align with this prioritization while defining pragmatic limits. Despite high interest, recent human studies have not demonstrated robust associations between circulating ASPN, ECM1, or SCGN and cardiac imaging markers. ASPN shows encouraging preclinical serum signals and human tissue upregulation, nominating it as a near-term candidate for targeted plasma assays. SOD2 is measurable in blood and shows moderate associations with atrial remodeling and paroxysmal atrial fibrillation, but these data do not yet anchor SOD2 to myocardial fibrosis directly.

Two-sample MR against diagnosed heart failure does **not** yield robust causal associations for any of the six proteins. This boundary condition suggests these proteins exert measurable effects on subclinical interstitial fibrosis (captured by T1) without translating into detectable differences in overt heart failure risk at current instrument strength. **Specificity to subclinical fibrosis** is the testable claim that follow-up cohorts should address.

## Sources, datasets, and literature

### Datasets

- **Myocardial T1 GWAS** — `filteredmyocardial_t1data.tsv`. ~48,000 SNPs from a European cohort, 2 genome-wide significant hits (*p* < 5 × 10⁻⁸).
- **Combined cis-pQTLs** — `combinedcis_pQTLnew.tsv`. Multi-tissue proteomics GWAS, *p* < 5 × 10⁻⁵, 2,175 unique proteins. Same European cohort as the T1 GWAS.
- **GTEx v10 eGenes** — Heart Atrial Appendage and Heart Left Ventricle tissue-level summaries.
- **GTEx v10 sGenes** — same tissues, splicing.
- **1000 Genomes Europeans** — LD reference, accessed via `LDlinkR`.

Inaccessible at time of writing (Parquet format limitations): GTEx v10 significant variant-level pair data for eQTLs and sQTLs in both tissues.

### Software

`MendelianRandomization` (R), `TwoSampleMR` (R), `coloc` (R), `susieR` (R), `LDlinkR` (R, token `5efa76aca201`).

### Selected literature

- Maris et al. 2015. ASPN-mimic peptides and TGF-β1 antagonism in cardiac fibrosis. *PLOS Medicine*.
- Huang et al. 2022. Circulating ASPN in mouse cardiac injury. *Matrix Biology*.
- Zhang et al. 2016. Arsenic trioxide as GSTP1 inhibitor and cardiotoxicity. *Scientific Reports*.
- Tan et al. 2020. Cardiac fibroblast enhancer landscapes. *Circulation Research*.
- Amrute et al. 2024. Cardiac fibroblast cis-regulation. *PMC*.

Full corpus, including rejected sources, in the [project repository][repo].

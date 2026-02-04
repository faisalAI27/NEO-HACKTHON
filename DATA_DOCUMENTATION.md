# 📊 TCGA-HNSC Multi-Modal Dataset Documentation

## Complete Data Reference Guide
**Generated:** January 29, 2026  
**Dataset:** The Cancer Genome Atlas - Head and Neck Squamous Cell Carcinoma (TCGA-HNSC)  
**Location:** `c:\Users\Ibrahim\Desktop\DNA sequence\`

---

## 🔬 Dataset Overview

This is a **comprehensive multi-modal cancer research dataset** from The Cancer Genome Atlas (TCGA) project, focusing on **Head and Neck Squamous Cell Carcinoma (HNSC)**. The dataset integrates multiple data modalities that together provide a complete molecular and clinical portrait of each patient's cancer.

### Quick Statistics
| Component | Count | Description |
|-----------|-------|-------------|
| Patients with full data | ~82-96 | Varies by data type |
| Whole Slide Images (.svs) | 172 | Histopathology slides |
| Genes in transcriptomics | 20,504 | RNA-seq expression |
| Genes in methylation | 20,115 | DNA methylation beta values |
| Clinical variables | ~200+ | Comprehensive clinical data |

### Disease Information
- **Cancer Type:** Head and Neck Squamous Cell Carcinoma
- **ICD-O-3 Morphology:** 8070/3 (Squamous Cell Neoplasms)
- **Primary Sites:** Tongue, Floor of mouth, Tonsil, Oropharynx, Larynx
- **TCGA Project ID:** TCGA-HNSC

---

## 📁 File-by-File Detailed Documentation

---

## 1. clinical.txt

### Basic Information
| Property | Value |
|----------|-------|
| **File Size** | Large (200+ columns) |
| **Format** | Tab-separated values (TSV) |
| **Rows** | ~82 patients |
| **Columns** | ~200+ clinical variables |

### Purpose
Contains comprehensive clinical information for each patient including demographics, diagnosis details, staging, treatment records, and survival outcomes. This is the **central linking file** that connects patient identifiers across all other data files.

### Key Column Categories

#### A. Project & Patient Identifiers
| Column | Description | Example Values |
|--------|-------------|----------------|
| `project.project_id` | TCGA project identifier | TCGA-HNSC |
| `cases.case_id` | Unique UUID for the case | 03aca47b-7653-4938-9178-ed7c37eee6d5 |
| `cases.submitter_id` | Human-readable patient ID | TCGA-CR-7392, TCGA-HD-7831 |

#### B. Demographic Information
| Column | Description | Example Values |
|--------|-------------|----------------|
| `demographic.age_at_index` | Age at diagnosis (years) | 52, 67, 74, 79 |
| `demographic.vital_status` | Survival status | Alive, Dead |
| `demographic.days_to_death` | Days from diagnosis to death | 548, 667, 1425 |
| `demographic.gender` | Patient sex | male, female |
| `demographic.race` | Racial background | white, black or african american |
| `demographic.ethnicity` | Hispanic/Latino status | not hispanic or latino |

#### C. Diagnosis Information
| Column | Description | Example Values |
|--------|-------------|----------------|
| `diagnoses.age_at_diagnosis` | Age at diagnosis (days) | 19236, 24383, 27083 |
| `diagnoses.primary_diagnosis` | ICD-10 diagnosis code | Squamous cell carcinoma, NOS |
| `diagnoses.tissue_or_organ_of_origin` | Anatomical site (ICD-O-3) | C02.9 (Tongue), C04.9 (Floor of mouth), C09.9 (Tonsil) |
| `diagnoses.site_of_resection_or_biopsy` | Where sample was taken | Tongue, NOS; Floor of mouth, NOS |
| `diagnoses.tumor_grade` | Histological grade | G1, G2, G3, GX |
| `diagnoses.morphology` | Cancer type code | 8070/3 (Squamous cell carcinoma) |

#### D. AJCC Staging (TNM Classification)
| Column | Description | Example Values |
|--------|-------------|----------------|
| `diagnoses.ajcc_staging_system_edition` | AJCC edition used | 6th, 7th |
| `diagnoses.ajcc_clinical_stage` | Overall clinical stage | Stage I, Stage II, Stage III, Stage IVA |
| `diagnoses.ajcc_clinical_t` | Tumor size/extent | T1, T2, T3, T4a |
| `diagnoses.ajcc_clinical_n` | Lymph node involvement | N0, N1, N2, N2a, N2b, N2c |
| `diagnoses.ajcc_clinical_m` | Distant metastasis | M0, M1 |
| `diagnoses.ajcc_pathologic_stage` | Post-surgery stage | Stage IVA |
| `diagnoses.ajcc_pathologic_t` | Pathologic T stage | pT1, pT2, pT3, pT4a |
| `diagnoses.ajcc_pathologic_n` | Pathologic N stage | pN0, pN1, pN2b |

#### E. Treatment Information
| Column | Description | Example Values |
|--------|-------------|----------------|
| `treatments.treatment_type` | Type of treatment | Radiation Therapy, Pharmaceutical Therapy, Surgery |
| `treatments.treatment_or_therapy` | Whether treatment given | yes, no |
| `treatments.therapeutic_agents` | Specific drugs used | Cisplatin, Carboplatin, Paclitaxel, Pemetrexed |
| `treatments.treatment_intent_type` | Goal of treatment | Cure, Adjuvant, Palliative |
| `treatments.treatment_outcome` | Response to treatment | Complete Response, Partial Response, Progressive Disease |
| `treatments.prescribed_dose` | Radiation/drug dose | 6000 (cGy for radiation), 80 (mg for chemo) |

### Relationship to Other Files
- **Primary Key:** `cases.submitter_id` (e.g., TCGA-CR-7392)
- **Links to:** All other files via patient ID prefix
- **SVS Images:** Match by removing sample suffix (TCGA-CR-7392.svs)
- **Molecular Data:** Match by adding sample suffix (-01 for tumor, -11 for normal)

### Clinical Insights
- **Stage Distribution:** Predominantly Stage IVA (advanced disease)
- **Treatment Patterns:** Multi-modal therapy common (surgery + radiation + chemotherapy)
- **Survival Data:** Both alive and deceased patients with detailed follow-up

---

## 2. exposure.txt

### Basic Information
| Property | Value |
|----------|-------|
| **File Size** | 116 lines |
| **Format** | Tab-separated values (TSV) |
| **Rows** | ~58 patients (some have multiple entries) |
| **Columns** | 40 exposure-related variables |

### Purpose
Documents patient lifestyle and environmental exposure history, with particular focus on **tobacco and alcohol use** - the two primary risk factors for head and neck cancer.

### Key Columns

#### A. Tobacco Exposure
| Column | Description | Example Values |
|--------|-------------|----------------|
| `exposures.tobacco_smoking_status` | Current smoking status | Current Smoker, Current Reformed Smoker for > 15 yrs, Current Reformed Smoker for < or = 15 yrs, Lifelong Non-Smoker |
| `exposures.pack_years_smoked` | Cumulative tobacco exposure | 0.01 to 110 (pack-years) |
| `exposures.tobacco_smoking_onset_year` | Year started smoking | 1936, 1948, 1954, etc. |
| `exposures.tobacco_smoking_quit_year` | Year quit smoking | 1956, 1979, 1995, etc. |
| `exposures.cigarettes_per_day` | Daily cigarette consumption | Numeric values |
| `exposures.years_smoked` | Duration of smoking | Numeric values |
| `exposures.exposure_source` | Source of exposure | Tobacco |

#### B. Alcohol Exposure
| Column | Description | Example Values |
|--------|-------------|----------------|
| `exposures.alcohol_history` | Any alcohol use history | Yes, No |
| `exposures.alcohol_days_per_week` | Drinking frequency | 0, 1, 7 (days) |
| `exposures.alcohol_drinks_per_day` | Daily consumption | 0, 1, 3, 6, 15 (drinks) |
| `exposures.alcohol_intensity` | Drinking pattern | Light, Moderate, Heavy |

### Exposure Statistics Observed
| Category | Range/Values |
|----------|--------------|
| Pack-years | 0.01 - 110 (extreme range!) |
| Alcohol days/week | 0 - 7 |
| Alcohol drinks/day | 0 - 15 |
| Smoking onset | 1936 - 1988 |

### Relationship to Other Files
- **Links via:** `cases.submitter_id`
- **Clinical Connection:** Exposure data contextualizes disease development; heavy smokers (110 pack-years) may have different tumor biology
- **Molecular Data:** Tobacco-associated tumors have distinct mutational signatures (e.g., C>A transversions)
- **Survival Analysis:** Exposure levels correlate with treatment response and survival

### Research Importance
- HNSC is strongly associated with tobacco/alcohol use
- HPV-negative tumors are typically tobacco-related
- Pack-years is a key prognostic variable
- Enables risk factor stratification in machine learning models

---

## 3. follow_up.txt

### Basic Information
| Property | Value |
|----------|-------|
| **File Size** | 281 lines |
| **Format** | Tab-separated values (TSV) |
| **Rows** | Multiple entries per patient (longitudinal) |
| **Columns** | ~200 columns (follow-up + molecular tests + other clinical attributes) |

### Purpose
Tracks patient outcomes over time after initial treatment, including disease status, recurrence events, and molecular testing results. This file provides **longitudinal survival data** essential for survival analysis.

### Key Column Categories

#### A. Follow-up Visit Information
| Column | Description | Example Values |
|--------|-------------|----------------|
| `follow_ups.days_to_follow_up` | Days since diagnosis | 0, 129, 361, 1425, 2347 |
| `follow_ups.disease_response` | Tumor status at visit | TF-Tumor Free, WT-With Tumor |
| `follow_ups.timepoint_category` | Type of visit | Follow-up, Last Contact, Post Initial Treatment |

#### B. Recurrence/Progression Data
| Column | Description | Example Values |
|--------|-------------|----------------|
| `follow_ups.progression_or_recurrence` | Did cancer return? | Yes, No |
| `follow_ups.progression_or_recurrence_anatomic_site` | Where it recurred | Lung, NOS; Oropharynx, NOS; Mouth, NOS; Lymph nodes of head, face and neck |
| `follow_ups.progression_or_recurrence_type` | Type of recurrence | Locoregional, Distant |
| `follow_ups.days_to_progression` | Days to progression | 50, 230, 447, 667, 1716, 2168 |
| `follow_ups.days_to_recurrence` | Days to recurrence | Numeric values |
| `follow_ups.evidence_of_recurrence_type` | How recurrence confirmed | Biopsy with Histologic Confirmation |

#### C. Molecular Testing Results
| Column | Description | Example Values |
|--------|-------------|----------------|
| `molecular_tests.gene_symbol` | Gene tested | CDKN2A, EGFR |
| `molecular_tests.laboratory_test` | Test for specific marker | Human Papillomavirus |
| `molecular_tests.molecular_analysis_method` | Testing method | ISH (In Situ Hybridization), Not Reported |
| `molecular_tests.test_result` | Result of test | Positive, Negative, Amplified |
| `molecular_tests.variant_type` | Type of alteration | Amplification |
| `molecular_tests.biospecimen_type` | Sample type tested | Involved Tissue, NOS |

### HPV Status (Critical Variable)
| Test | Method | Results Seen |
|------|--------|--------------|
| Human Papillomavirus | ISH | Positive, Negative |
| CDKN2A | Not Reported | Negative |
| EGFR | Not Reported | Amplified |

### Follow-up Duration Statistics
- **Minimum:** 0 days
- **Maximum:** 2,347 days (~6.4 years)
- **Multiple visits:** Most patients have 2-7 follow-up entries

### Relationship to Other Files
- **Links via:** `cases.submitter_id`
- **Clinical:** Extends survival data from clinical.txt with longitudinal outcomes
- **Pathology:** Recurrence sites relate to original pathology findings
- **Molecular Data:** HPV status should correlate with distinct gene expression signatures
- **Images:** Recurrence may show morphological predictors in original slides

### Research Importance
- **Survival Analysis:** Enables Kaplan-Meier curves, Cox regression
- **HPV Stratification:** HPV+ vs HPV- tumors are essentially different diseases
- **Recurrence Prediction:** Can build models to predict who will recur

---

## 4. pathology_detail.txt

### Basic Information
| Property | Value |
|----------|-------|
| **File Size** | ~82 rows |
| **Format** | Tab-separated values (TSV) |
| **Rows** | ~82 patients |
| **Columns** | 87 pathology-specific variables |

### Purpose
Contains detailed surgical pathology findings from tumor resection, including lymph node dissection results, surgical margins, and tissue invasion patterns. This represents the **gold standard** histopathological assessment.

### Key Column Categories

#### A. Lymph Node Assessment
| Column | Description | Example Values |
|--------|-------------|----------------|
| `lymph_node_dissection_method` | Surgical technique | Modified Radical Neck Dissection, Functional (Limited) Neck Dissection, Radical Neck Dissection |
| `lymph_node_dissection_site` | Location of dissection | Neck Left, Neck Right, Neck NOS |
| `lymph_nodes_tested` | Total nodes examined | 14, 25, 36, 64, 73 |
| `lymph_nodes_positive` | Nodes with cancer | 0, 1, 4, 8, 9 |
| `extranodal_extension` | Cancer beyond node capsule | No Extranodal Extension, Microscopic Extension, Gross Extension |

#### B. Surgical Margins
| Column | Description | Example Values |
|--------|-------------|----------------|
| `margin_status` | Completeness of resection | Uninvolved (clear margins), Involved (cancer at edge), Indeterminate |

#### C. Tissue Invasion Patterns
| Column | Description | Example Values |
|--------|-------------|----------------|
| `perineural_invasion_present` | Cancer along nerves | Yes, No |
| `vascular_invasion_present` | Cancer in blood vessels | Yes, No |
| `lymphatic_invasion_present` | Cancer in lymph vessels | Yes, No |

### Pathology Statistics Observed
| Metric | Range |
|--------|-------|
| Lymph nodes tested | 14 - 73 |
| Lymph nodes positive | 0 - 9+ |
| Most common dissection | Modified Radical Neck Dissection |

### Lymph Node Ratio
- Can be calculated: `lymph_nodes_positive / lymph_nodes_tested`
- Important prognostic factor in HNSC
- Example: 8/36 = 0.22 (22% positive)

### Relationship to Other Files
- **Links via:** `cases.submitter_id`
- **Clinical:** Pathologic staging (pT, pN) derives from this assessment
- **SVS Images:** Images are the visual representation of this pathology data
- **Molecular Data:** Invasion patterns may correlate with gene expression (EMT signatures)
- **Survival:** Positive margins, extranodal extension predict worse outcomes

### Research Importance
- **Prognosis:** Lymph node status is the #1 prognostic factor
- **Treatment Decisions:** Positive margins → adjuvant radiation
- **AI Targets:** Can train models to predict these features from SVS images

---

## 5. methylation.txt

### Basic Information
| Property | Value |
|----------|-------|
| **File Size** | Large matrix file |
| **Format** | Tab-separated values (TSV) |
| **Rows** | 20,116 (20,115 genes + header) |
| **Columns** | 97 (1 gene name + 96 samples) |

### Purpose
Contains **DNA methylation beta values** for gene promoter regions across all samples. Methylation is an epigenetic modification that typically **silences gene expression** when present at promoters.

### Data Structure
```
       Gene  | Sample1 | Sample2 | Sample3 | ...
       A1BG  |  0.559  |  0.616  |  0.648  | ...
       A1CF  |  0.573  |  0.472  |  0.635  | ...
       A2BP1 |  0.488  |  0.440  |  0.708  | ...
```

### Sample Naming Convention
| Pattern | Meaning | Example |
|---------|---------|---------|
| TCGA-XX-XXXX-01 | Primary Tumor | TCGA-BA-4074-01 |
| TCGA-XX-XXXX-11 | Normal Adjacent Tissue | TCGA-CV-5442-11 |

### Tumor-Normal Pairs Identified (14 pairs)
| Patient ID | Tumor Sample | Normal Sample |
|------------|--------------|---------------|
| TCGA-CV-5442 | TCGA-CV-5442-01 | TCGA-CV-5442-11 |
| TCGA-CV-5971 | TCGA-CV-5971-01 | TCGA-CV-5971-11 |
| TCGA-CV-5973 | TCGA-CV-5973-01 | TCGA-CV-5973-11 |
| TCGA-CV-5979 | TCGA-CV-5979-01 | TCGA-CV-5979-11 |
| TCGA-CV-6433 | TCGA-CV-6433-01 | TCGA-CV-6433-11 |
| TCGA-CV-6441 | TCGA-CV-6441-01 | TCGA-CV-6441-11 |
| TCGA-CV-6934 | TCGA-CV-6934-01 | TCGA-CV-6934-11 |
| TCGA-CV-6956 | TCGA-CV-6956-01 | TCGA-CV-6956-11 |
| TCGA-CV-6959 | TCGA-CV-6959-01 | TCGA-CV-6959-11 |
| TCGA-CV-7103 | TCGA-CV-7103-01 | TCGA-CV-7103-11 |
| TCGA-CV-7235 | TCGA-CV-7235-01 | TCGA-CV-7235-11 |
| TCGA-CV-7238 | TCGA-CV-7238-01 | TCGA-CV-7238-11 |
| TCGA-CV-7255 | TCGA-CV-7255-01 | TCGA-CV-7255-11 |
| TCGA-CV-7263 | TCGA-CV-7263-01 | TCGA-CV-7263-11 |

### Beta Value Interpretation
| Beta Value | Methylation Status | Gene Effect |
|------------|-------------------|-------------|
| 0.0 - 0.2 | Unmethylated | Gene typically active |
| 0.2 - 0.6 | Partially methylated | Variable expression |
| 0.6 - 1.0 | Hypermethylated | Gene typically silenced |

### Genes Included (Sample)
- A1BG, A1CF, A2BP1, A2LD1, A2M, A2ML1
- A4GALT, A4GNT, AAA1, AAAS, AACS, AACSL
- AADAC, AADACL2, AADACL3, AADACL4, AADAT
- ABCA family (ABCA1-ABCA13), ABCB family (ABCB1-ABCB11)
- ... 20,000+ additional genes

### Relationship to Other Files
- **Links via:** Sample ID (TCGA-XX-XXXX portion matches clinical)
- **Transcriptomics:** Methylation typically anti-correlates with expression
- **Mutations:** Some mutations affect methylation machinery (DNMT3A, TET2)
- **Clinical:** Methylation patterns differ by HPV status, stage
- **Pathology:** Methylation of specific genes (CDH1) relates to invasion

### Research Applications
1. **Differential Methylation Analysis:** Tumor vs Normal
2. **Methylation-Expression Integration:** Find epigenetically silenced genes
3. **Biomarker Discovery:** Methylation panels for diagnosis/prognosis
4. **HPV Subtyping:** HPV+ tumors have distinct methylation profiles

---

## 6. mutations.txt

### Basic Information
| Property | Value |
|----------|-------|
| **File Size** | Large (300+ columns) |
| **Format** | MAF (Mutation Annotation Format) |
| **Rows** | Multiple mutations per patient |
| **Columns** | 300+ annotation fields |

### Purpose
Contains all **somatic (cancer-specific) mutations** detected by Whole Exome Sequencing (WXS). Each row represents one mutation with extensive annotations about its genomic location, predicted impact, and clinical significance.

### Sequencing Information
| Property | Value |
|----------|-------|
| Platform | Illumina HiSeq |
| Assay | Whole Exome Sequencing (WXS) |
| Reference Genome | GRCh37 (hg19) |
| NCBI Build | 37 |

### Key Column Categories

#### A. Gene Information
| Column | Description | Example Values |
|--------|-------------|----------------|
| `Hugo_Symbol` | Gene name | TP53, BCL9, NOTCH2NL, SLC2A7 |
| `Entrez_Gene_Id` | NCBI gene ID | 155184, 607, 7157 |
| `Gene` | Gene symbol (alternate) | Same as Hugo_Symbol |

#### B. Genomic Location
| Column | Description | Example Values |
|--------|-------------|----------------|
| `Chromosome` | Chromosome number | 1, 2, 3, ... 22, X, Y |
| `Start_Position` | Mutation start coordinate | 9086388, 12423152, 43908207 |
| `End_Position` | Mutation end coordinate | Usually same as start for SNVs |
| `Strand` | DNA strand | +, - |

#### C. Mutation Details
| Column | Description | Example Values |
|--------|-------------|----------------|
| `Variant_Classification` | Mutation type | Missense_Mutation, Nonsense_Mutation, Silent, Frame_Shift_Del, Splice_Site |
| `Variant_Type` | Nucleotide change type | SNP, INS, DEL |
| `Reference_Allele` | Normal sequence | A, C, G, T |
| `Tumor_Seq_Allele1` | First tumor allele | A, C, G, T |
| `Tumor_Seq_Allele2` | Second tumor allele | A, C, G, T |

#### D. Protein Change
| Column | Description | Example Values |
|--------|-------------|----------------|
| `HGVSp_Short` | Protein change notation | p.A6E, p.V179M, p.S228* |
| `Protein_Change` | Amino acid change | A6E, V179M, I3433F |
| `Amino_Acid_Change` | Full annotation | p.Ala6Glu |
| `Codon_Change` | Codon-level change | c.17G>T |

#### E. Pathogenicity Predictions
| Column | Description | Interpretation |
|--------|-------------|----------------|
| `SIFT_pred` | SIFT prediction | T=Tolerated, D=Deleterious |
| `PolyPhen2_HDIV_pred` | PolyPhen2 (HDIV) | B=Benign, P=Possibly damaging, D=Probably damaging |
| `PolyPhen2_HVAR_pred` | PolyPhen2 (HVAR) | B=Benign, P=Possibly damaging, D=Probably damaging |
| `MutationTaster_pred` | MutationTaster | N=Neutral, D=Disease causing |
| `FATHMM_pred` | FATHMM | T=Tolerated, D=Damaging |
| `CADD_phred` | CADD score (phred-scaled) | Higher = more damaging (>20 = top 1%) |

#### F. Database Annotations
| Column | Description | Example Values |
|--------|-------------|----------------|
| `dbSNP_RS` | dbSNP identifier | rs12345678 |
| `COSMIC_ID` | COSMIC mutation ID | COSM12345 |
| `ExAC_AF` | Population frequency | 0.0001 (rare) |
| `CGC_Cancer_Gene` | Cancer Gene Census | Yes/No |

### Example Mutations Observed
| Gene | Mutation | Type | CADD Score | Significance |
|------|----------|------|------------|--------------|
| BCL9 | p.V179M | Missense | 28.1 | Cancer gene |
| NOTCH2NL | p.S228* | Nonsense | 21.0 | Stop-gain |
| VPS13D | p.I3433F | Missense | 18.93 | Likely damaging |
| SZT2 | p.H2633L | Missense | 18.42 | Likely damaging |
| LPHN2 | p.A926D | Missense | 17.70 | Uncertain |
| SLC2A7 | p.A6E | Missense | 1.238 | Likely benign |

### Relationship to Other Files
- **Links via:** Sample barcode (TCGA-XX-XXXX-01A portion)
- **Clinical:** Driver mutations affect prognosis and treatment options
- **Transcriptomics:** Mutations may alter gene expression levels
- **Methylation:** Some mutations affect epigenetic regulators
- **Images:** Cannot directly visualize mutations, but mutation burden may correlate with immune infiltration patterns

### Known HNSC Driver Genes
Common mutations in HNSC to look for:
- **TP53** - Most frequently mutated (~70%)
- **CDKN2A** - Tumor suppressor
- **PIK3CA** - Oncogene
- **NOTCH1** - Tumor suppressor in HNSC
- **FAT1** - Tumor suppressor
- **EGFR** - Often amplified

---

## 7. transcriptomics.txt

### Basic Information
| Property | Value |
|----------|-------|
| **File Size** | Large matrix file |
| **Format** | Tab-separated values (TSV) |
| **Rows** | 20,505 (20,504 genes + header) |
| **Columns** | 93 (1 gene name + 92 samples) |

### Purpose
Contains **RNA-seq gene expression counts** representing the transcriptional activity of all protein-coding genes. This measures how actively each gene is being transcribed into mRNA.

### Data Structure
```
       Gene  | Sample1 | Sample2 | Sample3 | ...
       A1BG  |    247  |    192  |    422  | ...
       A1CF  |      1  |      0  |      2  | ...
       A2BP1 |     55  |     43  |    118  | ...
```

### Sample Naming Convention
| Pattern | Meaning | Example |
|---------|---------|---------|
| TCGA-XX-XXXX-01A-XXR-XXXX-07 | Primary Tumor | TCGA-CV-6934-01A-11R-1915-07 |
| TCGA-XX-XXXX-11A-XXR-XXXX-07 | Normal Adjacent | TCGA-CV-6934-11A-01R-1915-07 |

### Sample Code Breakdown
```
TCGA-CV-6934-01A-11R-1915-07
     │    │    │  │  │   │
     │    │    │  │  │   └─ Analysis workflow version
     │    │    │  │  └───── RNA aliquot ID
     │    │    │  └──────── Sample portion
     │    │    └─────────── Sample type (01=tumor, 11=normal)
     │    └──────────────── Patient ID
     └───────────────────── Tissue source site
```

### Tumor-Normal Pairs Identified (10 pairs)
| Patient ID | Tumor Sample | Normal Sample |
|------------|--------------|---------------|
| TCGA-CV-6934 | -01A- | -11A- |
| TCGA-CV-6956 | -01A- | -11A- |
| TCGA-CV-7097 | -01A- | -11A- |
| TCGA-CV-7103 | -01A- | -11A- |
| TCGA-CV-7183 | -01A- | -11A- |
| TCGA-CV-7235 | -01A- | -11A- |
| TCGA-CV-7238 | -01A- | -11A- |
| TCGA-CV-7255 | -01A- | -11A- |
| TCGA-CV-7416 | -01A- | -11A- |
| TCGA-CV-7434 | -01A- | -11A- |

### Expression Value Interpretation
| Count Range | Expression Level | Note |
|-------------|------------------|------|
| 0-10 | Not expressed / Very low | May be noise |
| 10-100 | Low expression | |
| 100-1,000 | Moderate expression | |
| 1,000-10,000 | High expression | |
| >10,000 | Very high expression | Highly active genes |

### Expression Range Examples
| Gene | Min Count | Max Count | Range |
|------|-----------|-----------|-------|
| A2ML1 | 10 | 373,160 | Highly variable |
| A2M | 1,290 | 341,870 | Consistently high |
| A1BG | 38 | 1,553 | Moderate, stable |

### Genes Included (Sample)
- Housekeeping: ACTB, GAPDH, HPRT1
- Immune: CD4, CD8A, FOXP3, PDCD1
- Tumor suppressors: TP53, RB1, CDKN2A
- Oncogenes: MYC, EGFR, PIK3CA
- EMT markers: CDH1, VIM, SNAI1, ZEB1
- ... 20,000+ total genes

### Relationship to Other Files
- **Links via:** Sample barcode matches clinical (TCGA-XX-XXXX)
- **Methylation:** Expression should inversely correlate with promoter methylation
- **Mutations:** Truncating mutations may reduce expression
- **Clinical:** Gene signatures predict survival, HPV status, treatment response
- **Images:** Expression patterns (immune genes) may correlate with TIL infiltration visible in images

### Research Applications
1. **Differential Expression Analysis:** Tumor vs Normal
2. **Gene Signature Development:** Prognostic or predictive signatures
3. **Pathway Analysis:** GSEA, over-representation analysis
4. **Molecular Subtyping:** Classify tumors into subtypes
5. **Immune Profiling:** Estimate immune cell infiltration (CIBERSORT, xCell)

---

## 8. Copy of data/ (SVS Image Directory)

### Basic Information
| Property | Value |
|----------|-------|
| **Location** | `c:\Users\Ibrahim\Desktop\DNA sequence\Copy of data\` |
| **File Count** | 172 files |
| **File Format** | .svs (Aperio ScanScope Virtual Slide) |
| **Image Type** | Whole Slide Images (WSI) of H&E stained tissue |

### Purpose
Contains **gigapixel histopathology images** of H&E (Hematoxylin & Eosin) stained tissue sections from patient tumors. These are the same slides that pathologists examine for diagnosis.

### Technical Specifications
| Property | Typical Value |
|----------|---------------|
| Resolution | 40x or 20x magnification |
| Pixel Size | ~0.25 μm/pixel (40x) |
| Image Dimensions | 50,000 × 50,000+ pixels |
| File Size | 500 MB - 3 GB per image |
| Pyramid Levels | Multiple resolution levels |
| Format | TIFF-based, SVS extension |

### Sample Distribution by Tissue Source Site
| Site Code | Count | Institution |
|-----------|-------|-------------|
| TCGA-CV | 39 | Christiana Healthcare |
| TCGA-CN | 23 | UCSF |
| TCGA-CR | 19 | Cureline Inc |
| TCGA-CQ | 14 | MD Anderson |
| TCGA-BA | 11 | Thomas Jefferson |
| TCGA-BB | 7 | Greater Poland Cancer Centre |
| TCGA-HD | 7 | University of Chicago |
| TCGA-D6 | 6 | Roswell Park |
| TCGA-P3 | 6 | St. Joseph's Medical Center |
| TCGA-DQ | 5 | Duke University |
| Others | 35 | Various other sites |

### Sample List (172 images)
<details>
<summary>Click to expand full list</summary>

```
TCGA-4P-AA8J.svs    TCGA-CN-6020.svs    TCGA-CV-6940.svs    TCGA-CV-A6K0.svs
TCGA-BA-4074.svs    TCGA-CN-6995.svs    TCGA-CV-6941.svs    TCGA-CX-7082.svs
TCGA-BA-4077.svs    TCGA-CN-6996.svs    TCGA-CV-6945.svs    TCGA-CX-7086.svs
TCGA-BA-5153.svs    TCGA-CN-A498.svs    TCGA-CV-6948.svs    TCGA-D6-6516.svs
TCGA-BA-5556.svs    TCGA-CN-A49A.svs    TCGA-CV-6952.svs    TCGA-D6-6823.svs
TCGA-BA-5557.svs    TCGA-CN-A49C.svs    TCGA-CV-6953.svs    TCGA-D6-A4Z9.svs
TCGA-BA-6872.svs    TCGA-CN-A63V.svs    TCGA-CV-6956.svs    TCGA-D6-A6EM.svs
TCGA-BA-6873.svs    TCGA-CN-A6UY.svs    TCGA-CV-6959.svs    TCGA-D6-A6EN.svs
TCGA-BA-7269.svs    TCGA-CN-A6V7.svs    TCGA-CV-7097.svs    TCGA-D6-A6EO.svs
TCGA-BA-A6D8.svs    TCGA-CQ-5330.svs    TCGA-CV-7099.svs    TCGA-DQ-5624.svs
TCGA-BA-A6DG.svs    TCGA-CQ-5331.svs    TCGA-CV-7100.svs    TCGA-DQ-5625.svs
TCGA-BA-A6DJ.svs    TCGA-CQ-5332.svs    TCGA-CV-7102.svs    TCGA-DQ-5631.svs
TCGA-BB-4224.svs    TCGA-CQ-5334.svs    TCGA-CV-7103.svs    TCGA-DQ-7588.svs
TCGA-BB-4225.svs    TCGA-CQ-6218.svs    TCGA-CV-7183.svs    TCGA-DQ-7593.svs
TCGA-BB-4228.svs    TCGA-CQ-6219.svs    TCGA-CV-7235.svs    TCGA-F7-A50G.svs
TCGA-BB-8601.svs    TCGA-CQ-6227.svs    TCGA-CV-7236.svs    TCGA-F7-A61W.svs
TCGA-BB-A5HU.svs    TCGA-CQ-6228.svs    TCGA-CV-7238.svs    TCGA-F7-A624.svs
TCGA-BB-A5HZ.svs    TCGA-CQ-7063.svs    TCGA-CV-7255.svs    TCGA-H7-7774.svs
TCGA-BB-A6UM.svs    TCGA-CQ-7065.svs    TCGA-CV-7263.svs    TCGA-H7-A6C4.svs
TCGA-C9-A47Z.svs    TCGA-CQ-7068.svs    TCGA-CV-7411.svs    TCGA-HD-7831.svs
TCGA-C9-A480.svs    TCGA-CQ-A4C6.svs    TCGA-CV-7413.svs    TCGA-HD-8224.svs
TCGA-CN-4726.svs    TCGA-CQ-A4CH.svs    TCGA-CV-7416.svs    TCGA-HD-8314.svs
TCGA-CN-4729.svs    TCGA-CQ-A4CI.svs    TCGA-CV-7425.svs    TCGA-HD-A4C1.svs
TCGA-CN-4734.svs    TCGA-CR-6477.svs    TCGA-CV-7434.svs    TCGA-HD-A633.svs
TCGA-CN-4737.svs    TCGA-CR-6478.svs    TCGA-CV-A45Q.svs    TCGA-HD-A634.svs
TCGA-CN-4740.svs    TCGA-CR-6480.svs    TCGA-CV-A45R.svs    TCGA-HD-A6HZ.svs
TCGA-CN-4741.svs    TCGA-CR-6481.svs    TCGA-CV-A45U.svs    TCGA-IQ-A61G.svs
TCGA-CN-4742.svs    TCGA-CR-6487.svs    TCGA-CV-A45X.svs    ... and more
```
</details>

### Data Integration Statistics
| Metric | Count |
|--------|-------|
| Total SVS images | 172 |
| Patients with clinical data | 82 |
| **Perfect overlap** | **82** (all clinical patients have images) |
| Images without clinical data | 90 |

### Naming Convention
```
TCGA-CV-7235.svs
     │    │
     │    └─ Patient ID
     └────── Tissue Source Site
```

### Relationship to Other Files
- **Links via:** Filename matches `cases.submitter_id` in clinical data
- **Clinical:** Each image has corresponding clinical data (demographics, staging, treatment)
- **Pathology:** Images are the visual source for pathology_detail.txt (margins, invasion, etc.)
- **Molecular Data:** Same patient, different data modality - enables multi-modal AI
- **Follow-up:** Survival outcomes exist for patients with images

### Tools for Opening SVS Files
| Tool | Type | Platform |
|------|------|----------|
| **OpenSlide** | Python library | Cross-platform |
| **QuPath** | GUI application | Cross-platform |
| **ASAP** | GUI application | Windows/Linux |
| **libvips** | Image processing | Cross-platform |
| **Histolab** | Python library | Cross-platform |

### Research Applications
1. **Deep Learning Classification:** Predict grade, stage, survival from images
2. **Feature Extraction:** Use pretrained CNNs (ResNet, EfficientNet) or ViTs
3. **Multiple Instance Learning (MIL):** Handle gigapixel images
4. **Multi-Modal Learning:** Combine images with genomic data
5. **Tumor Segmentation:** Identify tumor vs stroma vs necrosis
6. **TIL Scoring:** Quantify tumor-infiltrating lymphocytes

---

## 🔗 Data Integration Map

### How All Files Connect

```
                           ┌─────────────────┐
                           │  clinical.txt   │
                           │  (Central Hub)  │
                           │  Patient IDs    │
                           └────────┬────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
   ┌───────────────┐       ┌───────────────┐        ┌───────────────┐
   │ exposure.txt  │       │ follow_up.txt │        │pathology_     │
   │ Risk factors  │       │ Survival/HPV  │        │detail.txt     │
   │ Tobacco/Alc.  │       │ Recurrence    │        │Lymph nodes    │
   └───────────────┘       └───────────────┘        └───────────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │          Patient ID           │
                    │       TCGA-XX-XXXX            │
                    └───────────────┬───────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌─────────────┐              ┌─────────────┐              ┌─────────────┐
│methylation  │              │transcripto- │              │mutations.txt│
│.txt         │◄────────────►│mics.txt     │◄────────────►│MAF format   │
│Epigenetics  │  Correlate   │Gene Express │   Affects    │Somatic vars │
│Beta values  │              │RNA-seq      │              │WXS data     │
└─────────────┘              └─────────────┘              └─────────────┘
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    │
                            ┌───────┴───────┐
                            │  SVS Images   │
                            │  172 WSIs     │
                            │  Histopath.   │
                            └───────────────┘
```

### Sample ID Linking Rules

| File | ID Format | Example | Link Method |
|------|-----------|---------|-------------|
| clinical.txt | TCGA-XX-XXXX | TCGA-CR-7392 | Direct match |
| exposure.txt | TCGA-XX-XXXX | TCGA-CR-7392 | Direct match |
| follow_up.txt | TCGA-XX-XXXX | TCGA-CR-7392 | Direct match |
| pathology_detail.txt | TCGA-XX-XXXX | TCGA-CR-7392 | Direct match |
| methylation.txt | TCGA-XX-XXXX-01 | TCGA-CR-7392-01 | Remove suffix |
| transcriptomics.txt | TCGA-XX-XXXX-01A-XXR-XXXX-07 | TCGA-CR-7392-01A-... | Extract patient ID |
| mutations.txt | TCGA-XX-XXXX-01A-XX... | TCGA-BA-4074-01A-... | Extract patient ID |
| SVS files | TCGA-XX-XXXX.svs | TCGA-CR-7392.svs | Remove extension |

---

## 🎯 Top 5 Research Projects (Best of the Best)

> **Selection Criteria:** These projects were selected based on: (1) Clinical Impact, (2) Scientific Novelty, (3) Multi-Modal Integration, (4) Feasibility with ~82 patients, (5) Publication Potential in top-tier journals, (6) State-of-the-art methodology for 2026.

---

### 🥇 Project 1: Multi-Modal Transformer for Survival Prediction with Cross-Modal Explainability

#### Why This is Best-of-the-Best
This project leverages the **full power of your multi-modal dataset** - something very few research groups have. Most survival prediction papers use only 1-2 modalities. Combining histopathology images + transcriptomics + methylation + mutations + clinical data with modern transformer architectures puts this at the absolute cutting edge. The explainability component (identifying WHICH features across modalities drive predictions) makes it clinically actionable, not just a black box.

#### The Problem
Current HNSC survival prediction relies on crude TNM staging, missing the rich molecular and morphological information that determines individual patient outcomes. Oncologists need to know not just "will this patient survive?" but "WHY is this patient high-risk?" to guide treatment decisions.

#### Your Approach
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-MODAL SURVIVAL TRANSFORMER                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ SVS Images  │  │Transcriptom.│  │ Methylation │  │ Mutations   │        │
│  │ (172 WSIs)  │  │ (20K genes) │  │ (20K genes) │  │ (MAF)       │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         ▼                ▼                ▼                ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Vision      │  │ Gene        │  │ Methylation │  │ Mutation    │        │
│  │ Transformer │  │ Encoder     │  │ Encoder     │  │ Encoder     │        │
│  │ (HIPT/UNI)  │  │ (Pathway)   │  │ (VAE)       │  │ (Binary)    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────┬───────┴────────────────┘               │
│                                   ▼                                        │
│                    ┌──────────────────────────┐                            │
│                    │   CROSS-MODAL ATTENTION  │                            │
│                    │   (Which modality matters │                            │
│                    │    for this patient?)     │                            │
│                    └─────────────┬────────────┘                            │
│                                  ▼                                         │
│                    ┌──────────────────────────┐                            │
│                    │  + Clinical Features     │                            │
│                    │  (Age, Stage, HPV, etc.) │                            │
│                    └─────────────┬────────────┘                            │
│                                  ▼                                         │
│                    ┌──────────────────────────┐                            │
│                    │   SURVIVAL PREDICTION    │                            │
│                    │   (Cox loss / DeepSurv)  │                            │
│                    └─────────────┬────────────┘                            │
│                                  ▼                                         │
│            ┌─────────────────────────────────────────────┐                 │
│            │  OUTPUT: Risk Score + Attention Heatmaps   │                 │
│            │  "High risk because: image region X shows  │                 │
│            │   invasion + EGFR overexpression + CDKN2A  │                 │
│            │   hypermethylation"                        │                 │
│            └─────────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Data Requirements (✓ You Have Everything)
| Component | Requirement | Your Data |
|-----------|-------------|-----------|
| Images | WSI with survival labels | ✓ 172 SVS + survival in clinical.txt |
| Expression | Gene-level counts | ✓ 20,504 genes in transcriptomics.txt |
| Methylation | Beta values | ✓ 20,115 genes in methylation.txt |
| Mutations | Somatic variants | ✓ MAF in mutations.txt |
| Survival | Time + Event | ✓ days_to_death, vital_status in clinical.txt |
| Clinical | Covariates | ✓ Age, stage, HPV status |

#### Technical Implementation
```python
# Key libraries
from lifelines import CoxPHFitter  # Survival analysis baseline
import torch
from torch import nn
from transformers import ViTModel  # Or use UNI/HIPT pretrained histopath models
from torch_geometric.nn import GATConv  # For pathway-aware gene encoding

# Architecture components:
# 1. Image encoder: HIPT or UNI (pretrained on histopathology)
# 2. Gene encoder: Pathway-aware transformer (KEGG/Reactome structure)
# 3. Methylation encoder: Variational autoencoder for dimensionality reduction
# 4. Mutation encoder: Binary matrix + attention over driver genes
# 5. Fusion: Cross-modal attention transformer
# 6. Output: Cox proportional hazards loss
```

#### Expected Outcomes
- **C-index:** 0.75-0.85 (vs 0.65-0.70 for clinical-only models)
- **Interpretable attention maps** showing which image regions/genes matter
- **Risk stratification** into 2-4 prognostic groups

#### Publication Target
- **Journal:** Nature Medicine, Cancer Cell, or Nature Communications
- **Novelty claim:** "First multi-modal transformer integrating histopathology with three omics layers for HNSC survival prediction with cross-modal explainability"

#### Timeline
| Phase | Duration | Tasks |
|-------|----------|-------|
| Phase 1 | 2 weeks | Data preprocessing, feature extraction from SVS |
| Phase 2 | 3 weeks | Individual modality encoders |
| Phase 3 | 3 weeks | Multi-modal fusion architecture |
| Phase 4 | 2 weeks | Explainability (attention visualization, SHAP) |
| Phase 5 | 2 weeks | Validation, ablation studies, paper writing |

---

### 🥈 Project 2: Tumor Microenvironment Characterization via Histopathology-Transcriptomics Integration

#### Why This is Best-of-the-Best
The **tumor microenvironment (TME)** is the hottest topic in cancer research because it determines immunotherapy response. You have a unique opportunity: estimate immune cell infiltration from H&E images AND validate it with transcriptomics-based deconvolution. This dual-validation approach is extremely rare and publishable. With checkpoint inhibitors becoming standard-of-care for HNSC, understanding TME has direct clinical implications.

#### The Problem
Immunotherapy (pembrolizumab, nivolumab) is now first-line for recurrent/metastatic HNSC, but only 15-20% of patients respond. We need better biomarkers to predict who will benefit. TIL (tumor-infiltrating lymphocyte) scoring is done manually by pathologists with high variability. Transcriptomic immune signatures exist but require expensive sequencing.

#### Your Approach
```
┌─────────────────────────────────────────────────────────────────────────────┐
│            DUAL-PATHWAY TME CHARACTERIZATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATHWAY A: Image-Based                 PATHWAY B: Transcriptomics-Based   │
│  ────────────────────────               ─────────────────────────────────  │
│                                                                             │
│  ┌─────────────┐                        ┌─────────────────┐                │
│  │ H&E Slides  │                        │ RNA-seq counts  │                │
│  │ (172 SVS)   │                        │ (20,504 genes)  │                │
│  └──────┬──────┘                        └────────┬────────┘                │
│         │                                        │                         │
│         ▼                                        ▼                         │
│  ┌─────────────────────┐              ┌──────────────────────┐            │
│  │ TIL Detection CNN   │              │ CIBERSORT / xCell    │            │
│  │ - Lymphocyte detect │              │ Immune deconvolution │            │
│  │ - Spatial density   │              │ - 22 immune types    │            │
│  │ - TIL score         │              │ - Signature genes    │            │
│  └──────────┬──────────┘              └───────────┬──────────┘            │
│             │                                     │                        │
│             ▼                                     ▼                        │
│  ┌─────────────────────┐              ┌──────────────────────┐            │
│  │ Image Features:     │              │ Transcriptomic:      │            │
│  │ - CD8+ T cell %     │              │ - CD8+ T cell %      │            │
│  │ - Macrophage %      │              │ - M1/M2 macrophage   │            │
│  │ - TIL spatial dist. │              │ - B cell %           │            │
│  │ - Hot/Cold tumors   │              │ - Cytotoxic score    │            │
│  └──────────┬──────────┘              └───────────┬──────────┘            │
│             │                                     │                        │
│             └─────────────────┬───────────────────┘                        │
│                               ▼                                            │
│               ┌───────────────────────────────┐                            │
│               │  CROSS-VALIDATION & FUSION    │                            │
│               │  - Correlate image vs RNA     │                            │
│               │  - Train image model with     │                            │
│               │    RNA-derived labels         │                            │
│               │  - Combined TME score         │                            │
│               └───────────────┬───────────────┘                            │
│                               ▼                                            │
│               ┌───────────────────────────────┐                            │
│               │  CLINICAL CORRELATIONS        │                            │
│               │  - HPV+ vs HPV- TME           │                            │
│               │  - TME vs survival            │                            │
│               │  - TME vs treatment response  │                            │
│               └───────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Innovation: Using RNA as Ground Truth for Image Model
Most histopathology AI papers train on pathologist annotations (expensive, subjective). You can use transcriptomics-derived immune scores as "ground truth" to train an image-based model - this is self-supervised from your own data!

#### Data Requirements (✓ You Have Everything)
| Component | Requirement | Your Data |
|-----------|-------------|-----------|
| Images | H&E WSI | ✓ 172 SVS files |
| Expression | For CIBERSORT/xCell | ✓ transcriptomics.txt |
| HPV Status | For stratification | ✓ follow_up.txt (HPV ISH) |
| Survival | For prognostic value | ✓ clinical.txt |
| Tumor-Normal | For baseline | ✓ 10 paired samples |

#### Technical Implementation
```python
# Transcriptomics-based immune deconvolution
from scipy.stats import pearsonr
import cibersortx  # or use TIMER2.0, xCell, MCPcounter

# Step 1: Get immune cell fractions from RNA-seq
immune_fractions = cibersort.deconvolute(expression_matrix)
# Output: 22 cell types per sample (CD8 T, CD4 T, Treg, M1/M2 macrophage, etc.)

# Step 2: Train image model to predict these fractions
# Use HoverNet for cell detection + classification
# Or train custom CNN with immune_fractions as regression target

# Step 3: Correlate image-derived vs RNA-derived scores
correlation = pearsonr(image_til_score, rna_til_score)

# Step 4: Define "Immune Hot" vs "Immune Cold" tumors
# Correlate with survival (Kaplan-Meier, log-rank test)
```

#### Expected Outcomes
| Analysis | Expected Finding |
|----------|------------------|
| Image-RNA correlation | r = 0.6-0.8 for TIL scores |
| HPV+ vs HPV- TME | HPV+ tumors are "immune hot" |
| TME survival impact | High TIL = better survival (HR 0.4-0.6) |
| TME heterogeneity | Identify spatial patterns (tumor core vs edge) |

#### Publication Target
- **Journal:** Clinical Cancer Research, Cancer Immunology Research, or JCO Precision Oncology
- **Novelty claim:** "Cross-validated histopathology-transcriptomics approach for tumor microenvironment characterization in HNSC"

---

### 🥉 Project 3: Morpho-Molecular Biomarker Discovery — What Can AI See That Pathologists Can't?

#### Why This is Best-of-the-Best
This project asks a fundamental question: **Which molecular alterations leave visible traces in tissue morphology?** If you can predict TP53 mutation status or EGFR expression from H&E images alone, you've discovered a visual biomarker that could replace expensive molecular tests. This is the holy grail of computational pathology - finding biology hidden in plain sight.

#### The Problem
Molecular testing (sequencing, IHC, FISH) is expensive, requires additional tissue, and takes days-weeks. If certain molecular features can be inferred from routine H&E slides (which are always available), it could democratize precision oncology, especially in resource-limited settings.

#### Your Approach
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                MORPHO-MOLECULAR BIOMARKER DISCOVERY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    SYSTEMATIC SCREENING                               │ │
│  │  For each molecular feature, ask: "Can images predict this?"         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  MOLECULAR TARGETS:                                                         │
│  ─────────────────                                                         │
│                                                                             │
│  From mutations.txt:          From transcriptomics.txt:                    │
│  ├── TP53 mutation (Y/N)      ├── EGFR expression (high/low)              │
│  ├── PIK3CA mutation (Y/N)    ├── EMT signature score                     │
│  ├── CDKN2A deletion (Y/N)    ├── Proliferation (MKI67)                   │
│  ├── NOTCH1 mutation (Y/N)    ├── Hypoxia signature                       │
│  └── Mutation burden (TMB)    └── Immune signature (CD8, FOXP3)           │
│                                                                             │
│  From methylation.txt:        From follow_up.txt:                          │
│  ├── CDKN2A methylation       └── HPV status (Positive/Negative)          │
│  ├── CDH1 methylation                                                      │
│  └── Global hypomethylation                                                │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         PIPELINE                                      │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│       ┌─────────────┐         ┌─────────────────┐                          │
│       │  SVS Image  │────────▶│  Tile Extraction │                         │
│       │  (Gigapixel)│         │  256×256 @ 20x   │                         │
│       └─────────────┘         └────────┬────────┘                          │
│                                        │                                   │
│                                        ▼                                   │
│                          ┌─────────────────────────┐                       │
│                          │  Feature Extraction     │                       │
│                          │  (ResNet50 / UNI /      │                       │
│                          │   CTransPath pretrained)│                       │
│                          └────────────┬────────────┘                       │
│                                       │                                    │
│                                       ▼                                    │
│                          ┌─────────────────────────┐                       │
│                          │  Multiple Instance      │                       │
│                          │  Learning (MIL)         │                       │
│                          │  - ABMIL / CLAM /       │                       │
│                          │    TransMIL             │                       │
│                          └────────────┬────────────┘                       │
│                                       │                                    │
│                                       ▼                                    │
│                          ┌─────────────────────────┐                       │
│                          │  Binary Classification  │                       │
│                          │  e.g., TP53 mut vs WT   │                       │
│                          └────────────┬────────────┘                       │
│                                       │                                    │
│                                       ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  RESULTS MATRIX: Which molecular features are "visible"?              │ │
│  │                                                                       │ │
│  │  Target              │ AUC   │ Interpretation                         │ │
│  │  ────────────────────┼───────┼──────────────────────────────────────  │ │
│  │  HPV status          │ 0.85+ │ ✓ Strong visual signal (known)         │ │
│  │  TP53 mutation       │ 0.70+ │ ✓ Moderate - nuclear atypia?           │ │
│  │  Immune infiltration │ 0.80+ │ ✓ TILs are visible                     │ │
│  │  EMT score           │ 0.65  │ ? Subtle - invasion patterns?          │ │
│  │  EGFR expression     │ 0.60  │ ? May need IHC                         │ │
│  │  Random gene         │ 0.50  │ ✗ No signal (negative control)         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  EXPLAINABILITY:                                                            │
│  ──────────────                                                            │
│  For predictable targets, generate attention heatmaps:                     │
│  "Model predicts TP53 mutation based on these tissue regions..."           │
│  → Discover morphological patterns associated with molecular events        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Scientific Questions
1. **Can we predict HPV status from H&E?** (Known to be possible - validate)
2. **Can we predict TP53 mutation?** (Nuclear atypia hypothesis)
3. **Can we predict immune infiltration?** (TILs are visible)
4. **Can we predict EMT/invasion?** (Morphological invasion patterns)
5. **What is NOT predictable?** (Equally important - defines limits)

#### Data Requirements (✓ You Have Everything)
| Component | Requirement | Your Data |
|-----------|-------------|-----------|
| Images | H&E WSI | ✓ 172 SVS files |
| Mutations | Binary labels | ✓ mutations.txt (TP53, PIK3CA, etc.) |
| Expression | Continuous/binary | ✓ transcriptomics.txt |
| Methylation | Binary (hyper/hypo) | ✓ methylation.txt |
| HPV | Binary label | ✓ follow_up.txt |

#### Expected Outcomes
| Target | Expected AUC | Clinical Impact |
|--------|--------------|-----------------|
| HPV status | 0.85-0.92 | Replace p16 IHC screening |
| High TIL score | 0.80-0.88 | Immunotherapy selection |
| TP53 mutation | 0.65-0.75 | Novel finding if achieved |
| EMT-high | 0.60-0.70 | Metastasis risk marker |

#### Publication Target
- **Journal:** Nature Medicine, Modern Pathology, or NPJ Precision Oncology
- **Novelty claim:** "Systematic morpho-molecular mapping reveals visually predictable genomic alterations in HNSC"

---

### 🏅 Project 4: HPV-Stratified Multi-Omics Molecular Subtyping

#### Why This is Best-of-the-Best
HPV-positive and HPV-negative HNSC are **essentially two different diseases** with different biology, prognosis, and treatment responses. But within each category, there's significant heterogeneity. Current treatment is one-size-fits-all. By creating molecularly-defined subtypes WITHIN HPV+ and HPV- groups, you enable precision medicine. This uses your full multi-omics arsenal and addresses a real clinical gap.

#### The Problem
HPV+ HNSC patients generally do well, but some still die. HPV- HNSC patients generally do poorly, but some survive long-term. We can't predict who. Current staging doesn't capture the molecular heterogeneity within these groups.

#### Your Approach
```
┌─────────────────────────────────────────────────────────────────────────────┐
│               HPV-STRATIFIED MOLECULAR SUBTYPING                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: STRATIFY BY HPV STATUS                                            │
│  ─────────────────────────────                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    All 82 Patients                                  │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│              ┌───────────────┴───────────────┐                              │
│              ▼                               ▼                              │
│  ┌───────────────────────┐       ┌───────────────────────┐                 │
│  │     HPV-POSITIVE      │       │     HPV-NEGATIVE      │                 │
│  │    (~20-30 patients)  │       │    (~50-60 patients)  │                 │
│  │    Better prognosis   │       │    Worse prognosis    │                 │
│  │    Oropharynx-biased  │       │    Tobacco-associated │                 │
│  └───────────┬───────────┘       └───────────┬───────────┘                 │
│              │                               │                              │
│              ▼                               ▼                              │
│  STEP 2: MULTI-OMICS INTEGRATION (per HPV group)                           │
│  ───────────────────────────────────────────────                           │
│                                                                             │
│       ┌─────────────────────────────────────────────────────┐              │
│       │              FOR EACH HPV GROUP:                    │              │
│       │                                                     │              │
│       │   ┌───────────┐ ┌───────────┐ ┌───────────┐        │              │
│       │   │Expression │ │Methylation│ │ Mutations │        │              │
│       │   │  Matrix   │ │  Matrix   │ │  Matrix   │        │              │
│       │   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘        │              │
│       │         │             │             │               │              │
│       │         └─────────────┼─────────────┘               │              │
│       │                       ▼                             │              │
│       │         ┌───────────────────────────┐               │              │
│       │         │  MULTI-OMICS INTEGRATION  │               │              │
│       │         │  - MOFA (recommended)     │               │              │
│       │         │  - iCluster               │               │              │
│       │         │  - SNF (Similarity        │               │              │
│       │         │    Network Fusion)        │               │              │
│       │         └─────────────┬─────────────┘               │              │
│       │                       │                             │              │
│       │                       ▼                             │              │
│       │         ┌───────────────────────────┐               │              │
│       │         │   CONSENSUS CLUSTERING    │               │              │
│       │         │   Identify k subtypes     │               │              │
│       │         │   (k = 2-4 typically)     │               │              │
│       │         └─────────────┬─────────────┘               │              │
│       │                       │                             │              │
│       └───────────────────────┼─────────────────────────────┘              │
│                               ▼                                             │
│  STEP 3: SUBTYPE CHARACTERIZATION                                          │
│  ────────────────────────────────                                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  HPV-NEGATIVE SUBTYPES (Example)                                    │   │
│  │                                                                     │   │
│  │  Subtype    │ Characteristics              │ Prognosis │ Treatment │   │
│  │  ───────────┼──────────────────────────────┼───────────┼───────────│   │
│  │  HPVneg-1   │ High EMT, invasive, TP53 mut │ Poor      │ Aggressive│   │
│  │  "Mesench"  │ Low immune infiltration      │ (HR=2.5)  │ chemoRT   │   │
│  │  ───────────┼──────────────────────────────┼───────────┼───────────│   │
│  │  HPVneg-2   │ High proliferation, EGFR+    │ Intermed. │ EGFR      │   │
│  │  "Prolif"   │ Intact DNA repair            │ (HR=1.5)  │ targeted  │   │
│  │  ───────────┼──────────────────────────────┼───────────┼───────────│   │
│  │  HPVneg-3   │ Immune hot, high TILs        │ Better    │ Immuno-   │   │
│  │  "Immune"   │ Activated immune pathways    │ (HR=0.6)  │ therapy?  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 4: VALIDATION WITH IMAGES                                            │
│  ──────────────────────────────                                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Can subtypes be distinguished from H&E images?                     │   │
│  │  Train classifier: SVS → Molecular Subtype                          │   │
│  │  If AUC > 0.7: Subtypes have morphological correlates!              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Innovation
Most subtyping papers use only expression. You're using expression + methylation + mutations. The addition of epigenetics often reveals subtypes that expression alone misses (epigenetically silenced genes).

#### Data Requirements (✓ You Have Everything)
| Component | Requirement | Your Data |
|-----------|-------------|-----------|
| HPV Status | Stratification | ✓ follow_up.txt |
| Expression | Subtyping input | ✓ transcriptomics.txt |
| Methylation | Subtyping input | ✓ methylation.txt |
| Mutations | Subtyping input | ✓ mutations.txt |
| Survival | Validation | ✓ clinical.txt |
| Images | Visual validation | ✓ 172 SVS files |

#### Technical Implementation
```python
# Multi-omics factor analysis (MOFA)
from mofapy2 import mofa

# Prepare data views
views = {
    'expression': expression_matrix,  # 20K genes × N samples
    'methylation': methylation_matrix,  # 20K genes × N samples
    'mutations': mutation_matrix  # M genes × N samples (binary)
}

# Run MOFA
model = mofa.run_mofa(views, K=15)  # Extract 15 factors
factors = model.get_factors()

# Cluster on factors
from sklearn.cluster import KMeans
subtypes = KMeans(n_clusters=3).fit_predict(factors)

# Validate with survival
from lifelines import KaplanMeierFitter
# Plot survival by subtype - expect significant separation (p < 0.01)
```

#### Expected Outcomes
| HPV Group | Expected Subtypes | Survival Separation |
|-----------|-------------------|---------------------|
| HPV+ | 2 subtypes | Moderate (p < 0.05) |
| HPV- | 3 subtypes | Strong (p < 0.001) |

#### Publication Target
- **Journal:** Cancer Discovery, Cell Reports Medicine, or Genome Medicine
- **Novelty claim:** "Multi-omics molecular subtyping within HPV-stratified HNSC reveals actionable patient subgroups"

---

### 🏅 Project 5: Epigenetic-Transcriptomic Integration: Discovering Silenced Tumor Suppressors and Drug Targets

#### Why This is Best-of-the-Best
You have **matched methylation and expression data** for the same patients - this is gold for integrative analysis. When a gene has high promoter methylation AND low expression, it's likely **epigenetically silenced**. If that gene is a tumor suppressor, you've found a therapeutic target (DNA methyltransferase inhibitors like azacitidine can reactivate it). This project can discover new drug targets with immediate translational potential.

#### The Problem
Many tumor suppressors are silenced by promoter methylation rather than mutation. These are invisible to standard sequencing but could be reactivated with epigenetic drugs. We need to systematically identify these silenced genes in HNSC.

#### Your Approach
```
┌─────────────────────────────────────────────────────────────────────────────┐
│        EPIGENETIC-TRANSCRIPTOMIC INTEGRATION ANALYSIS                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: PAIRED SAMPLE ANALYSIS (Tumor vs Normal)                          │
│  ────────────────────────────────────────────────                          │
│                                                                             │
│  You have ~10-14 tumor-normal pairs with BOTH methylation AND expression   │
│                                                                             │
│  For each gene:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │    METHYLATION                           EXPRESSION                 │   │
│  │    (Beta values)                         (RNA counts)               │   │
│  │                                                                     │   │
│  │    Normal: 0.2 ──────────────────────▶  Normal: 5000               │   │
│  │    Tumor:  0.8 ──────────────────────▶  Tumor:  200                │   │
│  │                                                                     │   │
│  │    ΔMethylation = +0.6                  ΔExpression = -4800        │   │
│  │    (Hypermethylated)                    (Downregulated)            │   │
│  │                                                                     │   │
│  │    CONCLUSION: Epigenetically silenced in tumor!                   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: GENOME-WIDE SCREEN                                                │
│  ──────────────────────────                                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │   For all 20,000 genes:                                            │   │
│  │                                                                     │   │
│  │   1. Calculate correlation: methylation vs expression              │   │
│  │      (across all samples)                                          │   │
│  │                                                                     │   │
│  │   2. Identify genes with:                                          │   │
│  │      - Strong negative correlation (r < -0.5)                      │   │
│  │      - Tumor hypermethylation (Δβ > 0.2)                          │   │
│  │      - Tumor downregulation (FC < 0.5)                            │   │
│  │                                                                     │   │
│  │   3. Cross-reference with:                                         │   │
│  │      - Known tumor suppressors (COSMIC CGC)                        │   │
│  │      - Druggable genes (DGIdb)                                    │   │
│  │      - Genes not mutated in your cohort (alternative silencing)   │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 3: EXPECTED DISCOVERIES                                              │
│  ────────────────────────────                                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Category A: Known Silenced Genes (Validation)                     │   │
│  │  ─────────────────────────────────────────────                     │   │
│  │  Gene     │ Function           │ Expected │ Clinical              │   │
│  │  ─────────┼────────────────────┼──────────┼──────────────────────│   │
│  │  CDKN2A   │ Cell cycle control │ Silenced │ Known in HNSC        │   │
│  │  RASSF1A  │ RAS signaling      │ Silenced │ Known in HNSC        │   │
│  │  CDH1     │ E-cadherin/EMT     │ Silenced │ Invasion marker      │   │
│  │  MGMT     │ DNA repair         │ Silenced │ Chemo sensitivity    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Category B: Novel Discoveries                                      │   │
│  │  ─────────────────────────────                                     │   │
│  │  Genes with hypermethylation + downregulation that:                │   │
│  │  - Are NOT known to be silenced in HNSC                           │   │
│  │  - Have tumor suppressor function in other cancers                 │   │
│  │  - Could be reactivated with DNMT inhibitors                       │   │
│  │                                                                     │   │
│  │  → These are your NOVEL FINDINGS for publication!                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 4: CLINICAL CORRELATIONS                                             │
│  ─────────────────────────────                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  For each silenced gene, test:                                     │   │
│  │                                                                     │   │
│  │  1. Survival impact:                                               │   │
│  │     High methylation → Worse survival?                             │   │
│  │     Low expression → Worse survival?                               │   │
│  │                                                                     │   │
│  │  2. Clinical associations:                                         │   │
│  │     Associated with stage? HPV status? Tobacco use?                │   │
│  │                                                                     │   │
│  │  3. Therapeutic potential:                                         │   │
│  │     Is gene product druggable if reactivated?                      │   │
│  │     Is there a synthetic lethal partner?                           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 5: VISUALIZE IN IMAGES                                               │
│  ───────────────────────────                                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Can silencing status be predicted from histopathology?            │   │
│  │                                                                     │   │
│  │  Example: CDH1 silencing (E-cadherin loss)                         │   │
│  │  - E-cadherin loss → EMT → invasion pattern visible in H&E?       │   │
│  │  - Train model: SVS → CDH1 methylation status                      │   │
│  │  - If predictable: morphological biomarker!                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Key Analysis Code
```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from lifelines import CoxPHFitter

# Load data
meth = pd.read_csv('methylation.txt', sep='\t', index_col=0)
expr = pd.read_csv('transcriptomics.txt', sep='\t', index_col=0)

# Find common genes and samples
common_genes = meth.index.intersection(expr.index)
# Match samples (remove -01/-11 suffix, match patient IDs)

# For each gene: correlation between methylation and expression
correlations = {}
for gene in common_genes:
    r, p = spearmanr(meth.loc[gene], expr.loc[gene])
    correlations[gene] = {'correlation': r, 'pvalue': p}

# Find epigenetically silenced genes
silenced = {g: c for g, c in correlations.items() 
            if c['correlation'] < -0.5 and c['pvalue'] < 0.01}

# Cross-reference with tumor suppressor database
tumor_suppressors = load_cosmic_tumor_suppressors()
novel_silenced = [g for g in silenced if g not in tumor_suppressors]

# Survival analysis for top candidates
clinical = pd.read_csv('clinical.txt', sep='\t')
for gene in top_candidates:
    # Methylation high vs low survival
    cph = CoxPHFitter()
    cph.fit(clinical[['methylation_' + gene, 'days_to_death', 'vital_status']])
```

#### Expected Deliverables
| Output | Description |
|--------|-------------|
| Silenced gene list | 50-200 genes with consistent hypermethylation-downregulation |
| Novel candidates | 10-20 genes not previously reported in HNSC |
| Prognostic markers | 5-10 genes where silencing predicts survival |
| Drug targets | 3-5 genes that could be reactivated with DNMT inhibitors |

#### Publication Target
- **Journal:** Clinical Cancer Research, Molecular Cancer, or Epigenetics
- **Novelty claim:** "Integrated methylation-transcriptomics analysis identifies novel epigenetically silenced tumor suppressors in HNSC"

---

## 📊 Project Comparison Matrix

| Criteria | Project 1 | Project 2 | Project 3 | Project 4 | Project 5 |
|----------|-----------|-----------|-----------|-----------|-----------|
| | Multi-Modal Survival | TME Characterization | Morpho-Molecular | HPV Subtyping | Epigenetic Integration |
| **Clinical Impact** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Novelty** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Technical Difficulty** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Uses All Modalities** | ✅ All 5 | ✅ Images + RNA | ✅ Images + Omics | ✅ 3 Omics | ✅ Meth + RNA |
| **Publication Tier** | Nature Medicine | Clin Cancer Res | Nature Medicine | Cancer Discovery | Mol Cancer |
| **Timeline** | 10-12 weeks | 6-8 weeks | 8-10 weeks | 6-8 weeks | 4-6 weeks |
| **Deep Learning** | Heavy | Moderate | Heavy | Light | None |
| **Sample Size OK?** | ✅ Transfer learning | ✅ Yes | ✅ MIL handles it | ⚠️ Borderline | ✅ Paired analysis |

## 🚀 Recommended Starting Point

**Start with Project 5 (Epigenetic Integration)** as a warm-up:
- Fastest to complete (4-6 weeks)
- No deep learning required
- Guaranteed results (methylation-expression correlation is established biology)
- Builds foundation for other projects

**Then proceed to Project 1 or 3** for maximum impact:
- These are the "hero" projects for high-tier publications
- Project 5's findings can be incorporated into these

---

## 📚 References & Resources

### TCGA Publications
- TCGA HNSC marker paper: Lawrence et al., Nature 2015
- GDC Data Portal: https://portal.gdc.cancer.gov/

### Technical Resources
- OpenSlide (SVS reading): https://openslide.org/
- PyTorch for medical imaging: MONAI
- Multi-modal learning: Perceiver, CLIP

### Python Libraries for This Data
```python
# Tabular data
import pandas as pd

# SVS images
import openslide
from histolab.slide import Slide

# Deep learning
import torch
from torchvision import transforms

# Bioinformatics
from lifelines import CoxPHFitter  # Survival analysis
import scanpy as sc  # Single-cell tools work for bulk too
```

---

## 📝 Notes

- Missing values are marked as `'--` in text files
- All coordinates in mutations.txt use GRCh37/hg19 reference
- Beta values in methylation range 0-1 (proportion methylated)
- Expression counts are raw (not normalized) - normalize before analysis
- SVS files require specialized libraries (OpenSlide) to read

---

*Documentation generated for TCGA-HNSC multi-modal cancer dataset*
*Last updated: January 29, 2026*

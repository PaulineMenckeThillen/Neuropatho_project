# DJ-1 Overexpression Defines a TCA/SDHB-Driven Metabolic Signature in IDH-Wildtype Glioblastoma

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![DPI](https://img.shields.io/badge/figures-600_DPI_TIFF-success.svg)](#reproducing-manuscript-figures)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete, self-contained codebase and curated source datasets to reproduce all figures and statistical analyses from the manuscript:

> **DJ-1 Overexpression Defines a TCA/SDHB-Driven Metabolic Signature in IDH-Wildtype Glioblastoma**  
> *Pauline Mencke et al.*

---

## Table of Contents
- [Scientific Overview](#scientific-overview)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Reproducing Manuscript Figures](#reproducing-manuscript-figures)
- [Panel-by-Panel Figure Mapping](#panel-by-panel-figure-mapping)
- [Digital Pathology Image Segmentation (MATLAB)](#digital-pathology-image-segmentation-matlab)
- [Data Availability & Provenance](#data-availability--provenance)
- [Citation & Contact](#citation--contact)

---

## Scientific Overview

Glioblastoma (IDH-wildtype) remains among the most aggressive and lethal primary brain malignancies. This study investigates the oncogenic and metabolic driver role of **DJ-1 (encoded by *PARK7*)** across multi-modal diagnostic cohorts:

1. **Neuropathological & Semi-Quantitative IRS Validation**: Semi-quantitative immunoreactive scoring (IRS) across human neuropathology autopsy and biopsy cohorts confirms significant overexpression of DJ-1 in IDH-wildtype glioblastomas compared to IDH-mutant astrocytomas, oligodendrogliomas, and non-neoplastic control brain tissue.
2. **Automated Digital Image Analysis**: High-throughput whole-slide tile analysis (red AEC chromogen for DJ-1 and brown DAB for GFAP) with custom computer vision algorithms for cell segmentation, stain deconvolution, and immunoreactivity quantification.
3. **In Silico Transcriptomic Profiling & Survival Analysis**: Utilizing the TCGA Diffuse Glioma cohort (*n* = 667), we demonstrate that *PARK7* transcript expression is markedly elevated in IDH-wildtype tumors and stratifies patient overall survival (log-rank *p* = 0.0045).
4. **TCA Cycle & Mitochondrial Pathway Coupling**: Interrogation of TCA cycle enzymes reveals distinct transcriptional dysregulation of *CS*, *ACO2*, *IDH1*, *IDH2*, *OGDH*, and *SDHB*.
5. **Proteogenomic & CPTAC3 Translation**: Validation in the CPTAC3 Glioblastoma discovery cohort confirms that DJ-1 protein abundance directly correlates with SDHB expression (*r* = 0.219, *p* = 0.0426), substantiating mitochondrial metabolic rewiring.
6. **Quantitative Tissue 1H-NMR Metabolomics**: Independent validation via metabolomic profiling (MetaboLights MTBLS3873) reveals characteristic shifts in key intermediates (citrate, aspartate, glutamate, 2-hydroxyglutarate, succinate, fumarate).

---

## Repository Structure

```text
Neuropatho_project/
├── MasterScriptManuscript.py   # Primary automated Python script reproducing all figures at 600 DPI
├── requirements.txt           # Python dependency specification
├── .gitignore                 # Standard ignore file for generated raster caches
├── README.md                  # Comprehensive documentation (this file)
│
├── main_AEC.m                 # MATLAB driver for DJ-1 (AEC red chromogen) tile analysis
├── main_GFAP.m                # MATLAB driver for GFAP (DAB brown chromogen) tile analysis
├── Image_analysis_AEC.m       # Segmentation, RGB thresholding & morphological filtering for AEC
├── Image_analysis_DAB.m       # Segmentation, RGB thresholding & morphological filtering for DAB
│
├── source_data/               # Curated, de-identified datasets (~22.5 MB total)
│   ├── ihc/
│   │   └── AuswertungAECGFAPscans.xlsx            # Manual IRS scoring & region metadata
│   ├── image_analysis/
│   │   ├── CaseIDs.xlsx                           # De-identified autopsy case identifier mapping
│   │   ├── AEC_Analysis_Results.csv               # Aggregated digital cellular AEC measurements
│   │   └── GFAP_DAB_Analysis_Results.csv          # Aggregated digital cellular DAB measurements
│   ├── images/
│   │   ├── Graphical_Abstract.png                 # Master schematic artwork
│   │   ├── Figure 2.jpg                           # Acquisition workflow diagram
│   │   └── Figure 3.png                           # 9-panel histology & segmentation composite
│   ├── tcga_genomics/
│   │   ├── data_clinical_patient.txt              # TCGA Diffuse Glioma clinical & survival data
│   │   ├── data_mutations.txt                     # Mutation calls (IDH1, IDH2, ATRX, TP53)
│   │   ├── data_mrna_seq_tpm.txt                  # Gene expression matrix (TPM)
│   │   └── data_mrna_seq_read_counts.txt          # Raw gene read counts for DESeq2 modeling
│   ├── cptac_proteomics/
│   │   ├── all_subtypes.v5.1.tsv                  # CPTAC3 glioblastoma molecular subtyping
│   │   ├── PDC_study_biospecimen_05082026_075912.tsv # PDC specimen metadata
│   │   └── CPTAC3_Glioblastoma_Multiforme_Proteome.tmt11.tsv # TMT11 proteomic abundance matrix
│   ├── metabolomics/
│   │   ├── s_MTBLS3873.txt                        # MetaboLights MTBLS3873 sample metadata
│   │   ├── clinical_metadata_cleaned.csv          # Patient clinical parameters
│   │   ├── differential_metabolites_ST1_full.csv  # Differential metabolite statistical table
│   │   └── figure7_metabolomics_statistics.csv    # Metabolite group statistics
│   └── statistical_tables/
│       ├── Table_DESeq2_TCA_Cycle_Enzymes_PanGlioma_All_IDH_WT_vs_Mut.csv
│       └── Table_PARK7_TCA_Correlations_AllGliomas.csv
│
└── figures_output/            # Destination for generated 600 DPI .tiff, .png, and vector .pdf files
    └── .gitkeep
```

---

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/PaulineMenckeThillen/Neuropatho_project.git
cd Neuropatho_project
```

### 2. Set Up a Python Virtual Environment
We recommend Python 3.9, 3.10, 3.11, or 3.12:
```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Reproducing Manuscript Figures

The master execution controller `MasterScriptManuscript.py` regenerates all manuscript figures at **publication standard (600 DPI)** in uncompressed/LZW-compressed TIFF (`.tiff`), high-resolution PNG (`.png`), and publication-ready vector PDF (`.pdf`).

### Generate All Figures in a Single Command
```bash
python MasterScriptManuscript.py
```
*Expected runtime: ~2 minutes. All files will be exported to `figures_output/`.*

### Generate Specific Figures
You can select individual figures using the `--figure` argument:
```bash
# Generate Graphical Abstract
python MasterScriptManuscript.py --figure ga

# Generate Figure 1 (IRS Scoring)
python MasterScriptManuscript.py --figure 1

# Generate Figure 4 (Digital Image Analysis)
python MasterScriptManuscript.py --figure 4

# Generate Figure 5 (TCGA Transcriptomics & Survival)
python MasterScriptManuscript.py --figure 5

# Generate Figure 6 (CPTAC3 Proteomics)
python MasterScriptManuscript.py --figure 6

# Generate Figure 7 (1H-NMR Tissue Metabolomics)
python MasterScriptManuscript.py --figure 7

# Generate Extended Figure 5 (DESeq2 normalized counts alternative)
python MasterScriptManuscript.py --figure deseq2
```

### Custom Options
```bash
# Export to a custom output directory at 600 DPI:
python MasterScriptManuscript.py --figure all --output-dir ./my_figures --dpi 600
```

---

## Panel-by-Panel Figure Mapping

| Manuscript Figure | Description | Key Panels | Source Datasets |
|---|---|---|---|
| **Graphical Abstract** | Multi-omics schematic of DJ-1 driven metabolic reprogramming | Comprehensive pathway model | `source_data/images/Graphical_Abstract.png` |
| **Figure 1** | Semi-quantitative Immunoreactive Scoring (IRS) | **A:** Frontal lobe control vs. IDH-wt GBM<br>**B:** Tumor core vs. infiltration vs. distant parenchyma<br>**C:** IDH-wt vs. IDH-mut vs. Control<br>**D:** Diagnosis subtype stratification | `source_data/ihc/AuswertungAECGFAPscans.xlsx` |
| **Figure 2** | Digital pathology acquisition workflow | Whole-slide scanning, tile export, AEC/DAB color deconvolution | `source_data/images/Figure 2.jpg` |
| **Figure 3** | Histological segmentation and multi-channel cellular classification | **A–C:** HE morphological overview<br>**D–F:** DJ-1 (AEC red) stain & cell boundaries<br>**G–I:** GFAP (DAB brown) stain & cell boundaries | `source_data/images/Figure 3.png` |
| **Figure 4** | Automated digital image analysis across autopsy cases | **A–B:** Mean cellular intensity (AEC / DAB)<br>**C–D:** Total segmented cell counts<br>**E–F:** Integrated immunoreactive total sums | `source_data/image_analysis/AEC_Analysis_Results.csv`<br>`source_data/image_analysis/GFAP_DAB_Analysis_Results.csv` |
| **Figure 5** | In silico transcriptomics & overall survival (TCGA) | **A:** *PARK7* across glioma subtypes<br>**B:** Kaplan-Meier survival curve (*p* = 0.0045)<br>**C–H:** TCA enzymes (*CS*, *ACO2*, *IDH1*, *IDH2*, *OGDH*, *SDHB*) | `source_data/tcga_genomics/data_mrna_seq_tpm.txt`<br>`source_data/tcga_genomics/data_clinical_patient.txt`<br>`source_data/tcga_genomics/data_mutations.txt` |
| **Figure 6** | Proteogenomic correlation & CPTAC3 translational validation | **A:** Top 15 metabolic enzymes correlated with *PARK7*<br>**B:** CPTAC3 protein abundance scatter plot (*PARK7* vs *SDHB*, *r* = 0.219, *p* = 0.0426) | `source_data/statistical_tables/Table_PARK7_TCA_Correlations_AllGliomas.csv`<br>`source_data/cptac_proteomics/CPTAC3_Glioblastoma_Multiforme_Proteome.tmt11.tsv` |
| **Figure 7** | Quantitative 1H-NMR tissue metabolomics | **A–F:** Citrate, Aspartate, Glutamate, 2-HG, Succinate, Fumarate in IDH-wt vs. IDH-mut | `source_data/metabolomics/figure7_metabolomics_statistics.csv`<br>`source_data/metabolomics/differential_metabolites_ST1_full.csv` |

---

## Digital Pathology Image Segmentation (MATLAB)

For users who wish to rerun the low-level tile segmentation pipeline on raw QuPath exported whole-slide image tiles:

1. Place raw `.png` or `.tif` image tiles into `./tiles` (or `./source_data/tiles_aec` / `./source_data/tiles_gfap`), or configure the environment variables:
   ```bash
   # Optional environment overrides:
   set QUPATH_AEC_TILES=path\to\aec_tiles
   set QUPATH_GFAP_TILES=path\to\gfap_tiles
   ```
2. In MATLAB, run:
   ```matlab
   % Run AEC (DJ-1) tile analysis:
   run('main_AEC.m')

   % Run DAB (GFAP) tile analysis:
   run('main_GFAP.m')
   ```
3. The scripts invoke `Image_analysis_AEC.m` and `Image_analysis_DAB.m`, applying RGB color channel thresholds, morphological opening/closing filters, watershed cell isolation, and export feature matrices (`AEC_Analysis_Results.csv` and `GFAP_DAB_Analysis_Results.csv`).

---

## Data Availability & Provenance

All source data included in `source_data/` are strictly anonymized and de-identified:
- **Human Neuropathology Autopsy Data**: Clinical autopsy and biopsy identifiers have been mapped to research codes (`A10/24`, `A21/24`, `A28/23`, `A28/24`) in compliance with local institutional ethics board regulations.
- **TCGA Diffuse Glioma**: Downloaded from the NCI Genomic Data Commons (GDC) / cBioPortal (TCGA Pan-Glioma, Cell 2016).
- **CPTAC3 Glioblastoma Discovery Study**: Protein abundance data retrieved from the NCI Proteomic Data Commons (PDC Study ID: PDC000204).
- **MetaboLights 1H-NMR Tissue Metabolomics**: Study identifier **MTBLS3873** (*Metabolic characterization of human IDH-mutant and IDH-wildtype gliomas*).

---

## Citation & Contact

If you use the code or data in this repository, please cite:

```bibtex
@article{mencke2026dj1,
  title={DJ-1 Overexpression Defines a TCA/SDHB-Driven Metabolic Signature in IDH-Wildtype Glioblastoma},
  author={Mencke, Pauline and colleagues},
  journal={Neuropathology and Applied Neurobiology},
  year={2026}
}
```

For correspondence or technical questions regarding code and data reproduction, please open an issue in this repository or contact the corresponding author.

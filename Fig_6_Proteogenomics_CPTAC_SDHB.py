"""
================================================================================
FIGURE 6: PROTEOGENOMIC TARGET RANKING & CPTAC3 PROTEOMICS CORRELATION
Manuscript: "DJ-1 Overexpression Defines a TCA/SDHB-Driven Metabolic Signature 
             in IDH-Wildtype Glioblastoma"
Author: Pauline Mencke et al.
Generates:
  - Fig 6A: Pearson correlation ranking of top 15 metabolic enzymes with PARK7
  - Fig 6B: Translational validation in CPTAC3 glioblastoma discovery cohort
            showing positive correlation between SDHB and PARK7 protein abundance
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------------------------------------------------------
# Paths and Aesthetics Configuration
# -----------------------------------------------------------------------------
PUBLICATION_OUT = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\Publication_Figures'
DIR_TCGA = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\InSilicoTranscriptomicsAndProteomics\difg_tcga_gdc\difg_tcga_gdc'
DIR_CPTAC_PROT = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\Proteomics\InSilicoProteomics\CPTAC_analysis\CPTAC3_Glioblastoma_Multiforme_Proteome.tmt11.tsv'
DIR_CPTAC_META = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\Proteomics\InSilicoProteomics\CPTAC_analysis\Clinical_Data_Subtypes\Clinical_Data_Subtypes\all_subtypes.v5.1.tsv'
DIR_CPTAC_MAP = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\Proteomics\InSilicoProteomics\CPTAC_analysis\PDC_study_biospecimen_05082026_075912.tsv'

os.makedirs(PUBLICATION_OUT, exist_ok=True)

sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['pdf.fonttype'] = 42

def get_sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "(n.s.)"

# -----------------------------------------------------------------------------
# PANEL A: Global Metabolic Network Ranking (TCGA)
# -----------------------------------------------------------------------------
print("Generating Figure 6A (Metabolic Network Ranking)...")
if os.path.exists(DIR_TCGA):
    df_mrna = pd.read_csv(os.path.join(DIR_TCGA, 'data_mrna_seq_tpm.txt'), sep='\t', comment='#', low_memory=False)
    df_mrna = df_mrna.set_index(df_mrna.columns[0])
    df_mrna = df_mrna.loc[~df_mrna.index.duplicated()].T.reset_index().rename(columns={'index': 'PATIENT_ID'})
    df_mrna['PATIENT_ID'] = df_mrna['PATIENT_ID'].astype(str).str[0:12]

    entrez = {
        '1431': 'CS', '50': 'ACO2', '4967': 'OGDH', '6390': 'SDHB', '11315': 'PARK7',
        '2023': 'ENO1', '7167': 'TPI1', '3945': 'LDHB', '4191': 'MDH2', '3099': 'HK2',
        '5213': 'PFKM', '3417': 'IDH1', '3418': 'IDH2', '3939': 'LDHA', '5211': 'PFKL',
        '2597': 'GAPDH', '6391': 'SDHC', '8803': 'SUCLG2'
    }
    df_mrna = df_mrna.rename(columns=entrez)

    for c in list(entrez.values()):
        if c in df_mrna.columns: df_mrna[c] = pd.to_numeric(df_mrna[c], errors='coerce')

    pathways = {
        'Glycolysis': ['ENO1', 'TPI1', 'LDHB', 'HK2', 'PFKM', 'LDHA', 'PFKL', 'GAPDH'],
        'TCA cycle': ['SDHB', 'MDH2', 'IDH2', 'SDHC', 'ACO2', 'SUCLG2']
    }

    res = []
    for p_name, genes in pathways.items():
        for g in genes:
            if g in df_mrna.columns:
                df_c = df_mrna[['PARK7', g]].dropna()
                r, p = stats.pearsonr(df_c['PARK7'], df_c[g])
                res.append({'Gene': g, 'Pathway': p_name, 'r': r, 'p': p, 'abs_r': abs(r)})

    res_df = pd.DataFrame(res).sort_values('abs_r', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.barplot(data=res_df, x='r', y='Gene', hue='Pathway', palette={'Glycolysis': '#D35400', 'TCA cycle': '#2E86C1'},
                dodge=False, edgecolor='black', ax=ax)
    for i, row in enumerate(res_df.itertuples()):
        ax.text(row.r + (0.015 if row.r > 0 else -0.015), i, get_sig_label(row.p), va='center',
                ha='left' if row.r > 0 else 'right')
    ax.axvline(0, color='black')
    ax.set_title("Potential Drug Targets: PARK7 Metabolic Network\nTop 15 strongest correlations globally")
    ax.set_xlabel("Pearson Correlation (r) with PARK7")
    plt.tight_layout()
    out_6a = os.path.join(PUBLICATION_OUT, "Fig_6A_Target_Ranking.png")
    plt.savefig(out_6a, dpi=300)
    plt.close()
    print(f"Saved: {out_6a}")

# -----------------------------------------------------------------------------
# PANEL B: Translational Validation in CPTAC3 Proteomics
# -----------------------------------------------------------------------------
print("Generating Figure 6B (CPTAC Proteomics Correlation)...")
if os.path.exists(DIR_CPTAC_MAP) and os.path.exists(DIR_CPTAC_PROT) and os.path.exists(DIR_CPTAC_META):
    df_map = pd.read_csv(DIR_CPTAC_MAP, sep='\t')
    df_prot = pd.read_csv(DIR_CPTAC_PROT, sep='\t').iloc[3:].set_index('Gene')
    df_meta = pd.read_csv(DIR_CPTAC_META, sep='\t')

    df_prot.columns = [c.replace(' Log Ratio', '').strip() for c in df_prot.columns]

    idh_wt_cases = df_meta[(df_meta['IDH1_alter_status'].isna()) & (
            (df_meta['TERT_alter_status'] == 'Promoter') | (df_meta['chr7_amp'] == True) | (
            df_meta['chr10_del'] == True))]['case'].unique().tolist()
    wt_aliquots = [a for a in df_map[df_map['Case Submitter ID'].isin(idh_wt_cases)]['Aliquot Submitter ID'] if
                   a in df_prot.columns]

    df_cptac = df_prot[wt_aliquots].apply(pd.to_numeric, errors='coerce').T
    corr_df = df_cptac[['PARK7', 'SDHB']].dropna()

    r, p_val = stats.pearsonr(corr_df['SDHB'], corr_df['PARK7'])

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.regplot(data=corr_df, x='SDHB', y='PARK7',
                scatter_kws={'alpha': 0.6, 'color': '#2c3e50', 'edgecolor': 'w', 's': 50},
                line_kws={'color': '#d62728', 'linestyle': '--'}, ax=ax)
    ax.set_title(f"Protein Correlation: SDHB vs PARK7\nPearson r = {r:.3f}, p = {p_val:.4f}", fontweight='bold')
    ax.set_xlabel("SDHB Abundance (Log Ratio)")
    ax.set_ylabel("PARK7 Abundance (Log Ratio)")
    plt.tight_layout()
    out_6b = os.path.join(PUBLICATION_OUT, "Fig_6B_CPTAC_Correlation.png")
    plt.savefig(out_6b, dpi=300)
    plt.close()
    print(f"Saved: {out_6b}")

print("\nFigure 6 generation completed successfully!")

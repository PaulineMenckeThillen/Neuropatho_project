"""
================================================================================
FIGURE 5: TCGA TRANSCRIPTOMICS & OVERALL SURVIVAL ANALYSIS
Manuscript: "DJ-1 Overexpression Defines a TCA/SDHB-Driven Metabolic Signature 
             in IDH-Wildtype Glioblastoma"
Author: Pauline Mencke et al.
Generates:
  - Fig 5A: PARK7 (DJ-1) mRNA Expression in IDH-Wildtype vs. IDH-Mutant Glioma
  - Fig 5B: Overall Survival Kaplan-Meier Curve by Global PARK7 mRNA Expression
  - Fig 5C: Citrate Synthase (CS) mRNA Expression
  - Fig 5D: Aconitase 2 (ACO2) mRNA Expression
  - Fig 5E: Isocitrate Dehydrogenase 1 (IDH1) mRNA Expression
  - Fig 5F: Isocitrate Dehydrogenase 2 (IDH2) mRNA Expression
  - Fig 5G: Oxoglutarate Dehydrogenase (OGDH / aKGDH) mRNA Expression
  - Fig 5H: Succinate Dehydrogenase Subunit B (SDHB) mRNA Expression
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from lifelines import KaplanMeierFitter, logrank_test

# -----------------------------------------------------------------------------
# Paths and Aesthetics Configuration
# -----------------------------------------------------------------------------
PUBLICATION_OUT = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\Publication_Figures'
DIR_TCGA = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\InSilicoTranscriptomicsAndProteomics\difg_tcga_gdc\difg_tcga_gdc'
os.makedirs(PUBLICATION_OUT, exist_ok=True)

sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['pdf.fonttype'] = 42

def get_sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "(n.s.)"

def format_pval_sci(p):
    if p == 0: return "0"
    base, exp = f"{p:.2e}".split('e')
    return f"{base}e{int(exp):02d}"

# -----------------------------------------------------------------------------
# Load & Process TCGA GDC Data
# -----------------------------------------------------------------------------
if not os.path.exists(DIR_TCGA):
    print(f"Error: {DIR_TCGA} not found.")
    exit(1)

df_mrna = pd.read_csv(os.path.join(DIR_TCGA, 'data_mrna_seq_tpm.txt'), sep='\t', comment='#', low_memory=False)
df_clin_tcga = pd.read_csv(os.path.join(DIR_TCGA, 'data_clinical_patient.txt'), sep='\t', skiprows=4)
df_mut = pd.read_csv(os.path.join(DIR_TCGA, 'data_mutations.txt'), sep='\t', comment='#', low_memory=False)

df_mrna = df_mrna.set_index(df_mrna.columns[0])
df_mrna = df_mrna.loc[~df_mrna.index.duplicated()].T.reset_index().rename(columns={'index': 'PATIENT_ID'})
df_mrna['PATIENT_ID'] = df_mrna['PATIENT_ID'].astype(str).str[0:12]

entrez = {
    '1431': 'CS', '50': 'ACO2', '4967': 'OGDH', '6390': 'SDHB', '11315': 'PARK7',
    '3417': 'IDH1', '3418': 'IDH2', '2023': 'ENO1', '7167': 'TPI1', '3945': 'LDHB',
    '4191': 'MDH2', '3099': 'HK2', '5213': 'PFKM', '3939': 'LDHA', '5211': 'PFKL',
    '2597': 'GAPDH', '6391': 'SDHC', '8803': 'SUCLG2'
}
df_mrna = df_mrna.rename(columns=entrez)

mut_ids = df_mut[df_mut['Hugo_Symbol'].isin(['IDH1', 'IDH2'])]['Tumor_Sample_Barcode'].str[0:12].unique()
df_tcga = pd.merge(df_mrna, df_clin_tcga.rename(columns={'Patient_ID': 'PATIENT_ID'}), on='PATIENT_ID', how='inner')
df_tcga['IDH_Status'] = df_tcga['PATIENT_ID'].apply(lambda pid: 'IDH Mutant' if pid in mut_ids else 'IDH Wildtype')

for c in list(entrez.values()):
    if c in df_tcga.columns: df_tcga[c] = pd.to_numeric(df_tcga[c], errors='coerce')

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def plot_tcga_box(gene, name, fname):
    df_c = df_tcga[['IDH_Status', gene]].dropna()
    wt, mut = df_c[df_c['IDH_Status'] == 'IDH Wildtype'][gene], df_c[df_c['IDH_Status'] == 'IDH Mutant'][gene]
    _, p = stats.mannwhitneyu(wt, mut, alternative='two-sided')

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.boxplot(data=df_c, x='IDH_Status', y=gene, order=['IDH Wildtype', 'IDH Mutant'],
                palette={'IDH Wildtype': '#263433', 'IDH Mutant': '#81A09A'}, showfliers=False, width=0.5, ax=ax)
    sns.stripplot(data=df_c, x='IDH_Status', y=gene, order=['IDH Wildtype', 'IDH Mutant'], color='#707070', size=5,
                  alpha=0.9, jitter=True, ax=ax)
    ax.set_title(f"{name} Expression: IDH Status\nMann-Whitney U p-value: {format_pval_sci(p)} ({get_sig_label(p)})")
    ax.set_ylabel(f"{name} mRNA (TPM)")
    plt.tight_layout()
    out_path = os.path.join(PUBLICATION_OUT, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# Figure 5A: PARK7
plot_tcga_box('PARK7', 'PARK7 (DJ-1)', "Fig_5A_PARK7_IDH.png")

# Figure 5B: Overall Survival
surv_df = df_tcga[['PARK7', 'OS_MONTHS', 'OS_STATUS']].copy()
surv_df['time'] = pd.to_numeric(surv_df['OS_MONTHS'], errors='coerce')
surv_df['event'] = surv_df['OS_STATUS'].apply(lambda x: 1 if '1' in str(x) or 'DECEASED' in str(x).upper() else 0)
surv_df = surv_df.dropna()

med = surv_df['PARK7'].median()
high, low = surv_df[surv_df['PARK7'] >= med], surv_df[surv_df['PARK7'] < med]
res = logrank_test(high['time'], low['time'], high['event'], low['event'])

fig, ax = plt.subplots(figsize=(7, 6))
kmf = KaplanMeierFitter()
kmf.fit(high['time'], high['event'], label='PARK7 mRNA (Global) High')
kmf.plot_survival_function(color='#C0392B', ci_show=False, ax=ax)
kmf.fit(low['time'], low['event'], label='PARK7 mRNA (Global) Low')
kmf.plot_survival_function(color='#2980B9', ci_show=False, ax=ax)
ax.set_title(
    f"Overall Survival by PARK7 mRNA Expression\nLog-rank test p-value: {res.p_value:.4f} ({get_sig_label(res.p_value)})")
ax.set_ylabel("Survival Probability")
plt.tight_layout()
out_surv = os.path.join(PUBLICATION_OUT, "Fig_5B_Survival_PARK7.png")
plt.savefig(out_surv, dpi=300)
plt.close()
print(f"Saved: {out_surv}")

# Figure 5C-H: TCA Cycle Enzymes
plot_tcga_box('CS', 'Citrate Synthase', "Fig_5C_CS_IDH.png")
plot_tcga_box('ACO2', 'Aconitase 2', "Fig_5D_ACO2_IDH.png")
plot_tcga_box('IDH1', 'Isocitrate Dehydrogenase 1', "Fig_5E_IDH1_IDH.png")
plot_tcga_box('IDH2', 'Isocitrate Dehydrogenase 2', "Fig_5F_IDH2_IDH.png")
plot_tcga_box('OGDH', 'Oxoglutarate Dehydrogenase (aKGDH)', "Fig_5G_OGDH_IDH.png")
plot_tcga_box('SDHB', 'Succinate Dehydrogenase Subunit B', "Fig_5H_SDHB_IDH.png")

print("\nFigure 5 generation completed successfully!")

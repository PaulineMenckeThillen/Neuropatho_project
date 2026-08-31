"""
================================================================================
FIGURE 4: AUTOMATED DIGITAL IMAGE ANALYSIS & TOTAL SUM QUANTIFICATION
Manuscript: "DJ-1 Overexpression Defines a TCA/SDHB-Driven Metabolic Signature 
             in IDH-Wildtype Glioblastoma"
Author: Pauline Mencke et al.
Generates:
  - Fig 4A: Normalized Fold-Change of DJ-1 Total Sum (Control Baseline = 1.0)
  - Fig 4B: Normalized Fold-Change of GFAP Total Sum (Control Baseline = 1.0)
  - Fig 4C: DJ-1 Total Sum in IDH-Wildtype vs. IDH-Mutant Glioma
  - Fig 4D: GFAP Total Sum in IDH-Wildtype vs. IDH-Mutant Glioma
  - Fig 4E: DJ-1 Total Sum stratified by ATRX expression status
  - Fig 4F: DJ-1 Total Sum stratified by MGMT promoter methylation status
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
FILE_AEC_CSV = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\ImageAnalysis\AEC_Analysis_Results.csv'
FILE_DAB_CSV = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\ImageAnalysis\GFAP_DAB_Analysis_Results.csv'
FILE_CLINICAL = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\ImageAnalysis\CaseIDs.xlsx'
os.makedirs(PUBLICATION_OUT, exist_ok=True)

sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['pdf.fonttype'] = 42

COLOR_CONTROL = '#D3D3D3'
COLOR_GLIOMA = '#2C5D63'
COLOR_WT = '#1A3636'
COLOR_MUT = '#7A9D96'
COLOR_POS = '#4F709C'
COLOR_NEG = '#D6AD60'

def get_sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "(n.s.)"

# -----------------------------------------------------------------------------
# Load & Merge Image Analysis CSVs
# -----------------------------------------------------------------------------
if not (os.path.exists(FILE_AEC_CSV) and os.path.exists(FILE_DAB_CSV) and os.path.exists(FILE_CLINICAL)):
    print("Error: Required image analysis CSV files not found.")
    exit(1)

df_dj1 = pd.read_csv(FILE_AEC_CSV).groupby('CaseID').agg(
    {'Nuclear_AEC': 'mean', 'Cytoplasmic_AEC': 'mean'}).reset_index()
df_gfap = pd.read_csv(FILE_DAB_CSV).groupby('CaseID').agg(
    {'Nuclear_DAB': 'mean', 'Cytoplasmic_DAB': 'mean'}).reset_index()
df_clin = pd.read_excel(FILE_CLINICAL)[
    ['CaseID', 'PatientID', 'IDH 1 Mut.', 'ATRX', 'MGMT Promotor Methylierung', 'comment']].dropna(subset=['PatientID'])

df_clin['PatientID'] = df_clin['PatientID'].apply(lambda x: str(x).split('_')[0].strip())
df_clin = df_clin[~df_clin['comment'].astype(str).str.lower().str.contains('exclude')].drop_duplicates('PatientID')

df_dj1 = pd.merge(df_dj1, df_clin[['CaseID', 'PatientID']], on='CaseID')
df_gfap = pd.merge(df_gfap, df_clin[['CaseID', 'PatientID']], on='CaseID')

df_biomarkers = pd.merge(df_dj1.groupby('PatientID').mean().reset_index(),
                         df_gfap.groupby('PatientID').mean().reset_index(), on='PatientID')
df_auto = pd.merge(df_biomarkers, df_clin, on='PatientID')

df_auto = df_auto[(df_auto['PatientID'] != 'N1070') & (~df_auto['PatientID'].str.contains('1401|1197|1461'))]

df_auto['Total_Sum_DJ1'] = df_auto['Nuclear_AEC'] + df_auto['Cytoplasmic_AEC']
df_auto['Total_Sum_GFAP'] = df_auto['Nuclear_DAB'] + df_auto['Cytoplasmic_DAB']

df_auto['ATRX_Numeric'] = df_auto['ATRX'].apply(
    lambda x: 1.0 if 'retained' in str(x).lower() else (0.0 if 'loss' in str(x).lower() else np.nan))
df_auto['MGMT_Numeric'] = df_auto['MGMT Promotor Methylierung'].apply(
    lambda x: 1.0 if 'ja' in str(x).lower() else (0.0 if 'nein' in str(x).lower() else np.nan))

df_auto['Is_Autopsy'] = df_auto['PatientID'].apply(lambda x: x.startswith('A'))
df_auto['Group_GvC'] = np.where(df_auto['Is_Autopsy'], 'Autopsy Control', 'Glioma')
df_auto['Group_3G'] = np.where(df_auto['Is_Autopsy'], 'Autopsy Control',
                               np.where(df_auto['IDH 1 Mut.'].str.lower().isin(['ja', 'mutiert']), 'IDH Mutant',
                                        'IDH Wildtype'))

df_glioma = df_auto[df_auto['Group_GvC'] == 'Glioma']

# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------
def plot_normalized_ttest(col, title, fname):
    glioma = df_glioma[col].dropna()
    autopsy = df_auto[df_auto['Group_GvC'] == 'Autopsy Control'][col].dropna()
    if glioma.empty or autopsy.empty: return

    aut_mean = autopsy.mean()
    t_stat, p_val = stats.ttest_1samp(glioma / aut_mean, 1.0)

    df_plot = pd.DataFrame({'Norm': pd.concat([autopsy / aut_mean, glioma / aut_mean]),
                            'Group': ['Autopsy Control'] * len(autopsy) + ['Glioma'] * len(glioma)})

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.boxplot(data=df_plot, x='Group', y='Norm', palette={'Autopsy Control': COLOR_CONTROL, 'Glioma': COLOR_GLIOMA},
                showfliers=False, width=0.4, ax=ax)
    sns.swarmplot(data=df_plot, x='Group', y='Norm', color='#333333', size=7, ax=ax)
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_title(
        f"{title}: Normalized to Control Baseline\nOne-Sample t-test p-value: {p_val:.4f} {get_sig_label(p_val)}")
    ax.set_ylabel("Fold-Change (Control Mean Baseline = 1.0)")
    plt.tight_layout()
    out_path = os.path.join(PUBLICATION_OUT, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_pure_idh(col, title, fname):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    wt = df_glioma[df_glioma['Group_3G'] == 'IDH Wildtype'][col].dropna()
    mut = df_glioma[df_glioma['Group_3G'] == 'IDH Mutant'][col].dropna()
    _, p_mw = stats.mannwhitneyu(wt, mut, alternative='two-sided')

    sns.boxplot(data=df_glioma, x='Group_3G', y=col, order=['IDH Wildtype', 'IDH Mutant'],
                palette={'IDH Wildtype': COLOR_WT, 'IDH Mutant': COLOR_MUT}, showfliers=False, width=0.4, ax=ax)
    sns.swarmplot(data=df_glioma, x='Group_3G', y=col, order=['IDH Wildtype', 'IDH Mutant'], color='#333333', size=7,
                  ax=ax)
    ax.axhline(wt.mean(), color='gray', linestyle='--')
    ax.set_title(f"{title}: IDH Wildtype vs. Mutant\nMann-Whitney U p-value: {p_mw:.4f} {get_sig_label(p_mw)}")
    ax.set_ylabel(title)
    plt.tight_layout()
    out_path = os.path.join(PUBLICATION_OUT, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_clinical_cat(clin_col, name_map, title_label, fname):
    sub = df_glioma[['Total_Sum_DJ1', clin_col]].dropna()
    sub['Label'] = sub[clin_col].map(name_map)
    g1, g0 = sub[sub[clin_col] == 1.0]['Total_Sum_DJ1'], sub[sub[clin_col] == 0.0]['Total_Sum_DJ1']
    _, p_mw = stats.mannwhitneyu(g1, g0, alternative='two-sided')

    fig, ax = plt.subplots(figsize=(6, 5.5))
    sns.boxplot(data=sub, x='Label', y='Total_Sum_DJ1', order=[name_map[1.0], name_map[0.0]],
                palette={name_map[1.0]: COLOR_POS, name_map[0.0]: COLOR_NEG}, showfliers=False, width=0.4, ax=ax)
    sns.swarmplot(data=sub, x='Label', y='Total_Sum_DJ1', order=[name_map[1.0], name_map[0.0]], color='#333333', size=7,
                  ax=ax)
    ax.set_title(f"DJ-1 Total Sum by {title_label} Status\nMann-Whitney U p-value: {p_mw:.4f} {get_sig_label(p_mw)}")
    ax.set_ylabel("DJ-1 Total Sum")
    plt.tight_layout()
    out_path = os.path.join(PUBLICATION_OUT, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# Generate Figure 4 Panels
plot_normalized_ttest('Total_Sum_DJ1', "DJ-1 Total Sum", "Fig_4A_DJ1_Norm_Total.png")
plot_normalized_ttest('Total_Sum_GFAP', "GFAP Total Sum", "Fig_4B_GFAP_Norm_Total.png")
plot_pure_idh('Total_Sum_DJ1', "DJ-1 Total Sum", "Fig_4C_DJ1_IDH.png")
plot_pure_idh('Total_Sum_GFAP', "GFAP Total Sum", "Fig_4D_GFAP_IDH.png")
plot_clinical_cat('ATRX_Numeric', {1.0: 'Retained (Yes)', 0.0: 'Loss (No)'}, "ATRX", "Fig_4E_DJ1_ATRX.png")
plot_clinical_cat('MGMT_Numeric', {1.0: 'Methylated (Yes)', 0.0: 'Unmethylated (No)'}, "MGMT", "Fig_4F_DJ1_MGMT.png")

print("\nFigure 4 generation completed successfully!")

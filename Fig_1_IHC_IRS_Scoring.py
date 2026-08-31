"""
================================================================================
FIGURE 1: MANUAL IHC IMMUNOREACTIVE SCORING (IRS)
Manuscript: "DJ-1 Overexpression Defines a TCA/SDHB-Driven Metabolic Signature 
             in IDH-Wildtype Glioblastoma"
Author: Pauline Mencke et al.
Generates:
  - Fig 1A: DJ-1 IRS in Glioma vs. Autopsy Controls
  - Fig 1B: GFAP IRS in Glioma vs. Autopsy Controls
  - Fig 1C: DJ-1 IRS stratified by IDH status (Autopsy vs. IDH-WT vs. IDH-Mut)
  - Fig 1D: GFAP IRS stratified by IDH status (Autopsy vs. IDH-WT vs. IDH-Mut)
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations

# -----------------------------------------------------------------------------
# Paths and Aesthetics Configuration
# -----------------------------------------------------------------------------
PUBLICATION_OUT = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\Publication_Figures'
FILE_IHC_EXCEL = r'C:\Users\pmenc\Documents\Pauline\Neuropathologie\IHC Analysis\AuswertungAECGFAPscans.xlsx'
os.makedirs(PUBLICATION_OUT, exist_ok=True)

sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['pdf.fonttype'] = 42

COLOR_CONTROL = '#D3D3D3'
COLOR_GLIOMA = '#2C5D63'
COLOR_WT = '#1A3636'
COLOR_MUT = '#7A9D96'

def get_sig_label(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "(n.s.)"

def clean_ihc_val(val):
    if pd.isna(val): return np.nan
    s = str(val).lower().strip().replace('%', '').replace(',', '.').replace('<', '').replace('>', '')
    if s in ['?', '-', 'negativ', 'unbekannt', 'none', 'na'] or '"' in s or 'positiv' in s: return np.nan
    try:
        return float(s)
    except:
        return np.nan

# -----------------------------------------------------------------------------
# Load & Process IHC Data
# -----------------------------------------------------------------------------
if not os.path.exists(FILE_IHC_EXCEL):
    print(f"Error: {FILE_IHC_EXCEL} not found.")
    exit(1)

xl = pd.ExcelFile(FILE_IHC_EXCEL)
df_f = xl.parse(xl.sheet_names[1])
df_t = xl.parse(xl.sheet_names[2])
df_ihc = pd.concat([df_f, df_t], ignore_index=True)

id_col = next(c for c in df_ihc.columns if any(x in str(c).upper() for x in ['CASE ID', 'FALL', 'ID', 'NR']))
df_ihc = df_ihc.rename(columns={id_col: 'Case_ID'})
df_ihc = df_ihc[~df_ihc['Case_ID'].astype(str).str.contains('1461')]
df_ihc['IDH_Status'] = df_ihc['IDH 1 Mut.'].map({'ja': 'IDH Mutant', 'nein': 'IDH Wildtype', 'mutiert': 'IDH Mutant'})

df_ihc['DJ1_IRS'] = df_ihc['IRS (DJ-1)'].apply(clean_ihc_val)
df_ihc['GFAP_IRS'] = df_ihc['IRS (GFAP)'].apply(clean_ihc_val)

TARGET_AUTOPSIES = ['A10/24', 'A21/24', 'A28/23', 'A28/24']
df_ihc['Is_Autopsy'] = df_ihc['Case_ID'].apply(
    lambda x: any(a.replace(" ", "") in str(x).upper().replace(" ", "") for a in TARGET_AUTOPSIES)) | df_ihc[
                           'Diagnosis'].astype(str).str.lower().str.contains('autopsy')
df_ihc['Triple_Group'] = np.where(df_ihc['Is_Autopsy'], 'Autopsy Control', df_ihc['IDH_Status'])
df_ihc['Group_GvC'] = np.where(df_ihc['Is_Autopsy'], 'Autopsy Control', 'Glioma')

df_aut_merged = df_ihc[df_ihc['Group_GvC'] == 'Autopsy Control'].groupby('Case_ID', as_index=False).agg(
    {'DJ1_IRS': 'mean', 'GFAP_IRS': 'mean', 'Group_GvC': 'first', 'Triple_Group': 'first'})
df_ihc_plot = pd.concat([df_ihc[df_ihc['Group_GvC'] == 'Glioma'], df_aut_merged], ignore_index=True)

# -----------------------------------------------------------------------------
# Plotting Function
# -----------------------------------------------------------------------------
def plot_ihc_boxplot(data, x, y, order, palette, title, ylabel, fname):
    plot_data = data.dropna(subset=[x, y]).copy()
    groups = [plot_data[plot_data[x] == g][y] for g in order if g in plot_data[x].unique()]

    stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided') if len(
        order) == 2 else stats.kruskal(*groups)
    test_name = "Mann-Whitney U" if len(order) == 2 else "Kruskal-Wallis"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.boxplot(data=plot_data, x=x, y=y, palette=palette, order=order, showfliers=False, width=0.45, ax=ax,
                boxprops=dict(edgecolor='#4a4a4a'))
    sns.swarmplot(data=plot_data, x=x, y=y, order=order, color='#333333', size=7, alpha=0.8, ax=ax, edgecolor='white',
                  linewidth=0.6)

    baseline = plot_data[plot_data[x] == order[0]][y].mean()
    ax.axhline(baseline, color='gray', linestyle='--', label=f'{order[0]} Mean ({baseline:.3f})')
    ax.legend(loc='upper left', frameon=True)

    if len(order) > 2:
        y_max, step = plot_data[y].max(), (plot_data[y].max() - plot_data[y].min()) * 0.08
        current_y = y_max + step
        for i, j in combinations(range(len(order)), 2):
            g1, g2 = groups[i], groups[j]
            if len(g1) > 1 and len(g2) > 1:
                _, p_pair = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                if p_pair < 0.05:
                    h = (plot_data[y].max() - plot_data[y].min()) * 0.02
                    ax.plot([i, i, j, j], [current_y, current_y + h, current_y + h, current_y], lw=1.2, color='#4a4a4a')
                    ax.text((i + j) * .5, current_y + h, get_sig_label(p_pair), ha='center', va='bottom', fontsize=13,
                            fontweight='bold')
                    current_y += (step * 1.3)
        ax.set_ylim(top=current_y + step * 0.5)

    ax.set_title(f"{title}\nOverall {test_name} p-value: {p_val:.4f} {get_sig_label(p_val)}", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Group")
    plt.tight_layout()
    out_path = os.path.join(PUBLICATION_OUT, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

# Generate Figure 1 Panels
plot_ihc_boxplot(df_ihc_plot, 'Group_GvC', 'DJ1_IRS', ['Autopsy Control', 'Glioma'],
                 {'Autopsy Control': COLOR_CONTROL, 'Glioma': COLOR_GLIOMA},
                 "IRS (DJ-1): Glioma vs. Autopsy Control", "IRS (DJ-1)", "Fig_1A_DJ1_IRS_GvC.png")

plot_ihc_boxplot(df_ihc_plot, 'Group_GvC', 'GFAP_IRS', ['Autopsy Control', 'Glioma'],
                 {'Autopsy Control': COLOR_CONTROL, 'Glioma': COLOR_GLIOMA},
                 "IRS (GFAP): Glioma vs. Autopsy Control", "IRS (GFAP)", "Fig_1B_GFAP_IRS_GvC.png")

plot_ihc_boxplot(df_ihc_plot, 'Triple_Group', 'DJ1_IRS', ['Autopsy Control', 'IDH Wildtype', 'IDH Mutant'],
                 {'Autopsy Control': COLOR_CONTROL, 'IDH Wildtype': COLOR_WT, 'IDH Mutant': COLOR_MUT},
                 "IRS (DJ-1): Autopsy vs. IDH Status", "IRS (DJ-1)", "Fig_1C_DJ1_IRS_3Way.png")

plot_ihc_boxplot(df_ihc_plot, 'Triple_Group', 'GFAP_IRS', ['Autopsy Control', 'IDH Wildtype', 'IDH Mutant'],
                 {'Autopsy Control': COLOR_CONTROL, 'IDH Wildtype': COLOR_WT, 'IDH Mutant': COLOR_MUT},
                 "IRS (GFAP): Autopsy vs. IDH Status", "IRS (GFAP)", "Fig_1D_GFAP_IRS_3Way.png")

print("\nFigure 1 generation completed successfully!")

"""
scatter_routes.py
=================
Pre/Post log change scatter plot by HS6 trade route classification.

For each HS6 manufacturing sector, computes:
  - d_log_mex_us  = log(1 + avg MEX->US post) - log(1 + avg MEX->US pre)
  - d_log_chn_us  = log(1 + avg CHN->US post) - log(1 + avg CHN->US pre)

Then classifies each sector by which bilateral flows were active
in the pre-tariff period (2016-2017), producing five route types.

Pre-tariff period:  Jan 2016 – Dec 2017
Post-tariff period: Jul 2018 – Nov 2024  (Section 301 List 1 onward)
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Generate scatter route figure.')
parser.add_argument('--data',   default='data',    help='path to input data folder')
parser.add_argument('--output', default='outputs', help='path to output folder')
args = parser.parse_args()

DATA_DIR = args.data
OUT_DIR  = args.output

PRE_START  = '2016-01-01'
PRE_END    = '2017-12-31'
POST_START = '2018-07-01'
POST_END   = '2024-11-01'

# USD/month threshold to count a flow as "active" in the pre-period
ACTIVE_THRESHOLD = 1_000   # $1,000/month

MANUF_HS2 = {
    '28','29','30','32','33','34','38','39','40',
    '54','55','56','57','58','59','60','61','62','63','64',
    '72','73','74','75','76','78','79','80',
    '84','85','86','87','88','89','90','91','92','93','94','95','96'
}

# ── STEP 1: LOAD AND AGGREGATE DATA ───────────────────────────────────────────
#
# We use the FULL unfiltered dataset here (not restricted to common_codes),
# because the scatter includes all HS6 codes present in any flow — including
# sectors where only China or only Mexico trades with the US.
# This gives a richer picture of the full trade landscape.

def load_and_aggregate(paths, value_col):
    """
    Load Comtrade CSVs, filter to HS6 manufacturing rows,
    and return a monthly HS6 x date panel.
    """
    dfs = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  WARNING: not found — {path}")
            continue
        df = pd.read_csv(path, encoding='latin1', low_memory=False)
        out = pd.DataFrame({
            'hs_code':    df['isOriginalClassification'].astype(str).str.strip().str.zfill(6),
            'hs_level':   df['cmdDesc'].astype(int),
            'fobvalue':   df['cifvalue'],
            'year_month': df['refMonth'].astype(str).str.zfill(6),
        })
        out['hs2']  = out['hs_code'].str[:2].str.zfill(2)
        out['date'] = pd.to_datetime(out['year_month'], format='%Y%m')
        # Keep only HS6 manufacturing rows
        out = out[(out['hs_level'] == 6) & (out['hs2'].isin(MANUF_HS2))]
        dfs.append(out[['hs_code', 'date', 'fobvalue']])

    combined = pd.concat(dfs, ignore_index=True)

    if not dfs:
        raise FileNotFoundError(
            f"\nNo files loaded for '{value_col}'.\n"
            f"Check that DATA_DIR is correct and files exist.\n"
            f"Run with:  python scatter_routes.py "
            f"--data /path/to/data --output /path/to/outputs"
        )

    # Sum to HS6 x month (in case of duplicate rows)
    return (combined
            .groupby(['hs_code', 'date'])['fobvalue']
            .sum()
            .reset_index()
            .rename(columns={'fobvalue': value_col}))


def period_average(df, value_col, start, end, suffix):
    """Compute average monthly value within a date window."""
    mask = (df['date'] >= start) & (df['date'] <= end)
    return (df[mask]
            .groupby('hs_code')[value_col]
            .mean()
            .rename(f"{value_col}_{suffix}"))


print("Loading data...")

mex_paths = [f"{DATA_DIR}/mex_usa_exports_{y}.csv" for y in range(2016, 2025)]
cm_paths  = [f"{DATA_DIR}/chn_mex_exports_{y}.csv" for y in range(2016, 2025)]
cu_paths  = [f"{DATA_DIR}/chn_usa_exports_{y}.csv" for y in range(2016, 2025)]

mex = load_and_aggregate(mex_paths, 'mex_us')
cm  = load_and_aggregate(cm_paths,  'chn_mex')
cu  = load_and_aggregate(cu_paths,  'chn_us')

print(f"  MEX->US:  {mex['hs_code'].nunique()} unique HS6 codes")
print(f"  CHN->MEX: {cm['hs_code'].nunique()} unique HS6 codes")
print(f"  CHN->US:  {cu['hs_code'].nunique()} unique HS6 codes")


# ── STEP 2: COMPUTE PRE/POST AVERAGES ─────────────────────────────────────────

mex_pre  = period_average(mex, 'mex_us',  PRE_START,  PRE_END,   'pre')
mex_post = period_average(mex, 'mex_us',  POST_START, POST_END,  'post')
cu_pre   = period_average(cu,  'chn_us',  PRE_START,  PRE_END,   'pre')
cu_post  = period_average(cu,  'chn_us',  POST_START, POST_END,  'post')
cm_pre   = period_average(cm,  'chn_mex', PRE_START,  PRE_END,   'pre')

# Build sector-level dataframe — union of all HS6 codes across all flows
all_codes = sorted(set(mex['hs_code']) | set(cm['hs_code']) | set(cu['hs_code']))
scatter = pd.DataFrame({'hs_code': all_codes})

for series in [mex_pre, mex_post, cu_pre, cu_post, cm_pre]:
    scatter = scatter.join(series, on='hs_code', how='left')

scatter = scatter.fillna(0)
scatter.columns = ['hs_code', 'mex_us_pre', 'mex_us_post',
                   'chn_us_pre', 'chn_us_post', 'chn_mex_pre']

print(f"\nTotal HS6 sectors (any flow): {len(scatter):,}")


# ── STEP 3: LOG DIFFERENCES ────────────────────────────────────────────────────
#
# We use log(1 + x) throughout so that sectors with zero trade in one period
# are included rather than dropped. This is consistent with the DiD model.
#
# d_log_mex_us > 0  means Mexico exported more to US post-tariff
# d_log_chn_us < 0  means China exported less to US post-tariff
#
# The upper-left quadrant (d_log_chn_us < 0, d_log_mex_us > 0) is the
# visual signature of substitution — China fell, Mexico rose.

scatter['d_log_mex_us'] = (np.log1p(scatter['mex_us_post'])
                           - np.log1p(scatter['mex_us_pre']))
scatter['d_log_chn_us'] = (np.log1p(scatter['chn_us_post'])
                           - np.log1p(scatter['chn_us_pre']))


# ── STEP 4: ROUTE CLASSIFICATION ──────────────────────────────────────────────
#
# Each sector is classified based on which flows were ACTIVE in the
# pre-tariff period (2016-2017). "Active" means average monthly value
# exceeds ACTIVE_THRESHOLD.
#
# ruta_hibrida:     all three flows active — Mexico and China both competing
#                   in US market, with Chinese goods also entering Mexico
# ruta_mixta:       MEX->US and CHN->US active, but CHN->MEX below threshold
# ruta_directa_mx:  only MEX->US active — Mexico has a US route, China doesn't
# ruta_directa_chn: only CHN->US active — China has a US route, Mexico doesn't
# sin_comercio:     neither MEX->US nor CHN->US active in pre-period

scatter['has_mex'] = scatter['mex_us_pre']  > ACTIVE_THRESHOLD
scatter['has_cu']  = scatter['chn_us_pre']  > ACTIVE_THRESHOLD
scatter['has_cm']  = scatter['chn_mex_pre'] > ACTIVE_THRESHOLD


def classify_route(row):
    m, c, k = row['has_mex'], row['has_cu'], row['has_cm']
    if   m and c and k:  return 'ruta_hibrida'
    elif m and c:        return 'ruta_mixta'
    elif m and not c:    return 'ruta_directa_mx'
    elif c and not m:    return 'ruta_directa_chn'
    else:                return 'sin_comercio'


scatter['ruta'] = scatter.apply(classify_route, axis=1)

print("\nRoute classification (pre-tariff trade presence):")
print(scatter['ruta'].value_counts().to_string())


# ── STEP 5: REGRESSION SLOPES PER ROUTE ───────────────────────────────────────
#
# For each route, we fit a simple OLS: d_log_mex_us ~ d_log_chn_us
# A negative slope = when China's US exports fell, Mexico's rose (substitution)
# A positive slope = both moved together (common macro shock, not substitution)
# A zero slope     = independent movements (neither substitution nor correlation)

print("\nOLS slope (d_log_chn_us -> d_log_mex_us) by route:")
ORDER = ['ruta_hibrida','ruta_mixta','ruta_directa_mx','ruta_directa_chn','sin_comercio']
slopes = {}
for ruta in ORDER:
    sub = scatter[scatter['ruta'] == ruta].dropna(
        subset=['d_log_chn_us', 'd_log_mex_us'])
    if len(sub) > 10:
        m, b = np.polyfit(sub['d_log_chn_us'], sub['d_log_mex_us'], 1)
        slopes[ruta] = (m, b)
        corr = sub['d_log_chn_us'].corr(sub['d_log_mex_us'])
        print(f"  {ruta:<22}  slope={m:>6.3f}  r={corr:>5.3f}  n={len(sub)}")


# ── STEP 6: PLOT ───────────────────────────────────────────────────────────────

COLORS = {
    'ruta_hibrida':     '#C0392B',   # red
    'ruta_mixta':       '#E67E22',   # orange
    'ruta_directa_mx':  '#2980B9',   # blue
    'ruta_directa_chn': '#27AE60',   # green
    'sin_comercio':     '#7F8C8D',   # grey
}

LABEL_MAP = {
    'ruta_hibrida':     'Hybrid route',
    'ruta_mixta':       'Mixed route',
    'ruta_directa_mx':  'Mexico direct',
    'ruta_directa_chn': 'China direct',
    'sin_comercio':     'No trade',
}

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.linewidth':    0.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.facecolor': 'white',
    'savefig.bbox':      'tight',
})

X_LIM = (-8, 4)
Y_LIM = (-7, 8)

TITLE_MAP = {
    'ruta_hibrida':     'Hybrid route',
    'ruta_mixta':       'Mixed route',
    'ruta_directa_mx':  'Mexico direct',
    'ruta_directa_chn': 'China direct',
    'sin_comercio':     'No trade',
}

subsets = [scatter] + [scatter[scatter['ruta'] == r] for r in ORDER]
titles  = [f"All sectors (n={len(scatter):,})"] + \
          [f"{TITLE_MAP[r]} (n={len(scatter[scatter['ruta']==r]):,})" for r in ORDER]

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.flatten()

for i, (sub, title) in enumerate(zip(subsets, titles)):
    ax = axes[i]

    if i == 0:
        # All-sector panel: color by route
        for ruta in ORDER:
            s = sub[sub['ruta'] == ruta]
            ax.scatter(s['d_log_chn_us'], s['d_log_mex_us'],
                       c=COLORS[ruta], s=10, alpha=0.45, linewidths=0,
                       label=LABEL_MAP[ruta], zorder=3)
        ax.legend(fontsize=6.5, frameon=False, loc='upper right',
                  markerscale=1.8, handletextpad=0.4)
    else:
        ruta = ORDER[i - 1]
        ax.scatter(sub['d_log_chn_us'], sub['d_log_mex_us'],
                   c=COLORS[ruta], s=14, alpha=0.55, linewidths=0, zorder=3)

        # OLS regression line
        if ruta in slopes:
            m, b = slopes[ruta]
            xr = np.linspace(*X_LIM, 200)
            ax.plot(xr, m * xr + b, color='black', linewidth=1.1,
                    linestyle='--', alpha=0.55, zorder=4)
            ax.text(0.05, 0.95, f'slope = {m:.3f}',
                    transform=ax.transAxes, fontsize=7.5,
                    va='top', color='#333')

    # Reference lines at zero
    ax.axhline(0, color='#bbb', linewidth=0.6, linestyle='--', zorder=1)
    ax.axvline(0, color='#bbb', linewidth=0.6, linestyle='--', zorder=1)

    # Upper-left quadrant shading (substitution zone)
    ax.axhspan(0, Y_LIM[1], xmin=0, xmax=(0 - X_LIM[0]) / (X_LIM[1] - X_LIM[0]),
               color='#e8f5e9', alpha=0.3, zorder=0)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel(r'$\Delta\log(\mathrm{CHN}{\to}\mathrm{US})$', fontsize=8.5)
    ax.set_ylabel(r'$\Delta\log(\mathrm{MEX}{\to}\mathrm{US})$', fontsize=8.5)
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.tick_params(labelsize=7.5)

fig.suptitle(
    r'Pre$\to$Post change in log trade flows by HS6 sector, classified by trade route',
    fontsize=10, y=1.01)
plt.tight_layout(h_pad=2.5, w_pad=2)

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(f"{OUT_DIR}/figure_1.pdf", format='pdf')
fig.savefig(f"{OUT_DIR}/figure_1.png", format='png')
plt.close()
print("\nFigure saved.")
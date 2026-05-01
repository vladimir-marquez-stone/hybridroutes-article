"""
generate_figures.py
===================
Generates all four figures for:
  "Competition and Dependence: Mexico, China, and the New Export Routes
   to the United States"

Figure 1 — Scatter plot: pre/post log change by HS6 trade route
Figure 2 — Indexed aggregate trade flows (2016–2024)
Figure 3 — Event study: MEX→US (nearshoring test)
Figure 4 — Event study: CHN→MEX (deflection test)

Usage:
  1. Set DATA_DIR and OUT_DIR below
  2. Run:  python generate_figures.py

Data file naming convention:
  mex_usa_exports_{2016..2024}.csv
  chn_mex_exports_{2016..2024}.csv
  chn_usa_exports_{2016..2024}.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

import argparse as _ap
_parser = _ap.ArgumentParser(description="Generate paper figures.")
_parser.add_argument("--data",   default="data",    help="input data folder")
_parser.add_argument("--output", default="figures", help="output folder for figures")
_args = _parser.parse_args()

DATA_DIR = _args.data
OUT_DIR  = _args.output

PRE_START  = '2016-01-01'
PRE_END    = '2017-12-31'
POST_START = '2018-07-01'
POST_END   = '2024-11-01'

ACTIVE_THRESHOLD = 1_000   # USD/month to count a flow as "active" (Figure 1)

MANUF_HS2 = {
    '28','29','30','32','33','34','38','39','40',
    '54','55','56','57','58','59','60','61','62','63','64',
    '72','73','74','75','76','78','79','80',
    '84','85','86','87','88','89','90','91','92','93','94','95','96'
}

# ── SHARED STYLE ───────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         10,
    'axes.linewidth':    0.6,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.facecolor': 'white',
})


# ── DATA LOADER ────────────────────────────────────────────────────────────────

def load_and_aggregate(paths, value_col):
    """Load Comtrade CSVs, filter to HS6 manufacturing, aggregate to HS6 x month."""
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
        out = out[(out['hs_level'] == 6) & (out['hs2'].isin(MANUF_HS2))]
        dfs.append(out[['hs_code', 'date', 'fobvalue']])
    combined = pd.concat(dfs, ignore_index=True)
    return (combined
            .groupby(['hs_code', 'date'])['fobvalue']
            .sum().reset_index()
            .rename(columns={'fobvalue': value_col}))


def period_average(df, value_col, start, end, suffix):
    """Average monthly value within a date window."""
    mask = (df['date'] >= start) & (df['date'] <= end)
    return df[mask].groupby('hs_code')[value_col].mean().rename(f"{value_col}_{suffix}")


def aggregate_monthly(paths, value_col):
    """Aggregate to total monthly value across all HS6 codes."""
    dfs = []
    for path in paths:
        if not os.path.exists(path):
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
        out = out[(out['hs_level'] == 6) & (out['hs2'].isin(MANUF_HS2))]
        dfs.append(out[['date', 'fobvalue']])
    combined = pd.concat(dfs, ignore_index=True)
    return combined.groupby('date')['fobvalue'].sum().reset_index().rename(columns={'fobvalue': value_col})


# ── LOAD DATA ──────────────────────────────────────────────────────────────────

print("Loading data...")

mex_paths = [f"{DATA_DIR}/mex_usa_exports_{y}.csv" for y in range(2016, 2025)]
cm_paths  = [f"{DATA_DIR}/chn_mex_exports_{y}.csv" for y in range(2016, 2025)]
cu_paths  = [f"{DATA_DIR}/chn_usa_exports_{y}.csv" for y in range(2016, 2025)]

# HS6-level data for Figure 1 (scatter)
mex_hs6 = load_and_aggregate(mex_paths, 'mex_us')
cm_hs6  = load_and_aggregate(cm_paths,  'chn_mex')
cu_hs6  = load_and_aggregate(cu_paths,  'chn_us')

# Aggregate monthly totals for Figure 2 (indexed trends)
mex_tot = aggregate_monthly(mex_paths, 'mex')
cu_tot  = aggregate_monthly(cu_paths,  'cu')
cm_tot  = aggregate_monthly(cm_paths,  'cm')

print("Data loaded.")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Scatter: pre/post log change by trade route
# ══════════════════════════════════════════════════════════════════════════════

print("\nGenerating Figure 1...")

# Pre/post averages per HS6
mex_pre  = period_average(mex_hs6, 'mex_us',  PRE_START,  PRE_END,  'pre')
mex_post = period_average(mex_hs6, 'mex_us',  POST_START, POST_END, 'post')
cu_pre   = period_average(cu_hs6,  'chn_us',  PRE_START,  PRE_END,  'pre')
cu_post  = period_average(cu_hs6,  'chn_us',  POST_START, POST_END, 'post')
cm_pre   = period_average(cm_hs6,  'chn_mex', PRE_START,  PRE_END,  'pre')

all_codes = sorted(set(mex_hs6['hs_code']) | set(cm_hs6['hs_code']) | set(cu_hs6['hs_code']))
sc = pd.DataFrame({'hs_code': all_codes})
for s in [mex_pre, mex_post, cu_pre, cu_post, cm_pre]:
    sc = sc.join(s, on='hs_code', how='left')
sc = sc.fillna(0)
sc.columns = ['hs_code', 'mex_us_pre', 'mex_us_post',
              'chn_us_pre', 'chn_us_post', 'chn_mex_pre']

sc['d_log_mex_us'] = np.log1p(sc['mex_us_post']) - np.log1p(sc['mex_us_pre'])
sc['d_log_chn_us'] = np.log1p(sc['chn_us_post']) - np.log1p(sc['chn_us_pre'])

sc['has_mex'] = sc['mex_us_pre']  > ACTIVE_THRESHOLD
sc['has_cu']  = sc['chn_us_pre']  > ACTIVE_THRESHOLD
sc['has_cm']  = sc['chn_mex_pre'] > ACTIVE_THRESHOLD

def classify_route(row):
    m, c, k = row['has_mex'], row['has_cu'], row['has_cm']
    if   m and c and k: return 'Hybrid route'
    elif m and c:       return 'Mixed route'
    elif m and not c:   return 'Mexico direct'
    elif c and not m:   return 'China direct'
    else:               return 'No trade'

sc['route'] = sc.apply(classify_route, axis=1)

ORDER  = ['Hybrid route', 'Mixed route', 'Mexico direct', 'China direct', 'No trade']
COLORS = {
    'Hybrid route':  '#C0392B',
    'Mixed route':   '#E67E22',
    'Mexico direct': '#2980B9',
    'China direct':  '#27AE60',
    'No trade':      '#7F8C8D',
}

# OLS slopes
slopes = {}
for route in ORDER:
    sub = sc[sc['route'] == route].dropna(subset=['d_log_chn_us', 'd_log_mex_us'])
    if len(sub) > 10:
        m, b = np.polyfit(sub['d_log_chn_us'], sub['d_log_mex_us'], 1)
        slopes[route] = (m, b)

X_LIM, Y_LIM = (-8, 4), (-7, 8)
subsets = [sc] + [sc[sc['route'] == r] for r in ORDER]
titles  = [f"All sectors (n={len(sc):,})"] + \
          [f"{r} (n={len(sc[sc['route']==r]):,})" for r in ORDER]

fig1, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.flatten()

for i, (sub, title) in enumerate(zip(subsets, titles)):
    ax = axes[i]
    if i == 0:
        for route in ORDER:
            s = sub[sub['route'] == route]
            ax.scatter(s['d_log_chn_us'], s['d_log_mex_us'],
                       c=COLORS[route], s=10, alpha=0.45, linewidths=0,
                       label=route, zorder=3)
        ax.legend(fontsize=6.5, frameon=False, loc='upper right',
                  markerscale=1.8, handletextpad=0.4)
    else:
        route = ORDER[i - 1]
        ax.scatter(sub['d_log_chn_us'], sub['d_log_mex_us'],
                   c=COLORS[route], s=14, alpha=0.55, linewidths=0, zorder=3)
        if route in slopes:
            m, b = slopes[route]
            xr = np.linspace(*X_LIM, 200)
            ax.plot(xr, m * xr + b, color='black', linewidth=1.1,
                    linestyle='--', alpha=0.55, zorder=4)
            ax.text(0.05, 0.95, f'slope = {m:.2f}',
                    transform=ax.transAxes, fontsize=7.5, va='top', color='#333')

    ax.axhline(0, color='#bbb', linewidth=0.6, linestyle='--', zorder=1)
    ax.axvline(0, color='#bbb', linewidth=0.6, linestyle='--', zorder=1)
    ax.axhspan(0, Y_LIM[1],
               xmin=0, xmax=(0 - X_LIM[0]) / (X_LIM[1] - X_LIM[0]),
               color='#e8f5e9', alpha=0.3, zorder=0)
    ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM)
    ax.set_xlabel(r'$\Delta\log(\mathrm{CHN}{\to}\mathrm{US})$', fontsize=8.5)
    ax.set_ylabel(r'$\Delta\log(\mathrm{MEX}{\to}\mathrm{US})$', fontsize=8.5)
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.tick_params(labelsize=7.5)

fig1.suptitle('Figure 1. Pre$\\to$Post change in log trade flows by HS6 sector, '
              'classified by trade route',
              fontsize=10, y=1.01)
plt.tight_layout(h_pad=2.5, w_pad=2)
fig1.savefig(f'{OUT_DIR}/figure_1.pdf', format='pdf')
fig1.savefig(f'{OUT_DIR}/figure_1.png', format='png')
plt.close()
print("  figure_1.pdf/.png saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Indexed aggregate trade flows
# ══════════════════════════════════════════════════════════════════════════════

print("Generating Figure 2...")

def index_series(df, col):
    """Index to 2016–2017 average = 100."""
    pre_avg = df[(df['date'] >= PRE_START) & (df['date'] <= PRE_END)][col].mean()
    df = df.copy()
    df['idx'] = df[col] / pre_avg * 100
    return df

mex_idx = index_series(mex_tot, 'mex')
cu_idx  = index_series(cu_tot,  'cu')
cm_idx  = index_series(cm_tot,  'cm')

fig2, ax2 = plt.subplots(figsize=(7, 4.0))

EVENTS = [
    ('2018-03', 'S.232',    '#BA7517'),
    ('2018-07', 'S.301 L1', '#C0392B'),
    ('2018-09', 'S.301 L3', '#C0392B'),
    ('2020-01', 'Phase 1',  '#27AE60'),
]
for ds, label, col in EVENTS:
    xd = pd.Timestamp(ds)
    ax2.axvline(xd, color=col, linewidth=0.7, linestyle='--', dashes=(4,3), alpha=0.7, zorder=1)
    ax2.text(xd - pd.Timedelta(days=20), 415, label, fontsize=6.5, color=col,
             rotation=90, va='top', ha='right', alpha=0.85)

ax2.plot(mex_idx['date'], mex_idx['idx'], color='#1D9E75', linewidth=2.0,
         label='Mexico $\\to$ US', zorder=3)
ax2.plot(cu_idx['date'],  cu_idx['idx'],  color='#378ADD', linewidth=1.6,
         linestyle='--', dashes=(6,2), label='China $\\to$ US', zorder=3)
ax2.plot(cm_idx['date'],  cm_idx['idx'],  color='#E24B4A', linewidth=1.6,
         linestyle=(0,(3,1,1,1)), label='China $\\to$ Mexico', zorder=3)

ax2.axvspan(pd.Timestamp('2016-01-01'), pd.Timestamp('2018-01-01'),
            color='#f0f0f0', alpha=0.6, zorder=0)
ax2.text(pd.Timestamp('2017-01-01'), 20, 'pre-tariff\nbaseline',
         ha='center', va='bottom', fontsize=7.5, color='#888')
ax2.axhline(100, color='#999', linewidth=0.5, zorder=1)
ax2.set_ylabel('Index (2016--2017 avg = 100)', fontsize=9)
ax2.set_ylim(0, 420)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(100))
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(labelsize=8)
ax2.legend(handles=[
    Line2D([0],[0], color='#1D9E75', linewidth=2.0,  label='Mexico $\\to$ US'),
    Line2D([0],[0], color='#378ADD', linewidth=1.6,  linestyle='--', dashes=(6,2),
           label='China $\\to$ US'),
    Line2D([0],[0], color='#E24B4A', linewidth=1.6,  linestyle=(0,(3,1,1,1)),
           label='China $\\to$ Mexico'),
], fontsize=8.5, frameon=False, loc='upper left')
ax2.set_title('Figure 2. Manufacturing export flows, indexed to pre-tariff baseline',
              fontsize=10, fontweight='normal', loc='left', pad=8)

fig2.savefig(f'{OUT_DIR}/figure_2.pdf', format='pdf')
fig2.savefig(f'{OUT_DIR}/figure_2.png', format='png')
plt.close()
print("  figure_2.pdf/.png saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 3 & 4 — Event studies (pre-computed coefficients)
#
# These coefficients were estimated from the main DiD panel.
# To update them, re-run did_estimation.py and paste the output of
# event_study.csv into the MEX_ES and CHN_ES lists below.
# ══════════════════════════════════════════════════════════════════════════════

MEX_ES = [
    {'x':-24,'y':0.1046,'ci_lo':-0.4275,'ci_hi':0.6367},
    {'x':-23,'y':0.3023,'ci_lo':-0.2265,'ci_hi':0.8311},
    {'x':-22,'y':0.1892,'ci_lo':-0.332, 'ci_hi':0.7104},
    {'x':-21,'y':0.1315,'ci_lo':-0.385, 'ci_hi':0.648},
    {'x':-20,'y':-0.1554,'ci_lo':-0.6634,'ci_hi':0.3526},
    {'x':-18,'y':-0.5505,'ci_lo':-1.0746,'ci_hi':-0.0264},
    {'x':-17,'y':0.2582,'ci_lo':-0.2624,'ci_hi':0.7788},
    {'x':-16,'y':0.0648,'ci_lo':-0.4489,'ci_hi':0.5785},
    {'x':-15,'y':-0.0608,'ci_lo':-0.5786,'ci_hi':0.457},
    {'x':-14,'y':-0.3585,'ci_lo':-0.9018,'ci_hi':0.1848},
    {'x':-13,'y':0.1437,'ci_lo':-0.3924,'ci_hi':0.6798},
    {'x':-12,'y':0.3428,'ci_lo':-0.1723,'ci_hi':0.8579},
    {'x':-11,'y':0.2663,'ci_lo':-0.2509,'ci_hi':0.7835},
    {'x':-10,'y':-0.2377,'ci_lo':-0.7612,'ci_hi':0.2858},
    {'x':-9, 'y':0.1016,'ci_lo':-0.4084,'ci_hi':0.6116},
    {'x':-8, 'y':-0.0974,'ci_lo':-0.6172,'ci_hi':0.4224},
    {'x':-6, 'y':-0.5764,'ci_lo':-1.1158,'ci_hi':-0.037},
    {'x':-5, 'y':0.2641,'ci_lo':-0.2477,'ci_hi':0.7759},
    {'x':-4, 'y':-0.2342,'ci_lo':-0.7542,'ci_hi':0.2858},
    {'x':-3, 'y':0.2451,'ci_lo':-0.2625,'ci_hi':0.7527},
    {'x':-2, 'y':-0.2221,'ci_lo':-0.7299,'ci_hi':0.2857},
    {'x':-1, 'y':0.0792,'ci_lo':-0.4557,'ci_hi':0.6141},
    {'x':0,  'y':0.2112,'ci_lo':-0.2919,'ci_hi':0.7143},
    {'x':1,  'y':0.4391,'ci_lo':-0.0701,'ci_hi':0.9483},
    {'x':2,  'y':0.1455,'ci_lo':-0.3588,'ci_hi':0.6498},
    {'x':3,  'y':0.0691,'ci_lo':-0.4289,'ci_hi':0.5671},
    {'x':4,  'y':0.6071,'ci_lo':0.1004, 'ci_hi':1.1138},
    {'x':6,  'y':-0.3188,'ci_lo':-0.8364,'ci_hi':0.1988},
    {'x':7,  'y':-0.3186,'ci_lo':-0.8237,'ci_hi':0.1865},
    {'x':8,  'y':-0.085, 'ci_lo':-0.605, 'ci_hi':0.435},
    {'x':9,  'y':0.1362,'ci_lo':-0.3644,'ci_hi':0.6368},
    {'x':10, 'y':-0.1114,'ci_lo':-0.6149,'ci_hi':0.3921},
    {'x':11, 'y':0.2536,'ci_lo':-0.2499,'ci_hi':0.7571},
    {'x':12, 'y':-0.2714,'ci_lo':-0.7712,'ci_hi':0.2284},
    {'x':13, 'y':-0.1543,'ci_lo':-0.6598,'ci_hi':0.3512},
    {'x':14, 'y':-0.07,  'ci_lo':-0.5639,'ci_hi':0.4239},
    {'x':15, 'y':0.5307,'ci_lo':0.0317, 'ci_hi':1.0297},
    {'x':16, 'y':0.8068,'ci_lo':0.295,  'ci_hi':1.3186},
    {'x':18, 'y':0.104, 'ci_lo':-0.4183,'ci_hi':0.6263},
    {'x':19, 'y':0.1976,'ci_lo':-0.3145,'ci_hi':0.7097},
    {'x':20, 'y':0.2817,'ci_lo':-0.2146,'ci_hi':0.778},
    {'x':21, 'y':-0.0615,'ci_lo':-0.6189,'ci_hi':0.4959},
    {'x':22, 'y':-0.2845,'ci_lo':-0.8545,'ci_hi':0.2855},
    {'x':23, 'y':-0.0682,'ci_lo':-0.6011,'ci_hi':0.4647},
    {'x':24, 'y':0.3871,'ci_lo':-0.1405,'ci_hi':0.9147},
    {'x':25, 'y':0.4515,'ci_lo':-0.0612,'ci_hi':0.9642},
    {'x':26, 'y':0.2399,'ci_lo':-0.2842,'ci_hi':0.764},
    {'x':27, 'y':0.3132,'ci_lo':-0.1946,'ci_hi':0.821},
    {'x':28, 'y':0.7984,'ci_lo':0.2794, 'ci_hi':1.3174},
    {'x':30, 'y':0.2161,'ci_lo':-0.2872,'ci_hi':0.7194},
]

CHN_ES = [
    {'x':-24,'y':0.1291,'ci_lo':-0.2776,'ci_hi':0.5358},
    {'x':-23,'y':0.6845,'ci_lo':0.2905, 'ci_hi':1.0785},
    {'x':-22,'y':0.399, 'ci_lo':-0.0124,'ci_hi':0.8104},
    {'x':-21,'y':0.3322,'ci_lo':-0.0818,'ci_hi':0.7462},
    {'x':-20,'y':0.2118,'ci_lo':-0.1894,'ci_hi':0.613},
    {'x':-18,'y':0.3485,'ci_lo':-0.0053,'ci_hi':0.7023},
    {'x':-17,'y':-0.2122,'ci_lo':-0.6064,'ci_hi':0.182},
    {'x':-16,'y':-0.3234,'ci_lo':-0.6942,'ci_hi':0.0474},
    {'x':-15,'y':-0.0619,'ci_lo':-0.4221,'ci_hi':0.2983},
    {'x':-14,'y':0.0565,'ci_lo':-0.2904,'ci_hi':0.4034},
    {'x':-13,'y':-0.111, 'ci_lo':-0.4638,'ci_hi':0.2418},
    {'x':-12,'y':0.2273,'ci_lo':-0.1182,'ci_hi':0.5728},
    {'x':-11,'y':0.0108,'ci_lo':-0.3297,'ci_hi':0.3513},
    {'x':-10,'y':-0.1081,'ci_lo':-0.4713,'ci_hi':0.2551},
    {'x':-9, 'y':-0.1129,'ci_lo':-0.4696,'ci_hi':0.2438},
    {'x':-8, 'y':-0.0966,'ci_lo':-0.4437,'ci_hi':0.2505},
    {'x':-6, 'y':0.0463,'ci_lo':-0.3094,'ci_hi':0.402},
    {'x':-5, 'y':-0.1994,'ci_lo':-0.5626,'ci_hi':0.1638},
    {'x':-4, 'y':-0.639, 'ci_lo':-1.0386,'ci_hi':-0.2394},
    {'x':-3, 'y':-0.3856,'ci_lo':-0.737, 'ci_hi':-0.0342},
    {'x':-2, 'y':-0.2524,'ci_lo':-0.5956,'ci_hi':0.0908},
    {'x':-1, 'y':0.0564,'ci_lo':-0.2801,'ci_hi':0.3929},
    {'x':0,  'y':-0.0275,'ci_lo':-0.3687,'ci_hi':0.3137},
    {'x':1,  'y':0.1308,'ci_lo':-0.1897,'ci_hi':0.4513},
    {'x':2,  'y':-0.0632,'ci_lo':-0.4105,'ci_hi':0.2841},
    {'x':3,  'y':-0.4247,'ci_lo':-0.7681,'ci_hi':-0.0813},
    {'x':4,  'y':-0.2676,'ci_lo':-0.6165,'ci_hi':0.0813},
    {'x':6,  'y':0.0979,'ci_lo':-0.221, 'ci_hi':0.4168},
    {'x':7,  'y':-0.4788,'ci_lo':-0.8418,'ci_hi':-0.1158},
    {'x':8,  'y':-1.2804,'ci_lo':-1.6248,'ci_hi':-0.936},
    {'x':9,  'y':-0.5452,'ci_lo':-0.8768,'ci_hi':-0.2136},
    {'x':10, 'y':-0.423, 'ci_lo':-0.7589,'ci_hi':-0.0871},
    {'x':11, 'y':-0.4624,'ci_lo':-0.7838,'ci_hi':-0.141},
    {'x':12, 'y':-0.279, 'ci_lo':-0.6051,'ci_hi':0.0471},
    {'x':13, 'y':-0.2025,'ci_lo':-0.5357,'ci_hi':0.1307},
    {'x':14, 'y':-0.7823,'ci_lo':-1.1177,'ci_hi':-0.4469},
    {'x':15, 'y':-0.5734,'ci_lo':-0.8982,'ci_hi':-0.2486},
    {'x':16, 'y':-0.7035,'ci_lo':-1.0253,'ci_hi':-0.3817},
    {'x':18, 'y':-0.5841,'ci_lo':-0.9146,'ci_hi':-0.2536},
    {'x':19, 'y':-0.6112,'ci_lo':-1.0469,'ci_hi':-0.1755},
    {'x':20, 'y':-0.7988,'ci_lo':-1.1569,'ci_hi':-0.4407},
    {'x':21, 'y':-1.2687,'ci_lo':-1.6289,'ci_hi':-0.9085},
    {'x':22, 'y':-0.6419,'ci_lo':-0.9949,'ci_hi':-0.2889},
    {'x':23, 'y':-0.4322,'ci_lo':-0.7926,'ci_hi':-0.0718},
    {'x':24, 'y':-0.2767,'ci_lo':-0.627, 'ci_hi':0.0736},
    {'x':25, 'y':-0.237, 'ci_lo':-0.5904,'ci_hi':0.1164},
    {'x':26, 'y':-0.1326,'ci_lo':-0.4795,'ci_hi':0.2143},
    {'x':27, 'y':-0.5041,'ci_lo':-0.8451,'ci_hi':-0.1631},
    {'x':28, 'y':-0.6344,'ci_lo':-0.9611,'ci_hi':-0.3077},
    {'x':30, 'y':-0.6798,'ci_lo':-1.0099,'ci_hi':-0.3497},
]


def draw_event_study(ax, data, color, title, notes, ylim):
    xs    = np.array([d['x']    for d in data])
    ys    = np.array([d['y']    for d in data])
    ci_lo = np.array([d['ci_lo'] for d in data])
    ci_hi = np.array([d['ci_hi'] for d in data])

    # CI band
    ax.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.13, linewidth=0)
    # Line
    ax.plot(xs, ys, color=color, linewidth=1.8, zorder=3)
    # Points — filled if significant
    for x, y, lo, hi in zip(xs, ys, ci_lo, ci_hi):
        sig = (lo > 0) or (hi < 0)
        ax.plot(x, y, marker='o',
                markersize=4 if sig else 3,
                color=color,
                markerfacecolor=color if sig else 'white',
                markeredgewidth=1.2, linewidth=0, zorder=4)

    # Reference lines
    ax.axvline(0, color='#444', linewidth=0.8, linestyle='--', dashes=(4,3), zorder=2)
    ax.axhline(0, color='#999', linewidth=0.5, zorder=1)
    # Pre-tariff shading
    ax.axvspan(xs.min()-0.5, 0, color='#f5f5f5', zorder=0)

    ax.set_xlim(xs.min()-1, xs.max()+1)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Months relative to Section 301 List 1 (Jul 2018)', fontsize=9)
    ax.set_ylabel('Coefficient (normalized)', fontsize=9)
    ax.xaxis.set_ticks(range(-24, 31, 6))
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10, fontweight='normal', loc='left', pad=8)

    ax.legend(handles=[
        Line2D([0],[0], color=color, linewidth=1.8, label='coefficient'),
        mpatches.Patch(facecolor=color, alpha=0.25, label='95% CI'),
        Line2D([0],[0], marker='o', color=color, markersize=5,
               markerfacecolor=color, linewidth=0, label='significant at 5%'),
        Line2D([0],[0], marker='o', color=color, markersize=4,
               markerfacecolor='white', markeredgewidth=1.2,
               linewidth=0, label='insignificant'),
    ], fontsize=8, frameon=False, loc='upper left', ncol=2)

    return notes


ES_SPECS = [
    (3, MEX_ES, '#1D9E75',
     'Figure 3. Event study --- MEX$\\to$US manufacturing exports (nearshoring test)',
     ('Notes: Coefficients from interacting the pre-tariff Chinese US import share with '
      'period dummies, normalized so the pre-period mean equals zero. Two-way FE '
      '(HS6 + year-month). SE clustered at HS2 level (41 clusters). December months '
      'excluded to remove Comtrade reporting artifacts. Filled circles denote months '
      'where the 95\\% CI excludes zero. Shaded region is the pre-tariff period. '
      '$n = 191{,}763$.'),
     (-1.4, 1.6),
     'figure_3'),
    (4, CHN_ES, '#C0392B',
     'Figure 4. Event study --- CHN$\\to$MEX manufacturing exports (deflection test)',
     ('Notes: Same specification as Figure 3. Outcome is Chinese exports into Mexico '
      'at the HS6-month level. The persistent negative shift beginning at approximately '
      'month $+8$ rules out transit deflection and is consistent with supply chain '
      'restructuring away from Chinese inputs as Mexican producers expand US-bound '
      'production.'),
     (-1.9, 1.3),
     'figure_4'),
]

import textwrap

for fig_num, data, color, title, notes, ylim, fname in ES_SPECS:
    print(f"Generating Figure {fig_num}...")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    draw_event_study(ax, data, color, title, notes, ylim)

    # Hard-wrap notes at 110 chars so they stay within the 7-inch figure width
    wrapped_notes = '\n'.join(textwrap.wrap(notes, width=110))
    fig.text(0.0, 0.01, wrapped_notes,
             fontsize=7.5, color='#555',
             transform=fig.transFigure,
             va='bottom', ha='left',
             multialignment='left',
             linespacing=1.4)

    fig.subplots_adjust(bottom=0.18, top=0.95, left=0.09, right=0.97)
    fig.savefig(f'{OUT_DIR}/{fname}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/{fname}.png', format='png', bbox_inches='tight')
    plt.close()
    print(f"  {fname}.pdf/.png saved.")

print("\nAll figures generated successfully.")
print(f"Output folder: {OUT_DIR}")

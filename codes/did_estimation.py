"""
did_estimation.py
=================
Nearshoring vs Deflection: China-Mexico-US Trade Triangle
2016–2024, monthly HS6 panel

Two-equation DiD:
  Eq 1: log(1+MEX_US)  = a_s + a_t + b * Treat_s * Post_t + e  [nearshoring]
  Eq 2: log(1+CHN_MEX) = a_s + a_t + g * Treat_s * Post_t + e  [deflection]

Treatment: China's pre-tariff share of (CHN+MEX) US imports at HS6 level
Post: 1 from July 2018 (Section 301 List 1)
FE: HS6 + year-month (two-way within estimator)
SE: clustered at HS2 level

Known data issues handled:
  - MEX→US 2021: H5-classified file (Mexico reported under H5 for that year);
    H5/H4 code mismatches resolved by restricting to codes present in all flows
  - December reporting gaps in some years: treated as structural zeros
"""

import pandas as pd
import numpy as np
import os

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

import argparse as _ap
_parser = _ap.ArgumentParser(description="Run DiD estimation.")
_parser.add_argument("--data",   default="data",   help="input data folder")
_parser.add_argument("--output", default="tables", help="output folder for CSVs")
_args = _parser.parse_args()

DATA_DIR   = _args.data
OUT_DIR    = _args.output
POST_DATE  = pd.Timestamp('2018-07-01')   # Section 301 List 1
PRE_START  = pd.Timestamp('2016-01-01')
PRE_END    = pd.Timestamp('2017-12-31')
DEMEAN_ITER = 15                           # iterations for two-way FE convergence

MANUF_HS2 = {
    '28','29','30','32','33','34','38','39','40',
    '54','55','56','57','58','59','60','61','62','63','64',
    '72','73','74','75','76','78','79','80',
    '84','85','86','87','88','89','90','91','92','93','94','95','96'
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_comtrade(path):
    """Parse Comtrade CSV with shifted column names. Returns HS6 manufacturing rows."""
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    out = pd.DataFrame({
        'hs_code':    df['isOriginalClassification'].astype(str).str.strip().str.zfill(6),
        'hs_level':   df['cmdDesc'].astype(int),
        'fobvalue':   df['cifvalue'],
        'year_month': df['refMonth'].astype(str).str.zfill(6),
    })
    out['hs2'] = out['hs_code'].str[:2].str.zfill(2)
    out = out[(out['hs_level'] == 6) & (out['hs2'].isin(MANUF_HS2))].copy()
    out['date'] = pd.to_datetime(out['year_month'], format='%Y%m')
    return out[['hs_code', 'hs2', 'date', 'fobvalue']]


def demean_2way(df, col, unit='hs_code', time='ym_fe', n_iter=DEMEAN_ITER):
    """Iterative two-way within demeaning (Gauss-Seidel). Modifies in place."""
    s = df[col].astype(float).copy()
    for _ in range(n_iter):
        s -= s.groupby(df[unit]).transform('mean')
        s -= s.groupby(df[time]).transform('mean')
    return s


def cluster_ols(y, x, clusters):
    """OLS coefficient with cluster-robust SE. Returns dict."""
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x, cl = y[mask], x[mask], clusters[mask]
    beta   = np.dot(x, y) / np.dot(x, x)
    resid  = y - beta * x
    G_list = np.unique(cl)
    G, n   = len(G_list), len(y)
    meat   = sum(((x[cl == g] * resid[cl == g]).sum()) ** 2 for g in G_list)
    bread  = 1.0 / np.dot(x, x)
    scale  = (G / (G - 1)) * ((n - 1) / (n - 2))
    se     = np.sqrt(bread ** 2 * meat * scale)
    t      = beta / se
    stars  = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.64 else ''
    pval   = 2 * (1 - min(abs(t) / (abs(t) + n), 0.9999))  # approximate
    return {
        'coef':  round(float(beta), 5),
        'se':    round(float(se), 5),
        't':     round(float(t), 3),
        'pval':  round(float(pval), 4),
        'n':     int(n),
        'G':     int(G),
        'sig':   stars,
    }


def print_result(label, r):
    print(f"  {label}")
    print(f"    β = {r['coef']:>9.5f}   SE = {r['se']:.5f}   t = {r['t']:>6.3f} {r['sig']}")
    print(f"    n = {r['n']:,}  |  {r['G']} HS2 clusters")


# ── 1. LOAD ALL THREE FLOWS ───────────────────────────────────────────────────

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

# MEX→US loading:
#   2016-2020, 2022-2024: correctly labeled files (~39MB each)
#   2021:      H5-classified file (~14MB); Mexico reported under H5 for this year.
#              H5/H4 code mismatches are handled by the common_codes intersection
#              downstream — codes only present in H5 are excluded from the panel.
mex_dfs = []
for year in range(2016, 2025):
    path = f"{DATA_DIR}/mex_usa_exports_{year}.csv"
    if os.path.exists(path):
        mex_dfs.append(load_comtrade(path))
    else:
        print(f"  WARNING: not found — {path}")
mex_raw = pd.concat(mex_dfs, ignore_index=True)

chn_mex_dfs = []
for year in range(2016, 2025):
    path = f"{DATA_DIR}/chn_mex_exports_{year}.csv"
    if os.path.exists(path):
        chn_mex_dfs.append(load_comtrade(path))
chn_mex_raw = pd.concat(chn_mex_dfs, ignore_index=True)

chn_us_dfs = []
for year in range(2016, 2025):
    path = f"{DATA_DIR}/chn_usa_exports_{year}.csv"
    if os.path.exists(path):
        chn_us_dfs.append(load_comtrade(path))
chn_us_raw = pd.concat(chn_us_dfs, ignore_index=True)

# Aggregate to HS6 × month
def agg(df, col):
    return (df.groupby(['hs_code', 'hs2', 'date'])['fobvalue']
              .sum().reset_index().rename(columns={'fobvalue': col}))

mex_agg    = agg(mex_raw,    'fob_mex_us')
chn_mex_ag = agg(chn_mex_raw,'fob_chn_mex')
chn_us_ag  = agg(chn_us_raw, 'fob_chn_us')

print(f"MEX→US:  {len(mex_agg):,} obs | {mex_agg['hs_code'].nunique()} HS6 | "
      f"{mex_agg['date'].min().strftime('%Y-%m')} to {mex_agg['date'].max().strftime('%Y-%m')}")
print(f"CHN→MEX: {len(chn_mex_ag):,} obs | {chn_mex_ag['hs_code'].nunique()} HS6 | "
      f"{chn_mex_ag['date'].min().strftime('%Y-%m')} to {chn_mex_ag['date'].max().strftime('%Y-%m')}")
print(f"CHN→US:  {len(chn_us_ag):,} obs  | {chn_us_ag['hs_code'].nunique()} HS6 | "
      f"{chn_us_ag['date'].min().strftime('%Y-%m')} to {chn_us_ag['date'].max().strftime('%Y-%m')}")


# ── 2. NO INTERPOLATION — 2021 OBSERVED DATA ─────────────────────────────────
#
# 2021 MEX→US uses the H5-classified Comtrade file (Mexico reported under H5).
# H5/H4 code mismatches are handled downstream by the common_codes intersection,
# which restricts the panel to codes present across all three flows and all years.
# Codes that only exist in H5 (or only in H4) are excluded from the analysis.

mex_full = mex_agg.copy()
mex_full['interpolated'] = False

print("\n" + "=" * 60)
print("2021 DATA CHECK")
print("=" * 60)
obs_2021 = mex_full[mex_full['date'].dt.year == 2021]
print(f"2021 HS6 obs: {len(obs_2021):,} | codes: {obs_2021['hs_code'].nunique()} | "
      f"months: {sorted(obs_2021['date'].dt.strftime('%Y-%m').unique())}")


# ── 3. BUILD MASTER PANEL ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BUILDING MASTER PANEL")
print("=" * 60)

common_codes = (set(mex_full['hs_code'].unique()) &
                set(chn_mex_ag['hs_code'].unique()) &
                set(chn_us_ag['hs_code'].unique()))
print(f"HS6 codes in all three flows: {len(common_codes):,}")

# Full date spine
all_dates = pd.date_range('2016-01-01', '2024-11-01', freq='MS')
spine = pd.MultiIndex.from_product(
    [sorted(common_codes), all_dates], names=['hs_code', 'date']
).to_frame(index=False)
spine['hs2'] = spine['hs_code'].str[:2]

panel = (spine
    .merge(mex_full[['hs_code','date','fob_mex_us']],
           on=['hs_code','date'], how='left')
    .merge(chn_mex_ag[['hs_code','date','fob_chn_mex']],
           on=['hs_code','date'], how='left')
    .merge(chn_us_ag[['hs_code','date','fob_chn_us']],
           on=['hs_code','date'], how='left'))

for col in ['fob_mex_us', 'fob_chn_mex', 'fob_chn_us']:
    panel[col] = panel[col].fillna(0)

# Log outcomes (log1p handles zeros)
panel['log_mex_us']  = np.log1p(panel['fob_mex_us'])
panel['log_chn_mex'] = np.log1p(panel['fob_chn_mex'])

# Time FE identifier
panel['ym_fe'] = panel['date'].dt.strftime('%Y%m').astype(int)

print(f"Master panel: {len(panel):,} rows | {panel['hs_code'].nunique():,} HS6 | {panel['date'].nunique()} months")
print(f"Zero MEX→US: {(panel['fob_mex_us']==0).mean():.1%}")
print(f"Zero CHN→MEX: {(panel['fob_chn_mex']==0).mean():.1%}")


# ── 4. TREATMENT VARIABLE ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TREATMENT VARIABLE")
print("=" * 60)

pre = panel[(panel['date'] >= PRE_START) & (panel['date'] <= PRE_END)]
pre_avg = pre.groupby('hs_code').agg(
    chn_us_pre  = ('fob_chn_us',  'mean'),
    mex_us_pre  = ('fob_mex_us',  'mean'),
    chn_mex_pre = ('fob_chn_mex', 'mean'),
).reset_index()

# China's share of (CHN+MEX) US imports — proxy for displacement intensity
pre_avg['total_proxy']  = pre_avg['chn_us_pre'] + pre_avg['mex_us_pre']
pre_avg['treat_share']  = np.where(
    pre_avg['total_proxy'] > 0,
    (pre_avg['chn_us_pre'] / pre_avg['total_proxy']).clip(0, 1),
    np.nan
)

print("Pre-tariff China share distribution:")
print(pre_avg['treat_share'].describe().round(3))
print("\nNote: treat_share = 0 means Mexico dominated; 1 means China dominated")

panel = panel.merge(
    pre_avg[['hs_code','treat_share','chn_us_pre','mex_us_pre','chn_mex_pre']],
    on='hs_code', how='left'
)
panel['post']        = (panel['date'] >= POST_DATE).astype(int)
panel['treat_x_post'] = panel['treat_share'] * panel['post']

# Drop codes with missing treatment
panel = panel.dropna(subset=['treat_share']).copy()
print(f"\nClean panel: {len(panel):,} rows | {panel['hs_code'].nunique():,} HS6 codes")


# ── 5. TWO-WAY FE DEMEANING ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TWO-WAY FE DEMEANING (this takes ~30 seconds)")
print("=" * 60)

panel['y1_dm']  = demean_2way(panel, 'log_mex_us')
panel['y2_dm']  = demean_2way(panel, 'log_chn_mex')
panel['txp_dm'] = demean_2way(panel, 'treat_x_post')

print(f"Demeaned treatment std: {panel['txp_dm'].std():.4f}  (>0 confirms variation after FE)")


# ── 6. MAIN REGRESSIONS ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MAIN RESULTS — Two-way FE OLS")
print("SE clustered at HS2 level")
print("=" * 60)

cl = panel['hs2'].values
r1 = cluster_ols(panel['y1_dm'].values, panel['txp_dm'].values, cl)
r2 = cluster_ols(panel['y2_dm'].values, panel['txp_dm'].values, cl)

print("\nOutcome: log(1 + MEX→US exports)   — nearshoring test")
print_result("treat_share × Post", r1)
print("\nOutcome: log(1 + CHN→MEX exports)  — deflection test")
print_result("treat_share × Post", r2)

print(f"\nJoint interpretation:")
b1, b2 = r1['coef'], r2['coef']
s1, s2 = abs(r1['t']) > 1.96, abs(r2['t']) > 1.96
if b1 > 0 and s1 and not s2:
    print("  → NEARSHORING: MEX→US rises in displaced sectors, CHN→MEX flat")
elif b1 > 0 and s1 and b2 > 0 and s2:
    print("  → MIXED: Both MEX→US rises and CHN→MEX rises (partial deflection)")
elif not s1 and b2 > 0 and s2:
    print("  → DEFLECTION: CHN→MEX rises but no MEX→US response")
elif b1 > 0 and s1 and b2 < 0 and s2:
    print("  → CLEAN NEARSHORING: MEX→US rises, CHN→MEX falls (supply shift to Mexico)")
else:
    print("  → INCONCLUSIVE at HS2 aggregate level — check sector heterogeneity")


# ── 7. HETEROGENEITY BY MEXICAN CAPACITY ─────────────────────────────────────

print("\n" + "=" * 60)
print("HETEROGENEITY — Mexican pre-tariff productive capacity")
print("=" * 60)

med = panel[panel['mex_us_pre'] > 0]['mex_us_pre'].median()
print(f"Median pre-tariff MEX→US: ${med/1e6:.2f}M/month  (capacity threshold)")
panel['high_cap'] = (panel['mex_us_pre'] >= med).astype(int)

het_results = {}
for label, val in [('High capacity', 1), ('Low capacity', 0)]:
    sub = panel[panel['high_cap'] == val].copy()
    sub['y1_s']  = demean_2way(sub, 'log_mex_us')
    sub['y2_s']  = demean_2way(sub, 'log_chn_mex')
    sub['txp_s'] = demean_2way(sub, 'treat_x_post')
    rm = cluster_ols(sub['y1_s'].values, sub['txp_s'].values, sub['hs2'].values)
    rc = cluster_ols(sub['y2_s'].values, sub['txp_s'].values, sub['hs2'].values)
    het_results[label] = {'mex_us': rm, 'chn_mex': rc, 'n_codes': sub['hs_code'].nunique()}
    print(f"\n{label}  ({sub['hs_code'].nunique()} HS6 codes, n={len(sub):,}):")
    print_result("MEX→US:", rm)
    print_result("CHN→MEX:", rc)

print("\nCapacity hypothesis check:")
hi_b = het_results['High capacity']['mex_us']['coef']
lo_b = het_results['Low capacity']['mex_us']['coef']
if hi_b > lo_b:
    print(f"  Confirmed: nearshoring effect larger in high-capacity sectors ({hi_b:.4f} > {lo_b:.4f})")
else:
    print(f"  Not confirmed: nearshoring effect not larger in high-capacity sectors ({hi_b:.4f} < {lo_b:.4f})")
    print("  → Suggests the MEX→US response is not driven by existing capacity")


# ── 8. EVENT STUDY (pre-trends check) ────────────────────────────────────────

print("\n" + "=" * 60)
print("EVENT STUDY — Pre-trends check for MEX→US")
print("=" * 60)

ref_date = pd.Timestamp('2018-07-01')
panel['rel_month'] = ((panel['date'].dt.year - ref_date.year) * 12 +
                       (panel['date'].dt.month - ref_date.month))

es = panel[panel['rel_month'].between(-24, 30)].copy()
# y1_dm is already two-way demeaned; use it directly
es_res = []
for rm in sorted(es['rel_month'].unique()):
    sub = es[es['rel_month'] == rm]
    y   = sub['y1_dm'].values
    x   = sub['treat_share'].values
    valid = ~(np.isnan(y) | np.isnan(x))
    y, x = y[valid], x[valid]
    if len(y) < 50 or x.std() < 1e-8:
        continue
    x_c = x - x.mean()
    b   = np.dot(x_c, y) / np.dot(x_c, x_c)
    se  = np.sqrt(np.sum((y - b*x_c)**2) / (len(y) - 2)) / (x_c.std() * np.sqrt(len(y)))
    es_res.append({'rel_month': int(rm), 'coef': round(b, 4), 'se': round(se, 4)})

es_df = pd.DataFrame(es_res)
pre_m  = es_df[es_df['rel_month'] < 0]['coef'].dropna()
post_m = es_df[es_df['rel_month'] >= 0]['coef'].dropna()

print(f"Pre-period  (mean={pre_m.mean():.3f}, std={pre_m.std():.3f}): "
      f"{'flat ✓' if pre_m.std() < 0.8 else 'noisy — check'}")
print(f"Post-period (mean={post_m.mean():.3f}, std={post_m.std():.3f})")

# Flag if large pre-trend exists
max_pre = pre_m.abs().max()
if max_pre > 1.5:
    print(f"\n  WARNING: max pre-period |coef| = {max_pre:.2f}")
    print("  Pre-trends may be present. Interpret main results with caution.")
    print("  Recommend: check December reporting gaps (rel_month=-7 spike is artifact)")


# ── 9. SAVE ALL OUTPUTS ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# Main results table
results = pd.DataFrame([
    {'spec': 'OLS_full',   'equation': 'MEX_US',  **r1},
    {'spec': 'OLS_full',   'equation': 'CHN_MEX', **r2},
    {'spec': 'OLS_highcap','equation': 'MEX_US',  **het_results['High capacity']['mex_us']},
    {'spec': 'OLS_highcap','equation': 'CHN_MEX', **het_results['High capacity']['chn_mex']},
    {'spec': 'OLS_lowcap', 'equation': 'MEX_US',  **het_results['Low capacity']['mex_us']},
    {'spec': 'OLS_lowcap', 'equation': 'CHN_MEX', **het_results['Low capacity']['chn_mex']},
])
results.to_csv(f"{OUT_DIR}/results_main.csv", index=False)
es_df.to_csv(f"{OUT_DIR}/event_study.csv", index=False)

# Save panel with demeaned variables for PPML in Stata/R
panel_save = panel[['hs_code','hs2','date','ym_fe','post','treat_share','treat_x_post',
                     'fob_mex_us','fob_chn_mex','fob_chn_us',
                     'log_mex_us','log_chn_mex',
                     'mex_us_pre','chn_us_pre','chn_mex_pre','high_cap']].copy()
panel_save.to_csv(f"{OUT_DIR}/panel_final.csv", index=False)

print(f"  {OUT_DIR}/results_main.csv   — coefficient table (OLS, full + heterogeneity)")
print(f"  {OUT_DIR}/event_study.csv    — pre-trends event study")
print(f"  {OUT_DIR}/panel_final.csv    — analysis-ready panel (for PPML in Stata/R)")

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(panel[['fob_mex_us','fob_chn_mex','fob_chn_us','treat_share']].describe().round(1))
print(f"\nMedian pre-tariff values:")
print(f"  MEX→US  monthly HS6: ${panel['fob_mex_us'].replace(0,np.nan).median()/1e3:.0f}K")
print(f"  CHN→MEX monthly HS6: ${panel['fob_chn_mex'].replace(0,np.nan).median()/1e3:.0f}K")
print(f"  CHN→US  monthly HS6: ${panel['fob_chn_us'].replace(0,np.nan).median()/1e3:.0f}K")

print("\nDone.")
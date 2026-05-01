"""
trade_model_final.py
====================
Nearshoring vs Deflection: China-Mexico-US Trade Triangle
2016-2024, monthly HS6 panel

Two-equation DiD:
  Eq 1: log(1+MEX_US)  = a_s + a_t + b * Treat_s * Post_t + e  [nearshoring]
  Eq 2: log(1+CHN_MEX) = a_s + a_t + g * Treat_s * Post_t + e  [deflection]

Treatment: China's pre-tariff share of (CHN+MEX) US imports at HS6 level
Post:      1 from July 2018 (Section 301 List 1)
FE:        HS6 + year-month (two-way within estimator, iterative demeaning)
SE:        clustered at HS2 level

Data file naming convention (set DATA_DIR below):
  mex_usa_exports_{2016..2024}.csv   — Mexico -> US (2021 is H5-classified)
  chn_mex_exports_{2016..2024}.csv   — China  -> Mexico
  chn_usa_exports_{2016..2024}.csv   — China  -> US

Known data issues handled:
  - MEX->US 2021: H5-classified file (Mexico reported under H5 for that year)
    H5/H4 code mismatches resolved by restricting to codes present in all
    three flows across all years (common_codes intersection)
  - December reporting gaps: flagged in event study; drop-December
    robustness check included in section 8
"""

import pandas as pd
import numpy as np
import os

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

import argparse as _ap
_parser = _ap.ArgumentParser(description="Build panel and run regressions.")
_parser.add_argument("--data",   default="data",   help="input data folder")
_parser.add_argument("--output", default="tables", help="output folder for CSVs")
_args = _parser.parse_args()

DATA_DIR    = _args.data
OUT_DIR     = _args.output
POST_DATE   = pd.Timestamp('2018-07-01')  # Section 301 List 1
PRE_START   = pd.Timestamp('2016-01-01')
PRE_END     = pd.Timestamp('2017-12-31')
DEMEAN_ITER = 15  # Gauss-Seidel iterations for two-way FE convergence

MANUF_HS2 = {
    '28','29','30','32','33','34','38','39','40',
    '54','55','56','57','58','59','60','61','62','63','64',
    '72','73','74','75','76','78','79','80',
    '84','85','86','87','88','89','90','91','92','93','94','95','96'
}


# ── HELPERS ────────────────────────────────────────────────────────────────────

def load_comtrade(path):
    """
    Load a UN Comtrade CSV and return clean HS6 manufacturing rows.

    Column mapping (Comtrade shifted-column format):
      isOriginalClassification -> hs_code
      cmdDesc                  -> HS aggregation level (2, 4, or 6)
      cifvalue                 -> FOB trade value (USD)
      refMonth                 -> year-month integer (YYYYMM)
    """
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


def agg_to_hs6_month(df, value_col):
    """Aggregate raw rows to HS6 x month totals."""
    return (df.groupby(['hs_code', 'hs2', 'date'])['fobvalue']
              .sum().reset_index()
              .rename(columns={'fobvalue': value_col}))


def demean_2way(df, col, unit='hs_code', time='ym_fe', n_iter=DEMEAN_ITER):
    """
    Iterative two-way within demeaning (Gauss-Seidel).
    Absorbs unit (hs_code) and time (ym_fe) fixed effects.
    Returns a Series aligned to df's index.
    """
    s = df[col].astype(float).copy()
    for _ in range(n_iter):
        s -= s.groupby(df[unit]).transform('mean')
        s -= s.groupby(df[time]).transform('mean')
    return s


def cluster_ols(y, x, clusters):
    """
    OLS with cluster-robust standard errors (HC1 small-sample correction).
    Returns a dict with coef, se, t, n, G, sig.
    """
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
    return {
        'coef': round(float(beta), 5),
        'se':   round(float(se),   5),
        't':    round(float(t),    3),
        'n':    int(n),
        'G':    int(G),
        'sig':  stars,
    }


def print_result(label, r):
    print(f"  {label}")
    print(f"    b = {r['coef']:>9.5f}   SE = {r['se']:.5f}   "
          f"t = {r['t']:>6.3f} {r['sig']}")
    print(f"    n = {r['n']:,}  |  {r['G']} HS2 clusters")


# ── 1. LOAD DATA ───────────────────────────────────────────────────────────────

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

mex_dfs, cm_dfs, cu_dfs = [], [], []
for year in range(2016, 2025):
    for dfs, prefix in [(mex_dfs, 'mex_usa_exports'),
                        (cm_dfs,  'chn_mex_exports'),
                        (cu_dfs,  'chn_usa_exports')]:
        path = f"{DATA_DIR}/{prefix}_{year}.csv"
        if os.path.exists(path):
            dfs.append(load_comtrade(path))
        else:
            print(f"  WARNING: not found -- {path}")

mex_raw     = pd.concat(mex_dfs, ignore_index=True)
chn_mex_raw = pd.concat(cm_dfs,  ignore_index=True)
chn_us_raw  = pd.concat(cu_dfs,  ignore_index=True)

mex_agg     = agg_to_hs6_month(mex_raw,     'fob_mex_us')
chn_mex_agg = agg_to_hs6_month(chn_mex_raw, 'fob_chn_mex')
chn_us_agg  = agg_to_hs6_month(chn_us_raw,  'fob_chn_us')

for label, df in [('MEX->US',  mex_agg),
                  ('CHN->MEX', chn_mex_agg),
                  ('CHN->US',  chn_us_agg)]:
    print(f"{label}: {len(df):,} obs | {df['hs_code'].nunique()} HS6 | "
          f"{df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}")

# Quick 2021 check (H5-classified file)
obs_2021 = mex_agg[mex_agg['date'].dt.year == 2021]
print(f"\n2021 MEX->US: {len(obs_2021):,} obs | "
      f"{obs_2021['hs_code'].nunique()} codes | "
      f"months: {sorted(obs_2021['date'].dt.strftime('%Y-%m').unique())}")


# ── 2. BUILD MASTER PANEL ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("BUILDING MASTER PANEL")
print("=" * 60)

# Restrict to HS6 codes present in all three flows
# This automatically handles H5/H4 mismatches in the 2021 MEX->US file
common_codes = (set(mex_agg['hs_code'].unique()) &
                set(chn_mex_agg['hs_code'].unique()) &
                set(chn_us_agg['hs_code'].unique()))
print(f"HS6 codes in all three flows: {len(common_codes):,}")

# Full date spine: all codes x all months 2016-01 to 2024-11
all_dates = pd.date_range('2016-01-01', '2024-11-01', freq='MS')
spine = pd.MultiIndex.from_product(
    [sorted(common_codes), all_dates], names=['hs_code', 'date']
).to_frame(index=False)
spine['hs2'] = spine['hs_code'].str[:2]

panel = (spine
    .merge(mex_agg[['hs_code','date','fob_mex_us']],
           on=['hs_code','date'], how='left')
    .merge(chn_mex_agg[['hs_code','date','fob_chn_mex']],
           on=['hs_code','date'], how='left')
    .merge(chn_us_agg[['hs_code','date','fob_chn_us']],
           on=['hs_code','date'], how='left'))

# Fill structural zeros (non-trading pairs retained at zero for PPML)
for col in ['fob_mex_us', 'fob_chn_mex', 'fob_chn_us']:
    panel[col] = panel[col].fillna(0)

# Log outcomes (log1p handles structural zeros)
panel['log_mex_us']  = np.log1p(panel['fob_mex_us'])
panel['log_chn_mex'] = np.log1p(panel['fob_chn_mex'])

# Time FE identifier (integer YYYYMM for groupby)
panel['ym_fe'] = panel['date'].dt.strftime('%Y%m').astype(int)

print(f"Master panel: {len(panel):,} rows | "
      f"{panel['hs_code'].nunique():,} HS6 | "
      f"{panel['date'].nunique()} months")
print(f"Zero MEX->US:  {(panel['fob_mex_us']==0).mean():.1%}")
print(f"Zero CHN->MEX: {(panel['fob_chn_mex']==0).mean():.1%}")


# ── 3. TREATMENT VARIABLES ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TREATMENT VARIABLES")
print("=" * 60)

# -- Main treatment: pre-tariff China share of (CHN+MEX) US imports
pre_panel = panel[(panel['date'] >= PRE_START) & (panel['date'] <= PRE_END)]
pre_avg = pre_panel.groupby('hs_code').agg(
    chn_us_pre  = ('fob_chn_us',  'mean'),
    mex_us_pre  = ('fob_mex_us',  'mean'),
    chn_mex_pre = ('fob_chn_mex', 'mean'),
).reset_index()

pre_avg['total_proxy'] = pre_avg['chn_us_pre'] + pre_avg['mex_us_pre']
pre_avg['treat_share'] = np.where(
    pre_avg['total_proxy'] > 0,
    (pre_avg['chn_us_pre'] / pre_avg['total_proxy']).clip(0, 1),
    np.nan
)
print("TreatShare (pre-tariff China share of US imports):")
print(pre_avg['treat_share'].describe().round(3))

# -- Alternative treatment: realized delta_share 2017 -> 2019
y2017 = (panel[panel['date'].dt.year == 2017]
         .groupby('hs_code')
         .agg(chn_17=('fob_chn_us','mean'), mex_17=('fob_mex_us','mean'))
         .reset_index())
y2019 = (panel[panel['date'].dt.year == 2019]
         .groupby('hs_code')
         .agg(chn_19=('fob_chn_us','mean'), mex_19=('fob_mex_us','mean'))
         .reset_index())
alt = y2017.merge(y2019, on='hs_code', how='inner')
alt['share_17'] = np.where(alt['chn_17']+alt['mex_17'] > 0,
                            alt['chn_17']/(alt['chn_17']+alt['mex_17']), np.nan)
alt['share_19'] = np.where(alt['chn_19']+alt['mex_19'] > 0,
                            alt['chn_19']/(alt['chn_19']+alt['mex_19']), np.nan)
# Positive delta_share = China lost share = more displacement
alt['delta_share'] = alt['share_17'] - alt['share_19']
print(f"\ndelta_share (realized displacement, {(alt['delta_share']>0).sum()} codes positive):")
print(alt['delta_share'].describe().round(3))

# Merge both treatments into panel
panel = panel.merge(
    pre_avg[['hs_code','treat_share','chn_us_pre','mex_us_pre','chn_mex_pre']],
    on='hs_code', how='left')
panel = panel.merge(alt[['hs_code','delta_share']], on='hs_code', how='left')

panel['post']             = (panel['date'] >= POST_DATE).astype(int)
panel['treat_x_post']     = panel['treat_share']  * panel['post']
panel['alt_treat_x_post'] = panel['delta_share']   * panel['post']

# Drop codes with missing main treatment
panel = panel.dropna(subset=['treat_share']).copy()
print(f"\nClean panel: {len(panel):,} rows | {panel['hs_code'].nunique():,} HS6 codes")


# ── 4. CAPACITY SPLIT ──────────────────────────────────────────────────────────

med_cap = panel[panel['mex_us_pre'] > 0]['mex_us_pre'].median()
panel['high_cap'] = (panel['mex_us_pre'] >= med_cap).astype(int)
print(f"\nCapacity split at ${med_cap/1e6:.2f}M/month (median pre-tariff MEX->US)")
print(f"  High capacity: {panel[panel['high_cap']==1]['hs_code'].nunique()} HS6 codes")
print(f"  Low  capacity: {panel[panel['high_cap']==0]['hs_code'].nunique()} HS6 codes")


# ── 5. TWO-WAY FE DEMEANING ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TWO-WAY FE DEMEANING  (~30 seconds)")
print("=" * 60)

panel['y1_dm']  = demean_2way(panel, 'log_mex_us')
panel['y2_dm']  = demean_2way(panel, 'log_chn_mex')
panel['txp_dm'] = demean_2way(panel, 'treat_x_post')

print(f"Demeaned treatment std: {panel['txp_dm'].std():.4f}  (>0 = variation after FE)")


# ── 6. MAIN REGRESSIONS ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MAIN RESULTS")
print("Two-way FE OLS | SE clustered at HS2")
print("=" * 60)

cl = panel['hs2'].values
r1 = cluster_ols(panel['y1_dm'].values, panel['txp_dm'].values, cl)
r2 = cluster_ols(panel['y2_dm'].values, panel['txp_dm'].values, cl)

print("\nEq 1 -- MEX->US (nearshoring test):")
print_result("TreatShare x Post", r1)
print("\nEq 2 -- CHN->MEX (deflection test):")
print_result("TreatShare x Post", r2)

b1, b2 = r1['coef'], r2['coef']
s1, s2 = abs(r1['t']) > 1.96, abs(r2['t']) > 1.96
print("\nJoint interpretation:")
if   b1 > 0 and s1 and not s2:
    print("  -> NEARSHORING: MEX->US rises, CHN->MEX insignificant (no deflection)")
elif b1 > 0 and s1 and b2 < 0 and s2:
    print("  -> CLEAN NEARSHORING: MEX->US rises, CHN->MEX falls (supply shift)")
elif b1 > 0 and s1 and b2 > 0 and s2:
    print("  -> MIXED: MEX->US rises, CHN->MEX also rises (possible deflection)")
elif not s1 and b2 > 0 and s2:
    print("  -> DEFLECTION: CHN->MEX rises, no MEX->US response")
else:
    print("  -> INCONCLUSIVE -- check sector-level heterogeneity")


# ── 7. HETEROGENEITY BY CAPACITY ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("HETEROGENEITY -- Mexican pre-tariff productive capacity")
print("=" * 60)

het_results = {}
for label, val in [('High capacity', 1), ('Low capacity', 0)]:
    sub = panel[panel['high_cap'] == val].copy()
    sub['y1_s']  = demean_2way(sub, 'log_mex_us')
    sub['y2_s']  = demean_2way(sub, 'log_chn_mex')
    sub['txp_s'] = demean_2way(sub, 'treat_x_post')
    rm = cluster_ols(sub['y1_s'].values, sub['txp_s'].values, sub['hs2'].values)
    rc = cluster_ols(sub['y2_s'].values, sub['txp_s'].values, sub['hs2'].values)
    het_results[label] = {'mex_us': rm, 'chn_mex': rc}
    print(f"\n{label}  ({sub['hs_code'].nunique()} HS6, n={len(sub):,}):")
    print_result("MEX->US:", rm)
    print_result("CHN->MEX:", rc)


# ── 8. ROBUSTNESS CHECKS ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("ROBUSTNESS CHECKS")
print("=" * 60)

rob_results = []

# (1) Drop December months (removes Comtrade reporting artifacts)
panel_nd = panel[panel['date'].dt.month != 12].copy()
panel_nd['y1_nd']  = demean_2way(panel_nd, 'log_mex_us')
panel_nd['y2_nd']  = demean_2way(panel_nd, 'log_chn_mex')
panel_nd['txp_nd'] = demean_2way(panel_nd, 'treat_x_post')
r_nd1 = cluster_ols(panel_nd['y1_nd'].values, panel_nd['txp_nd'].values,
                    panel_nd['hs2'].values)
r_nd2 = cluster_ols(panel_nd['y2_nd'].values, panel_nd['txp_nd'].values,
                    panel_nd['hs2'].values)
print(f"\n(1) Drop December (n={len(panel_nd):,}):")
print_result("MEX->US:", r_nd1)
print_result("CHN->MEX:", r_nd2)
rob_results += [{'spec':'(1)_Baseline_NoDec', 'eq':'MEX_US',  **r_nd1},
                {'spec':'(1)_Baseline_NoDec', 'eq':'CHN_MEX', **r_nd2}]

# (2) Alternative treatment: realized delta_share 2017->2019
panel_alt = panel.dropna(subset=['delta_share']).copy()
panel_alt['y1_a']  = demean_2way(panel_alt, 'log_mex_us')
panel_alt['y2_a']  = demean_2way(panel_alt, 'log_chn_mex')
panel_alt['atp_a'] = demean_2way(panel_alt, 'alt_treat_x_post')
r_alt1 = cluster_ols(panel_alt['y1_a'].values, panel_alt['atp_a'].values,
                     panel_alt['hs2'].values)
r_alt2 = cluster_ols(panel_alt['y2_a'].values, panel_alt['atp_a'].values,
                     panel_alt['hs2'].values)
print(f"\n(2) Alt. treatment -- realized delta_share 2017->2019 (n={len(panel_alt):,}):")
print_result("MEX->US:", r_alt1)
print_result("CHN->MEX:", r_alt2)
rob_results += [{'spec':'(2)_AltTreat_DeltaShare', 'eq':'MEX_US',  **r_alt1},
                {'spec':'(2)_AltTreat_DeltaShare', 'eq':'CHN_MEX', **r_alt2}]

# (3) Drop HS87 automotive (USMCA rules of origin may confound identification)
panel_no87 = panel[panel['hs2'].astype(int) != 87].copy()
panel_no87['y1_87']  = demean_2way(panel_no87, 'log_mex_us')
panel_no87['y2_87']  = demean_2way(panel_no87, 'log_chn_mex')
panel_no87['txp_87'] = demean_2way(panel_no87, 'treat_x_post')
r_871 = cluster_ols(panel_no87['y1_87'].values, panel_no87['txp_87'].values,
                    panel_no87['hs2'].values)
r_872 = cluster_ols(panel_no87['y2_87'].values, panel_no87['txp_87'].values,
                    panel_no87['hs2'].values)
print(f"\n(3) Drop HS87 automotive (n={len(panel_no87):,}):")
print_result("MEX->US:", r_871)
print_result("CHN->MEX:", r_872)
rob_results += [{'spec':'(3)_Drop_HS87', 'eq':'MEX_US',  **r_871},
                {'spec':'(3)_Drop_HS87', 'eq':'CHN_MEX', **r_872}]


# ── 9. EVENT STUDY ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EVENT STUDY -- Pre-trends (December excluded, both outcomes)")
print("=" * 60)

ref_date  = pd.Timestamp('2018-07-01')
panel_es  = panel[panel['date'].dt.month != 12].copy()
panel_es['y1_es'] = demean_2way(panel_es, 'log_mex_us')
panel_es['y2_es'] = demean_2way(panel_es, 'log_chn_mex')
panel_es['rel_month'] = ((panel_es['date'].dt.year  - ref_date.year)  * 12 +
                          (panel_es['date'].dt.month - ref_date.month))

es_res = []
for outcome, col in [('MEX_US', 'y1_es'), ('CHN_MEX', 'y2_es')]:
    for rm in sorted(panel_es['rel_month'].unique()):
        if not (-24 <= rm <= 30):
            continue
        sub = panel_es[panel_es['rel_month'] == rm]
        y   = sub[col].values
        x   = sub['treat_share'].values
        valid = ~(np.isnan(y) | np.isnan(x))
        y, x  = y[valid], x[valid]
        if len(y) < 50 or x.std() < 1e-8:
            continue
        x_c = x - x.mean()
        b   = np.dot(x_c, y) / np.dot(x_c, x_c)
        se  = (np.sqrt(np.sum((y - b*x_c)**2) / (len(y)-2))
               / (x_c.std() * np.sqrt(len(y))))
        es_res.append({'outcome': outcome, 'rel_month': int(rm),
                       'coef': round(b, 4), 'se': round(se, 4)})

es_df = pd.DataFrame(es_res)

# Normalize each outcome to pre-period mean = 0
for outcome in ['MEX_US', 'CHN_MEX']:
    mask     = es_df['outcome'] == outcome
    pre_mean = es_df[mask & (es_df['rel_month'] < 0)]['coef'].mean()
    es_df.loc[mask, 'coef_norm'] = (es_df.loc[mask, 'coef'] - pre_mean).round(4)
    es_df.loc[mask, 'ci_lo'] = (es_df.loc[mask, 'coef_norm']
                                - 1.96 * es_df.loc[mask, 'se']).round(4)
    es_df.loc[mask, 'ci_hi'] = (es_df.loc[mask, 'coef_norm']
                                + 1.96 * es_df.loc[mask, 'se']).round(4)

for outcome in ['MEX_US', 'CHN_MEX']:
    sub  = es_df[es_df['outcome'] == outcome]
    pre  = sub[sub['rel_month'] < 0]['coef_norm'].dropna()
    post = sub[sub['rel_month'] >= 0]['coef_norm'].dropna()
    flag = 'flat (pre-trends OK)' if pre.std() < 0.8 else 'noisy -- check'
    print(f"\n{outcome}:")
    print(f"  Pre  mean={pre.mean():.4f}  std={pre.std():.4f}  {flag}")
    print(f"  Post mean={post.mean():.4f}  std={post.std():.4f}")


# ── 10. SAVE OUTPUTS ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

os.makedirs(OUT_DIR, exist_ok=True)

# Main results (full + heterogeneity)
results_main = pd.DataFrame([
    {'spec':'OLS_full',    'equation':'MEX_US',  **r1},
    {'spec':'OLS_full',    'equation':'CHN_MEX', **r2},
    {'spec':'OLS_highcap', 'equation':'MEX_US',  **het_results['High capacity']['mex_us']},
    {'spec':'OLS_highcap', 'equation':'CHN_MEX', **het_results['High capacity']['chn_mex']},
    {'spec':'OLS_lowcap',  'equation':'MEX_US',  **het_results['Low capacity']['mex_us']},
    {'spec':'OLS_lowcap',  'equation':'CHN_MEX', **het_results['Low capacity']['chn_mex']},
])
results_main.to_csv(f"{OUT_DIR}/results_main.csv", index=False)

# Robustness
pd.DataFrame(rob_results).to_csv(f"{OUT_DIR}/results_robustness_full.csv", index=False)

# Event study (normalized, both outcomes)
es_df.to_csv(f"{OUT_DIR}/event_study.csv", index=False)

# Analysis-ready panel (for PPML in Stata or R via fixest/ppmlhdfe)
panel_save = panel[[
    'hs_code','hs2','date','ym_fe','post',
    'treat_share','treat_x_post',
    'delta_share','alt_treat_x_post',
    'fob_mex_us','fob_chn_mex','fob_chn_us',
    'log_mex_us','log_chn_mex',
    'mex_us_pre','chn_us_pre','chn_mex_pre','high_cap'
]].copy()
panel_save.to_csv(f"{OUT_DIR}/panel_final.csv", index=False)

print(f"  results_main.csv            -- main + heterogeneity coefficients")
print(f"  results_robustness_full.csv -- robustness checks (1)-(3)")
print(f"  event_study.csv             -- pre-trends, both outcomes, normalized")
print(f"  panel_final.csv             -- analysis-ready panel for PPML")

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(panel[['fob_mex_us','fob_chn_mex','fob_chn_us','treat_share']].describe().round(1))
print(f"\nMedian non-zero monthly HS6 values:")
for col, label in [('fob_mex_us','MEX->US'),
                   ('fob_chn_mex','CHN->MEX'),
                   ('fob_chn_us','CHN->US')]:
    m = panel[col].replace(0, np.nan).median()
    print(f"  {label}: ${m/1e3:.0f}K/month")

print("\nDone.")

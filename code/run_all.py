"""
run_all.py
==========
Master script — runs the full replication pipeline in order.

Usage:
  python run_all.py --data path/to/data --output path/to/tables --figures path/to/figures

Defaults (relative to repo root):
  --data    data/
  --output  tables/
  --figures figures/
"""

import argparse
import subprocess
import sys
import os

parser = argparse.ArgumentParser(description='Run full replication pipeline.')
parser.add_argument('--data',    default='data',    help='input data folder')
parser.add_argument('--output',  default='tables',  help='output folder for CSVs/tables')
parser.add_argument('--figures', default='figures', help='output folder for figures')
args = parser.parse_args()

# Resolve script paths relative to this file's location
ROOT = os.path.dirname(os.path.abspath(__file__))

steps = [
    {
        'script':  os.path.join(ROOT, 'code', 'trade_model_final.py'),
        'args':    ['--data', args.data, '--output', args.output],
        'desc':    'Step 1/3 — Build panel and run DiD regressions',
    },
    {
        'script':  os.path.join(ROOT, 'code', 'scatter_routes.py'),
        'args':    ['--data', args.data, '--output', args.figures],
        'desc':    'Step 2/3 — Generate Figure 1 (scatter routes)',
    },
    {
        'script':  os.path.join(ROOT, 'code', 'generate_figures.py'),
        'args':    ['--data', args.data, '--output', args.figures],
        'desc':    'Step 3/3 — Generate Figures 2, 3, 4',
    },
]

print("=" * 60)
print("REPLICATION PIPELINE")
print("=" * 60)
print(f"  data:    {args.data}")
print(f"  tables:  {args.output}")
print(f"  figures: {args.figures}")
print()

for step in steps:
    print(f"\n{step['desc']}")
    print("-" * 60)
    cmd = [sys.executable, step['script']] + step['args']
    result = subprocess.run(cmd, check=True)

print("\n" + "=" * 60)
print("All steps completed successfully.")
print(f"Tables saved to:  {args.output}/")
print(f"Figures saved to: {args.figures}/")
print("=" * 60)

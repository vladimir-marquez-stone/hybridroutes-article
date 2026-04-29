# Competition and Dependence: Mexico, China, and the New Export Routes

## Data

Raw data is not included. Download from UN Comtrade Plus
(https://comtradeplus.un.org/) with these parameters:
- Reporters: Mexico, China
- Partners: USA, Mexico
- Classification: HS, monthly, 2016–2024
- Save to: data/

File naming convention:
  mex_usa_exports_{year}.csv
  chn_mex_exports_{year}.csv
  chn_usa_exports_{year}.csv

## Replication

Install dependencies:
  pip install -r requirements.txt

Generate figures:
  python code/generate_figures.py

Run DiD estimation:
  python code/did_estimation.py

Compile paper:
  cd paper && pdflatex competencia_dependencia && bibtex competencia_dependencia
  pdflatex competencia_dependencia && pdflatex competencia_dependencia
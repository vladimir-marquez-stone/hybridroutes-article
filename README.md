[![DOI](https://zenodo.org/badge/1224043927.svg)](https://doi.org/10.5281/zenodo.19935466)

**Vladimir Márquez Stone & Seyka Sandoval**
Facultad de Economía, UNAM

Working paper available on SSRN: http://ssrn.com/abstract=6685999

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

Run the full pipeline with one command:

    python run_all.py --data data/ --output tables/ --figures figures/

Or step by step:

    python codes/trade_model_final.py   # builds panel, runs DiD, saves tables
    python codes/scatter_routes.py      # Figure 1
    python codes/generate_figures.py    # Figures 2-4

Compile paper:

    cd paper
    pdflatex competencia_dependencia
    bibtex competencia_dependencia
    pdflatex competencia_dependencia
    pdflatex competencia_dependencia

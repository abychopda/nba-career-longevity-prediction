# 36120-26SP-group36-experiments

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

AAdv ML AT1 repo for group 36

## Getting Started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.9.

```bash
uv sync
```

`nba-career-toolkit` resolves from TestPyPI automatically; pandas, scikit-learn and the rest
resolve from PyPI as normal. No extra flags are needed for `uv sync` — a plain `pip install
-r requirements.txt` will not find the toolkit, since only `uv sync` reads the index
configuration in `pyproject.toml`.

**Before running any notebook**, add the Kaggle competition's raw files to `data/raw/`:
`train.csv`, `test.csv`, `sample_submission.csv`, `metadata.csv`. This directory is
gitignored and not part of the repository.

```bash
uv run jupyter lab
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         nba_career_predictor and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── nba_career_predictor   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes nba_career_predictor a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


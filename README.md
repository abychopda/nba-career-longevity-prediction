# NBA Career Prediction

## Project Overview

This project investigates whether college-level basketball statistics can be used to predict whether a basketball player will go on to have a successful NBA career. A successful NBA career is defined as playing at least three NBA seasons.

The repository contains the group's experiments, trained models, prediction outputs, notebooks, reports and supporting documentation.

## Repository Structure

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
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
│                         36120_26sp_group36_experiments and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
│
│
└── nba_predictions   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes 36120_26sp_group36_experiments a Python module
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

## Setup

The project uses Python and the dependencies specified in `pyproject.toml` and `uv.lock`.

Install the project dependencies before running the notebooks.

## Running the Experiments

The notebooks in `notebooks/` contain the data processing, model training, evaluation and prediction workflows.

Run the notebooks in the appropriate order to reproduce the experiments and results.

## Models and Predictions

Trained models and associated model artefacts are stored in `models/`.

NBA prediction outputs are stored in `nba_predictions/`.

## Results

Experiment results, reports and generated figures are stored in `reports/`.

--------


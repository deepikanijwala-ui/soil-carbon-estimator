# Soil Carbon Estimator

Predicts soil organic carbon (SOC) from satellite-derived indices using machine learning.
Mirrors remote sensing and MRV workflows used in climate tech companies like Klim and Seqana.

## What This Project Does

Estimates SOC percentage across agricultural landscapes using three satellite indices:
- **NDVI** — Normalized Difference Vegetation Index
- **EVI** — Enhanced Vegetation Index  
- **BSI** — Bare Soil Index

A Random Forest regression model is trained on field samples, then applied spatially
to produce carbon maps and a summary MRV report.

## Key Results

| Metric | Value |
|--------|-------|
| Model MAE | 0.178% SOC |
| Model R² | 0.843 |
| Top predictive index | NDVI (62% importance) |
| Regenerative vs conventional carbon benefit | +8.46 t/ha |

## Project Structure

```
soil-carbon-estimator/
├── generate_data.py       # Synthetic satellite + SOC dataset (500 samples)
├── carbon_model.py        # Random Forest regression + feature importance
├── spatial_analysis.py    # Spatial SOC prediction across 30x30 km grid
├── carbon_report.py       # MRV-style summary report with carbon stock estimates
├── data/                  # Generated dataset
└── outputs/               # Charts and report outputs
```

## Scripts

**`generate_data.py`**  
Generates 500 synthetic soil samples with NDVI, EVI, BSI indices and SOC labels.
Includes four land use types: conventional, regenerative, degraded, fallow.

**`carbon_model.py`**  
Trains a Random Forest regressor to predict SOC from satellite indices.
Outputs prediction accuracy metrics and feature importance analysis.

**`spatial_analysis.py`**  
Creates a 30x30 spatial grid simulating satellite coverage.
Predicts SOC at every pixel and visualises as a carbon map.

**`carbon_report.py`**  
Aggregates all results into an MRV-style report.
Calculates SOC stocks in tonnes/ha and quantifies the regenerative farming carbon benefit.

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib
```

Run scripts in order:

```bash
python generate_data.py
python carbon_model.py
python spatial_analysis.py
python carbon_report.py
```

## Background

Built as part of a portfolio targeting environmental data science roles in climate tech.
Concepts applied: remote sensing indices, regression ML, spatial analysis, carbon accounting (MRV).

## Author

Deepika Nijwala  
Environmental & Agricultural Scientist  
github.com/deepikanijwala-ui
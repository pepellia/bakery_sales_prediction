# Scripts Overview

Last updated: December 6, 2023

## Data Preparation Scripts

### Core Data Processing
- `data_preparation.py` (03:54 PM) - Main script for data preprocessing and feature engineering
- `config.py` (04:06 PM) - Configuration file containing constants and shared functions
- `data_preparation.ipynb` (01:18 PM) - Jupyter notebook version with exploratory analysis

### Analysis Scripts
- `analyse_umsatz.py` (Nov 25) - Initial sales data analysis
- `analyse_correlations.py` (04:11 PM) - Analysis of correlations between product groups
- `analyse_windjammer.py` (Nov 25) - Analysis of Windjammer event impact
- `create_windjammer_csv.py` (Nov 25) - Script to create Windjammer events dataset

### Machine Learning Scripts
- `linear_regression_umsatz.py` (04:12 PM) - Basic linear regression model
- `linear_regression_umsatz_v2.py` (Dec 8) - Enhanced version with additional features
- `linear_regression_umsatz.ipynb` (01:18 PM) - Interactive model development notebook

### Feature Engineering
- `feature_engineering_todos.md` (04:07 PM) - Documentation of planned feature improvements

### Input Data Files
- `umsatzdaten_gekuerzt.csv` (Nov 21) - Main sales data
- `wetter.csv` (Nov 21) - Weather data
- `kiwo.csv` (Nov 21) - Kieler Woche event data
- `windjammer.csv` (12:56 PM) - Windjammer event data
- `Feiertage-SH.csv` (Nov 28) - Public holidays in Schleswig-Holstein
- `Schulferientage-SH.csv` (Nov 28) - School holidays in Schleswig-Holstein

### Generated Visualizations
- `warengruppen_correlations.png` (04:10 PM) - Product group correlation heatmap
- `feature_importance.png` (04:10 PM) - Feature importance visualization
- `prediction_vs_actual.png` (04:10 PM) - Model prediction comparison
- `umsatz_trend.png` (Nov 28) - Sales trend analysis
- `umsatz_by_group.png` (Nov 28) - Sales by product group
- `umsatz_by_weekday.png` (Nov 28) - Sales patterns by weekday
- `umsatz_prediction.png` (Nov 28) - Prediction analysis visualization

### Example and Documentation Files
- `umsatzdaten_gekuerzt_example.csv` (Dec 3) - Example data structure
- `INSTRUCTIONS.md` (Nov 21) - Project instructions
- `README.md` (Nov 21) - Project overview

## Directory Structure
- `visualizations/` - Directory containing additional plots and figures
- `__pycache__/` - Python cache directory (not version controlled)
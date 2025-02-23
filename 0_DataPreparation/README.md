# Data Preparation

This directory contains all data preparation and feature engineering code for the bakery sales prediction project.

## Directory Structure

```
0_DataPreparation/
├── input/
│   ├── competition_data/     # Original competition files
│   │   ├── train.csv        # Training data (2013-2017)
│   │   ├── test.csv         # Test data (2018)
│   │   ├── sample_submission.csv
│   │   ├── wetter.csv       # Weather data
│   │   └── kiwo.csv         # Kieler Woche events
│   │
│   └── compiled_data/       # Additional data compiled by team
│       ├── Feiertage-SH.csv           # Public holidays
│       ├── Schulferientage-SH.csv     # School holidays
│       ├── easter_saturday.csv        # Easter Saturday dates
│       ├── wettercode.csv             # Weather code mappings
│       ├── windjammer.csv             # Windjammer events
│       └── school_holidays/           # School holidays by state
│           └── *.csv                  # Individual state files
│
├── output/
│   ├── correlation_analysis/    # Correlation analysis results
│   ├── sales_analysis/         # Sales analysis results
│   └── windjammer_analysis/    # Windjammer impact analysis
│
└── visualizations/
    ├── correlations/           # Correlation analysis plots
    ├── feature_selection/      # Feature importance plots
    ├── linear_regression/      # Basic model results
    ├── linear_regression_v2/   # Enhanced model results
    └── sales_analysis/         # Sales patterns visualization
```

## Python Scripts

### Core Scripts
- `config.py`: Central configuration file with paths and constants
- `data_preparation.py` & `.ipynb`: Main data processing and feature engineering pipeline

### Data Generation & Processing
- `create_windjammer_csv.py` & `.ipynb`: Creates the Windjammer events dataset
- `generate_calendar_files.py`: Generates calendar-based feature files
- `process_school_holidays.py`: Processes school holiday data

### Analysis Scripts
- `analyse_correlations.py`: Analyzes correlations between features and sales
- `analyse_umsatz.py` & `.ipynb`: Sales data analysis and visualization
- `analyse_windjammer.py` & `.ipynb`: Analysis of Windjammer events impact
- `linear_regression_umsatz.py` & `.ipynb`: Initial linear regression analysis
- `linear_regression_umsatz_v2.py`: Enhanced version of linear regression analysis

### Documentation
- `INSTRUCTIONS.md`: Setup and execution instructions
- `feature_engineering_todos.md`: Feature engineering tasks and progress
- `scripts_overview.md`: Detailed script documentation

## Usage

1. Ensure all input files are in place:
   - Competition data in `input/competition_data/`
   - Additional data in `input/compiled_data/`
2. Run data preparation scripts in sequence:
   - First: `process_school_holidays.py` and `create_windjammer_csv.py`
   - Then: `data_preparation.py` for main feature engineering
   - Optional: Run analysis scripts for insights

## Notes

- All scripts use configurations from `config.py`
- Generated features are saved in `output/` directory
- Analysis results and visualizations are stored in their respective directories

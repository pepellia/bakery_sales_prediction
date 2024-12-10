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
│       ├── Feiertage-SH.csv      # Public holidays
│       ├── Schulferientage-SH.csv # School holidays
│       └── windjammer.csv         # Windjammer events
│
├── output/
│   └── features/            # Generated feature files
│
└── visualizations/
    ├── correlations/        # Correlation analysis plots
    └── features/            # Feature analysis plots
```

## Data Files

### Competition Data
- `train.csv`: Training dataset with daily sales from July 2013 to July 2017
- `test.csv`: Test dataset for predictions from August 2017 to July 2018
- `sample_submission.csv`: Template for competition submissions
- `wetter.csv`: Daily weather data including temperature, cloud cover, etc.
- `kiwo.csv`: Kieler Woche event dates and information

### Team-Compiled Data
- `Feiertage-SH.csv`: Public holidays in Schleswig-Holstein
- `Schulferientage-SH.csv`: School holiday periods in Schleswig-Holstein
- `windjammer.csv`: Windjammer parade events and dates

## Scripts
- `config.py`: Central configuration file with paths and constants
- `data_preparation.py`: Main data processing and feature engineering
- Additional analysis and visualization scripts

## Usage
1. Place competition data files in `input/competition_data/`
2. Place or update team-compiled data in `input/compiled_data/`
3. Run data preparation scripts
4. Generated features will be saved in `output/features/`
5. Visualizations will be saved in respective subdirectories

## Notes
- All paths are managed centrally in `config.py`
- Use relative imports from config for consistency
- Visualizations are saved with timestamps for version tracking

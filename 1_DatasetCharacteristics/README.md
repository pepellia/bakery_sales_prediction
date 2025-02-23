# Dataset Characteristics

This directory contains scripts and visualizations for analyzing the characteristics of our bakery sales dataset. The analysis focuses on feature selection, temporal patterns through Fourier analysis, and the impact of special events on sales.

## Directory Structure

```
1_DatasetCharacteristics/
├── Python Scripts
│   ├── Feature Selection
│   │   ├── feature_selection.py & .ipynb   # Base feature analysis
│   │   ├── feature_selection_v2.py         # Enhanced feature importance
│   │   ├── feature_selection_v3.py         # Advanced correlation analysis
│   │   └── feature_selection_v4.py         # Final feature optimization
│   │
│   ├── Fourier Analysis
│   │   ├── fourier_analysis.py            # Basic frequency analysis
│   │   └── fourier_analysis_v2.py         # Enhanced temporal patterns
│   │
│   └── special_events_barplot.py          # Event impact visualization
│
└── visualizations/                         # Generated visualizations
    ├── correlation_matrix.png             # Feature correlations
    ├── univariate_feature_scores.png      # Feature importance scores
    └── Distribution Plots
        ├── Weather Features
        │   ├── dist_Bewoelkung.png        # Cloud cover
        │   ├── dist_Temperatur.png        # Temperature
        │   ├── dist_Wettercode.png        # Weather code
        │   └── dist_Windgeschwindigkeit.png # Wind speed
        │
        ├── Temporal Features
        │   ├── dist_Jahr.png              # Year
        │   ├── dist_Monat.png             # Month
        │   ├── dist_Tag_im_Monat.png      # Day of month
        │   ├── dist_Woche_im_Jahr.png     # Week of year
        │   └── dist_Wochentag.png         # Day of week
        │
        └── Event Features
            ├── dist_is_kiwo.png           # Kieler Woche
            ├── dist_is_weekend.png        # Weekend indicator
            └── dist_is_windjammer.png     # Windjammer events
```

## Scripts

### Feature Selection Evolution
- `feature_selection.py` & `.ipynb`: Initial analysis of feature relationships and importance
- `feature_selection_v2.py`: Enhanced feature importance scoring and selection
- `feature_selection_v3.py`: Advanced correlation analysis and feature interactions
- `feature_selection_v4.py`: Final optimization with refined selection criteria

### Time Series Analysis
- `fourier_analysis.py`: Basic frequency domain analysis of sales patterns
- `fourier_analysis_v2.py`: Enhanced temporal pattern detection and seasonality analysis

### Event Analysis
- `special_events_barplot.py`: Visualization and analysis of special events' impact on sales

## Documentation
- `INSTRUCTIONS.md`: Detailed instructions for running the analysis scripts

## Visualizations

The `visualizations/` directory contains various plots and charts that help understand:
1. Feature correlations and importance scores
2. Distribution of weather-related features
3. Temporal patterns in the data
4. Impact of special events and holidays

## Usage

1. Start with feature selection scripts to understand key predictors
2. Run Fourier analysis to identify temporal patterns
3. Use special events analysis to quantify event impacts
4. Review visualizations in the respective directories

## Notes

- Scripts are designed to be run sequentially, building upon previous analyses
- All visualizations are automatically saved to the `visualizations/` directory
- Feature selection results inform the modeling approaches in subsequent project phases

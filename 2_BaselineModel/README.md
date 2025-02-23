# Baseline Model for Bakery Sales Prediction

This directory contains the implementation of baseline models for bakery sales prediction. The models primarily utilize weekday patterns and basic statistical methods to establish fundamental sales forecasting benchmarks.

## Directory Structure

```
2_BaselineModel/
├── analysis/
│   ├── results/
│   │   └── linear_regression/
│   │       └── simple_weekday_model/
│   │           └── special_events_analysis.txt
│   │
│   └── scripts/
│       └── linear_regression/
│           └── analyze_submission.py
│
├── Core Models
│   ├── simple_weekday_model.py           # Basic weekday-based model
│   ├── simple_weekday_model_group1.py    # Specialized for product group 1
│   └── weekday_model_by_product.py       # Product-specific predictions
│
└── Evaluation
    ├── analyze_submission.py             # Results analysis
    ├── evaluate_simple_model.py & .ipynb # Model evaluation
    └── model_training.py & .ipynb        # Model training pipeline
```

## Scripts Overview

### Core Model Implementations
- `simple_weekday_model.py`: Basic model using average sales by weekday
- `simple_weekday_model_group1.py`: Specialized version optimized for product group 1
- `weekday_model_by_product.py`: Enhanced version with product-specific patterns

### Training and Evaluation
- `model_training.py` & `.ipynb`: Main training pipeline
  - Data preprocessing
  - Model training
  - Model persistence
  
- `evaluate_simple_model.py` & `.ipynb`: Comprehensive evaluation suite
  - Performance metrics calculation
  - Model validation
  - Results visualization

### Analysis Tools
- `analyze_submission.py`: Prediction analysis tool
  - Detailed performance metrics
  - Error analysis
  - Special events impact assessment

## Documentation
- `INSTRUCTIONS.md`: Detailed setup and execution instructions

## Usage

1. Model Training:
   ```bash
   python model_training.py
   ```
   - Creates baseline models for each product category
   - Saves trained models in designated output directory

2. Generate Predictions:
   ```bash
   python weekday_model_by_product.py  # For product-specific predictions
   # or
   python simple_weekday_model.py      # For basic weekday-based predictions
   ```

3. Evaluate Results:
   ```bash
   python evaluate_simple_model.py
   python analyze_submission.py
   ```

## Key Features

- Weekday-based sales pattern analysis
- Product-specific modeling capabilities
- Special events consideration
- Comprehensive evaluation metrics
- Interactive notebooks for analysis and visualization

## Notes

- Models primarily rely on weekday patterns and basic statistical features
- Different model variants available for specific use cases
- Evaluation scripts provide detailed performance analysis
- Results and analysis are stored in the `analysis/results/` directory

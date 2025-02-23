# Advanced Models for Bakery Sales Prediction

This directory contains the implementation of various advanced modeling approaches for bakery sales prediction, including Linear Regression, Neural Networks, Fourier Analysis, and Hybrid models.

## Directory Structure

```
3_Model/
├── analysis/
│   ├── results/
│   │   ├── linear_regression/
│   │   │   └── [v7-v11]/
│   │   │       └── special_events_analysis.txt
│   │   │
│   │   └── neural_network/
│   │       └── [v12-v14]/
│   │           ├── easter_saturday_analysis.txt
│   │           └── summary_statistics.txt
│   │
│   └── scripts/
│       ├── data_quality/
│       │   └── analyze_missing_data.py
│       │
│       ├── fourier_analysis/
│       │   └── analyze_fourier_analysis.py
│       │
│       ├── linear_regression/
│       │   ├── analyze_submission.py
│       │   └── create_interactive_timeline.py
│       │
│       └── neural_network/
│           └── analyze_neural_network.py
│
└── models/
    ├── fourier_analysis/
    │   └── fourier_analysis.py
    │
    ├── hybrid/
    │   └── hybrid_model_v1.py
    │
    ├── linear_regression/
    │   ├── Evolution Models
    │   │   ├── linear_regression_[v6-v12].py
    │   │   └── weekday_and_product_model_*.py
    │   │
    │   └── Base Models
    │       └── weekday_and_date_by_product.py
    │
    └── neural_network/
        └── neural_network_model_[v1-v16].py
```

## Model Overview

### Neural Network Evolution (v1-v16)
- **v1-v5**: Basic architectures with progressive feature integration
- **v6-v10**: Enhanced architectures with advanced feature engineering
- **v11-v13**: Improved event handling and temporal features
- **v14**: Best performing model with comprehensive feature set
  - Weather data integration
  - Special events (Kieler Woche, Easter, Windjammer)
  - Advanced temporal features
- **v15-v16**: Experimental architectures and feature combinations

### Linear Regression Series (v6-v12)
- Progressive enhancement of linear models
- Feature importance analysis
- Special event impact modeling
- Product-specific regression models

### Specialized Models
- **Fourier Analysis**: Time series decomposition and frequency analysis
- **Hybrid Model**: Combining multiple modeling approaches
- **Weekday and Product Models**: Specialized product category predictions

## Analysis Tools

### Data Quality
- Missing data analysis
- Feature distribution checks
- Data integrity validation

### Model Analysis
- Submission analysis for competition metrics
- Interactive timeline visualization
- Neural network performance analysis
- Fourier analysis visualization

### Results Analysis
- Special events impact assessment
- Easter Saturday analysis
- Model performance statistics
- Comparative analysis across model versions

## Documentation
- `INSTRUCTIONS.md`: Detailed setup and execution guide

## Key Features

### Data Integration
- Weather data (temperature, cloud cover, wind speed)
- Special events (Kieler Woche, Easter, Windjammer)
- Calendar features (holidays, weekends)
- Product categories

### Model Capabilities
- Multi-output regression
- Time series forecasting
- Event-based sales prediction
- Product-specific modeling

### Analysis Capabilities
- Performance metrics calculation
- Feature importance analysis
- Event impact assessment
- Error analysis and visualization

## Usage

1. Select Model Version:
   ```python
   # For best performance
   from models.neural_network.neural_network_model_v14 import NeuralNetworkModel
   # For specific product categories
   from models.linear_regression.weekday_and_product_model import LinearRegressionModel
   ```

2. Run Analysis:
   ```python
   # Analyze model results
   python analysis/scripts/neural_network/analyze_neural_network.py
   # Generate timeline visualization
   python analysis/scripts/linear_regression/create_interactive_timeline.py
   ```

## Notes

- Neural Network v14 is the current best-performing model
- Analysis results are stored in `analysis/results/`
- Each model version includes specific improvements and feature additions
- Documentation for each model version is available in their respective files

## Neural Network Model for Bakery Sales Prediction

### Version 14 (Current Best)
- Enhanced deep neural network with comprehensive feature integration
- Features:
  - Date-based: weekday, month, day of month, week of year
  - Product categories (6 groups)
  - Weather data: temperature, cloud cover, wind speed, weather code
  - Event data:
    - Kieler Woche festival
    - Windjammer Parade
    - Public Holidays (Schleswig-Holstein)
    - School Holidays
    - Easter Saturday
- Architecture:
  - Input layer with batch normalization
  - Dense layers (128 → 64 → 32 → 16 → 1)
  - Dropout layers (0.3, 0.3, 0.2) for regularization
  - ReLU activation for hidden layers
  - MSE loss with Adam optimizer

### Key Evolution
1. V1-V5: Basic architectures and initial feature integration
2. V6-V10: Enhanced architectures and feature engineering
3. V11-V13: Improved event handling and temporal features
4. V14: Comprehensive integration of all event types and weather data
5. V15-V16: Experimental architectures (post-optimization)

## Performance
- Training: July 2013 to July 2017
- Validation: August 2017 to July 2018
- Early stopping with patience monitoring
- Learning rate reduction on plateau
- Metrics:
  - MSE (primary loss function)
  - MAE (monitoring metric)
  - MAPE (per product group analysis)

## Feature Engineering
- One-hot encoding for categorical features:
  - Weekdays
  - Product groups
  - Event types
- Standardization of numerical features:
  - Weather measurements
  - Temporal features
- Missing value handling:
  - Mean imputation for weather data
  - Forward fill for sequential data
- Binary indicators for special events:
  - Holidays
  - Festivals
  - School breaks

## Key Insights
- Weather conditions significantly impact sales patterns
- Special events show varying degrees of sales uplift:
  - Kieler Woche festival: highest impact
  - Windjammer events: moderate impact
  - Public holidays: category-specific effects
- Product groups exhibit distinct temporal patterns:
  - Weekday preferences
  - Seasonal variations
  - Holiday effects
- Model successfully captures:
  - Long-term seasonal trends
  - Event-based variations
  - Weather-dependent fluctuations

## Tools & Libraries
- TensorFlow/Keras for neural network implementation
- Pandas for data processing and feature engineering
- Scikit-learn for preprocessing and validation
- Matplotlib/Seaborn for analysis and visualization
- NumPy for numerical operations
# Neural Network Model for Bakery Sales Prediction

## Model Evolution

### Version 3 (Final)
- Deep neural network with batch normalization and dropout
- Features:
  - Date-based: weekday, month, day of month
  - Product categories (6 groups)
  - Weather data: temperature, cloud cover, wind speed, weather code
  - Event data: Kieler Woche festival dates
- Architecture:
  - Input layer with batch normalization
  - Dense layers (64 → 32 → 16 → 1)
  - Dropout layers (0.3, 0.2) for regularization
  - ReLU activation for hidden layers
  - MSE loss with Adam optimizer

### Key Improvements
1. V1: Base model with date and product features
2. V2: Added weather data (temperature, clouds, wind)
3. V3: Incorporated Kieler Woche festival dates

## Performance
- Training: July 2013 to July 2017
- Validation: August 2017 to July 2018
- Early stopping with patience=15
- Learning rate reduction on plateau
- Final validation MAE: ~41€

## Feature Engineering
- One-hot encoded weekdays and product groups
- Standardized numerical features
- Missing values handled with mean imputation
- Binary indicator for Kieler Woche festival days

## Key Insights
- Weather significantly impacts sales patterns
- Kieler Woche festival shows notable sales uplift
- Product groups show distinct weekday patterns
- Model captures seasonal and event-based variations

## Tools & Libraries
- TensorFlow/Keras for neural network
- Pandas for data processing
- Scikit-learn for preprocessing
- Matplotlib/Seaborn for analysis
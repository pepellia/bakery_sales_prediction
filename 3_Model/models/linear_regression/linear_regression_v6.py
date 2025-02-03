"""
Linear Regression Model with Weekday and Product Features (with validation)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 3_Model directory

# Data paths
DATA_DIR = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'competition_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# Product group configurations
WARENGRUPPEN = {
    1: 'Brot',
    2: 'Brötchen',
    3: 'Croissant',
    4: 'Konditorei',
    5: 'Kuchen',
    6: 'Saisonbrot'
}

def get_warengruppe_name(code):
    """Get the name of a product group by its code"""
    return WARENGRUPPEN.get(code, f'Unknown ({code})')

# Create output directories
MODEL_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(MODEL_ROOT, 'output', 'linear_regression', MODEL_NAME)
VIZ_DIR = os.path.join(MODEL_ROOT, 'visualizations', 'linear_regression', MODEL_NAME)

for directory in [OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_and_prepare_data():
    """Load and prepare the data with enhanced date features"""
    # Load training data
    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_PATH)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    
    # Load weather data
    weather_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    weather_codes_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'wettercode.csv')
    
    weather_data = pd.read_csv(weather_path)
    weather_codes = pd.read_csv(weather_codes_path, sep=';', header=None, names=['code', 'description'])
    weather_data['Datum'] = pd.to_datetime(weather_data['Datum'])
    
    # Merge weather data with codes
    weather_data = pd.merge(weather_data, weather_codes[['code', 'description']], 
                           left_on='Wettercode', right_on='code', how='left')
    
    # Create weather features
    weather_dummies = pd.get_dummies(weather_data['description'], prefix='weather')
    weather_features = pd.concat([
        weather_data[['Datum', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']],
        weather_dummies
    ], axis=1)
    
    # Calculate feels_like_temperature
    weather_features['feels_like_temperature'] = weather_features['Temperatur'] - (0.2 * weather_features['Windgeschwindigkeit'])
    
    # Create is_good_weather feature
    weather_features['is_good_weather'] = (
        (weather_features['Temperatur'] >= 15) &  # Comfortable temperature
        (weather_features['Bewoelkung'] <= 5) &   # Less than 50% cloud cover
        (~weather_features.filter(like='weather_Regen').any(axis=1)) &  # No rain
        (~weather_features.filter(like='weather_Schnee').any(axis=1))   # No snow
    ).astype(int)
    
    # Get weather condition columns for later use
    weather_condition_columns = [col for col in weather_dummies.columns]
    
    # Create train/validation split based on dates
    train_mask = (train_df['Datum'] >= '2013-07-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Load test data for final predictions
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
    test_df['Warengruppe'] = test_df['id'].astype(str).str[-1].astype(int)
    
    # Add features to all datasets
    datasets = []
    for df in [train_data, val_data, test_df]:
        df = df.copy()
        
        # Merge weather features
        df = pd.merge(df, weather_features, on='Datum', how='left')
        
        # Basic date components
        df['Wochentag'] = df['Datum'].dt.dayofweek
        df['is_weekend'] = df['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
        df['week_of_year'] = df['Datum'].dt.isocalendar().week
        df['month'] = df['Datum'].dt.month
        df['day_of_month'] = df['Datum'].dt.day
        df['year'] = df['Datum'].dt.year
        
        # Seasonal features (based on importance)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # Special days (based on importance)
        df['is_silvester'] = ((df['month'] == 12) & (df['day_of_month'] == 31)).astype(int)
        
        # Month position indicators
        df['is_month_end'] = df['Datum'].dt.is_month_end.astype(int)
        
        # Time-based cyclical features
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Quarter information
        df['quarter'] = df['Datum'].dt.quarter
        
        # Add product group names
        df['Warengruppe_Name'] = df['Warengruppe'].map(get_warengruppe_name)
        
        # Fill any missing values from the weather merge
        df = df.ffill().bfill()
        
        datasets.append(df)
    
    return datasets[0], datasets[1], datasets[2], weather_condition_columns

def create_feature_pipeline(weather_condition_columns):
    """Create a pipeline for feature preprocessing"""
    # Define the preprocessing steps for different types of features
    numeric_features = [
        # Time-based features
        'Wochentag',
        'week_sin', 'week_cos', 
        'month_sin', 'month_cos',
        # Weather features (high importance)
        'Temperatur',
        'feels_like_temperature',
        'Bewoelkung'
    ]
    
    categorical_features = ['Warengruppe']
    
    binary_features = [
        # Time-based binary features
        'is_weekend',
        'is_summer', 'is_winter',
        'is_silvester', 'is_month_end',
        # Weather binary features
        'is_good_weather'
    ]
    
    # Add selected weather condition features
    selected_weather_conditions = [
        'weather_Regen',
        'weather_Schnee',
        'weather_Nebel oder Eisnebel',
        'weather_Gewitter'
    ]
    binary_features.extend([col for col in weather_condition_columns if col in selected_weather_conditions])
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)
        ],
        remainder='drop'  # Drop any columns not specified
    )
    
    # Create pipeline with preprocessor and model
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return pipeline

def evaluate_model(y_true, y_pred, dataset_name):
    """Calculate and print evaluation metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - mse / np.var(y_true)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    print(f"\nMetrics for {dataset_name}:")
    print(f"RMSE: {rmse:.2f}€")
    print(f"MAE: {mae:.2f}€")
    print(f"R²: {r2:.3f}")
    
    return metrics

def plot_predictions(train_df, train_pred, val_df=None, val_pred=None):
    """Create visualizations of the model's predictions"""
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    
    # Training data
    plt.scatter(train_df['Umsatz'], train_pred, alpha=0.5, label='Training')
    
    # Validation data if available
    if val_df is not None and val_pred is not None:
        plt.scatter(val_df['Umsatz'], val_pred, alpha=0.5, label='Validation')
    
    max_val = max(train_df['Umsatz'].max(), train_pred.max())
    if val_df is not None and val_pred is not None:
        max_val = max(max_val, val_df['Umsatz'].max(), val_pred.max())
    
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Sales (€)')
    plt.ylabel('Predicted Sales (€)')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    
    # Plot 2: Prediction Errors by Product Group
    plt.subplot(1, 2, 2)
    error_data = []
    
    # Training errors
    train_errors = train_pred - train_df['Umsatz']
    train_error_df = pd.DataFrame({
        'Product': train_df['Warengruppe_Name'],
        'Error': train_errors,
        'Set': 'Training'
    })
    error_data.append(train_error_df)
    
    # Validation errors if available
    if val_df is not None and val_pred is not None:
        val_errors = val_pred - val_df['Umsatz']
        val_error_df = pd.DataFrame({
            'Product': val_df['Warengruppe_Name'],
            'Error': val_errors,
            'Set': 'Validation'
        })
        error_data.append(val_error_df)
    
    # Combine error data
    error_df = pd.concat(error_data, ignore_index=True)
    
    # Create box plot
    sns.boxplot(data=error_df, x='Product', y='Error', hue='Set')
    plt.xticks(rotation=45)
    plt.xlabel('Product Group')
    plt.ylabel('Prediction Error (€)')
    plt.title('Prediction Errors by Product Group')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'prediction_analysis.png'))
    plt.close()

def plot_weekday_patterns(train_df, train_pred, test_df, test_pred, val_df=None, val_pred=None):
    """Create plots showing weekly patterns for each product group"""
    # Add predictions to data
    train_data = train_df.copy()
    train_data['Predicted_Sales'] = train_pred
    test_data = test_df.copy()
    test_data['Predicted_Sales'] = test_pred
    
    if val_df is not None and val_pred is not None:
        val_data = val_df.copy()
        val_data['Predicted_Sales'] = val_pred
    
    # Create a figure with subplots for each product group
    n_products = len(WARENGRUPPEN)
    fig, axes = plt.subplots(n_products, 1, figsize=(12, 4*n_products))
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    
    for (product_group, product_name), ax in zip(WARENGRUPPEN.items(), axes):
        # Get data for this product
        train_product = train_data[train_data['Warengruppe'] == product_group]
        test_product = test_data[test_data['Warengruppe'] == product_group]
        
        # Calculate average sales by weekday
        train_weekday_avg = train_product.groupby('Wochentag').agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        })
        test_weekday_avg = test_product.groupby('Wochentag')['Predicted_Sales'].mean()
        
        # Plot weekday patterns
        ax.plot(weekday_names, train_weekday_avg['Umsatz'], 
                marker='o', label='Actual (Train)', color='blue')
        ax.plot(weekday_names, train_weekday_avg['Predicted_Sales'], 
                marker='o', linestyle='--', label='Predicted (Train)', color='red')
        
        # Add validation data if available
        if val_df is not None and val_pred is not None:
            val_product = val_data[val_data['Warengruppe'] == product_group]
            val_weekday_avg = val_product.groupby('Wochentag').agg({
                'Umsatz': 'mean',
                'Predicted_Sales': 'mean'
            })
            ax.plot(weekday_names, val_weekday_avg['Umsatz'],
                   marker='o', label='Actual (Val)', color='green')
            ax.plot(weekday_names, val_weekday_avg['Predicted_Sales'],
                   marker='o', linestyle='--', label='Predicted (Val)', color='orange')
        
        ax.plot(weekday_names, test_weekday_avg, 
                marker='o', label='Predicted (Test)', color='purple', linestyle='--')
        
        ax.set_title(f'{product_name} - Weekly Sales Pattern')
        ax.set_xlabel('Weekday')
        ax.set_ylabel('Average Sales (€)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'weekly_patterns.png'))
    plt.close()

def plot_seasonal_patterns(train_df, train_pred, val_df=None, val_pred=None):
    """Create visualizations of seasonal patterns"""
    print("\nCreating seasonal pattern visualizations...")
    
    # Create a season column for plotting
    def get_season(row):
        if row['is_winter']: return 'winter'
        if row['is_spring']: return 'spring'
        if row['is_summer']: return 'summer'
        if row['is_fall']: return 'fall'
        return 'unknown'
    
    # Add season and predictions to dataframes
    train_plot = train_df.copy()
    train_plot['season'] = train_plot.apply(get_season, axis=1)
    train_plot['Predicted_Sales'] = train_pred
    
    if val_df is not None and val_pred is not None:
        val_plot = val_df.copy()
        val_plot['season'] = val_plot.apply(get_season, axis=1)
        val_plot['Predicted_Sales'] = val_pred
    
    # Calculate seasonal patterns
    seasonal_data = train_plot.groupby(['season', 'Warengruppe_Name']).agg({
        'Umsatz': ['mean', 'std'],
        'Predicted_Sales': ['mean', 'std']
    }).round(2)
    
    if val_df is not None and val_pred is not None:
        val_seasonal = val_plot.groupby(['season', 'Warengruppe_Name']).agg({
            'Umsatz': ['mean', 'std'],
            'Predicted_Sales': ['mean', 'std']
        }).round(2)
    
    # Plot seasonal patterns for each product group
    product_groups = train_plot['Warengruppe_Name'].unique()
    seasons = ['winter', 'spring', 'summer', 'fall']
    
    n_groups = len(product_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 4*n_groups))
    if n_groups == 1:
        axes = [axes]
    
    for ax, product in zip(axes, product_groups):
        # Get data for this product
        product_data = train_plot[train_plot['Warengruppe_Name'] == product]
        
        # Calculate means for actual and predicted
        seasonal_means = product_data.groupby('season').agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        })
        
        # Reorder seasons
        seasonal_means = seasonal_means.reindex(seasons)
        
        # Plot actual vs predicted
        ax.plot(seasons, seasonal_means['Umsatz'], 
                marker='o', label='Actual (Train)', color='blue')
        ax.plot(seasons, seasonal_means['Predicted_Sales'], 
                marker='o', linestyle='--', label='Predicted (Train)', color='red')
        
        # Add validation data if available
        if val_df is not None and val_pred is not None:
            val_product = val_plot[val_plot['Warengruppe_Name'] == product]
            val_means = val_product.groupby('season').agg({
                'Umsatz': 'mean',
                'Predicted_Sales': 'mean'
            }).reindex(seasons)
            
            ax.plot(seasons, val_means['Umsatz'],
                    marker='o', label='Actual (Val)', color='green')
            ax.plot(seasons, val_means['Predicted_Sales'],
                    marker='o', linestyle='--', label='Predicted (Val)', color='orange')
        
        ax.set_title(f'Seasonal Sales Pattern - {product}')
        ax.set_xlabel('Season')
        ax.set_ylabel('Average Sales')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'seasonal_patterns.png'))
    plt.close()
    
    # Save seasonal statistics
    seasonal_data.to_csv(os.path.join(OUTPUT_DIR, 'seasonal_statistics.csv'))
    if val_df is not None:
        val_seasonal.to_csv(os.path.join(OUTPUT_DIR, 'validation_seasonal_statistics.csv'))

def generate_submission_predictions(model, feature_columns, train_df):
    """Generate predictions for submission"""
    print("\nGenerating predictions for submission...")
    
    # Load submission template
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Create date and product group from ID
    pred_data = pd.DataFrame()
    pred_data['Datum'] = pd.to_datetime('20' + submission_df['id'].astype(str).str[:6], format='%Y%m%d')
    pred_data['Warengruppe'] = submission_df['id'].astype(str).str[-1].astype(int)
    
    # Load weather data
    weather_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    weather_codes_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'wettercode.csv')
    
    weather_data = pd.read_csv(weather_path)
    weather_codes = pd.read_csv(weather_codes_path, sep=';', header=None, names=['code', 'description'])
    weather_data['Datum'] = pd.to_datetime(weather_data['Datum'])
    
    # Merge weather data with codes
    weather_data = pd.merge(weather_data, weather_codes[['code', 'description']], 
                           left_on='Wettercode', right_on='code', how='left')
    
    # Create weather features
    weather_dummies = pd.get_dummies(weather_data['description'], prefix='weather')
    weather_features = pd.concat([
        weather_data[['Datum', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']],
        weather_dummies
    ], axis=1)
    
    # Calculate feels_like_temperature
    weather_features['feels_like_temperature'] = weather_features['Temperatur'] - (0.2 * weather_features['Windgeschwindigkeit'])
    
    # Create is_good_weather feature
    weather_features['is_good_weather'] = (
        (weather_features['Temperatur'] >= 15) &  # Comfortable temperature
        (weather_features['Bewoelkung'] <= 5) &   # Less than 50% cloud cover
        (~weather_features.filter(like='weather_Regen').any(axis=1)) &  # No rain
        (~weather_features.filter(like='weather_Schnee').any(axis=1))   # No snow
    ).astype(int)
    
    # Merge weather features
    pred_data = pd.merge(pred_data, weather_features, on='Datum', how='left')
    
    # Add all required features
    pred_data['Wochentag'] = pred_data['Datum'].dt.dayofweek
    pred_data['is_weekend'] = pred_data['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
    pred_data['week_of_year'] = pred_data['Datum'].dt.isocalendar().week
    pred_data['month'] = pred_data['Datum'].dt.month
    pred_data['day_of_month'] = pred_data['Datum'].dt.day
    pred_data['year'] = pred_data['Datum'].dt.year
    
    # Seasonal features
    pred_data['is_summer'] = pred_data['month'].isin([6, 7, 8]).astype(int)
    pred_data['is_winter'] = pred_data['month'].isin([12, 1, 2]).astype(int)
    pred_data['is_spring'] = pred_data['month'].isin([3, 4, 5]).astype(int)
    pred_data['is_fall'] = pred_data['month'].isin([9, 10, 11]).astype(int)
    
    # Special days
    pred_data['is_silvester'] = ((pred_data['month'] == 12) & (pred_data['day_of_month'] == 31)).astype(int)
    
    # Month position indicators
    pred_data['is_month_end'] = pred_data['Datum'].dt.is_month_end.astype(int)
    
    # Time-based cyclical features
    pred_data['week_sin'] = np.sin(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['week_cos'] = np.cos(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['month_sin'] = np.sin(2 * np.pi * pred_data['month'] / 12)
    pred_data['month_cos'] = np.cos(2 * np.pi * pred_data['month'] / 12)
    
    # Quarter information
    pred_data['quarter'] = pred_data['Datum'].dt.quarter
    
    # Fill any missing values from the weather merge
    pred_data = pred_data.ffill().bfill()
    
    # Make predictions
    predictions = model.predict(pred_data[feature_columns])
    
    # Add predictions to submission dataframe
    submission_df['Umsatz'] = predictions.clip(min=0)  # Ensure no negative sales predictions
    
    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    
    return submission_df

def main():
    """Main function to run the combined model with validation"""
    print("Starting combined model training with validation...")
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    train_df, val_df, test_df, weather_condition_columns = load_and_prepare_data()
    
    # Debug: Print data info
    print("\nTraining data info:")
    print(train_df.describe())
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    # Create and train model
    print("\nTraining model...")
    feature_columns = [
        # Base features
        'Warengruppe',
        'Wochentag', 'is_weekend',
        
        # Cyclical time features
        'week_sin', 'week_cos',
        'month_sin', 'month_cos',
        
        # Important temporal features
        'is_silvester',  # Highest importance
        'is_summer',     # High importance
        'is_winter',
        'is_month_end',
        
        # Weather features (high importance)
        'Temperatur',
        'feels_like_temperature',
        'Bewoelkung',
        'is_good_weather'
    ]
    
    # Add selected weather condition features based on importance
    selected_weather_conditions = [
        'weather_Regen',
        'weather_Schnee',
        'weather_Nebel oder Eisnebel',
        'weather_Gewitter'
    ]
    feature_columns.extend([col for col in weather_condition_columns if col in selected_weather_conditions])
    
    target_column = 'Umsatz'
    
    # Debug: Print feature statistics
    print("\nFeature statistics:")
    for col in feature_columns:
        if col in train_df.columns:
            print(f"\n{col}:")
            print(train_df[col].describe())
        else:
            print(f"\nMissing column: {col}")
    
    model = create_feature_pipeline(weather_condition_columns)
    
    # Fit model and make predictions
    model.fit(train_df[feature_columns], train_df[target_column])
    train_pred = model.predict(train_df[feature_columns])
    val_pred = model.predict(val_df[feature_columns])
    
    # Debug: Print prediction statistics
    print("\nPrediction statistics:")
    print("Training predictions:", pd.Series(train_pred).describe())
    print("Validation predictions:", pd.Series(val_pred).describe())
    print("Actual values:", train_df[target_column].describe())
    
    # Save predictions
    train_predictions_df = pd.DataFrame({
        'predicted_sales': train_pred,
        'actual_sales': train_df[target_column],
        'Warengruppe': train_df['Warengruppe'],
        'Datum': train_df['Datum']
    })
    train_predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'train_predictions.csv'), index=False)
    
    val_predictions_df = pd.DataFrame({
        'predicted_sales': val_pred,
        'actual_sales': val_df[target_column],
        'Warengruppe': val_df['Warengruppe'],
        'Datum': val_df['Datum']
    })
    val_predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'val_predictions.csv'), index=False)
    
    # Evaluate model
    print("\nEvaluating model performance...")
    evaluate_model(train_df[target_column], train_pred, "Training")
    evaluate_model(val_df[target_column], val_pred, "Validation")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_predictions(train_df, train_pred, val_df, val_pred)
    plot_weekday_patterns(train_df, train_pred, test_df, None, val_df, val_pred)
    plot_seasonal_patterns(train_df, train_pred, val_df, val_pred)
    
    # Generate submission predictions
    submission_df = generate_submission_predictions(model, feature_columns, train_df)
    
    print("\nModel training and evaluation complete!")
    return model, submission_df

if __name__ == "__main__":
    main()
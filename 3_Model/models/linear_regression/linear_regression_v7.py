"""
Linear Regression Model with All Features and Easter Saturday
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
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

# Additional data paths
EASTER_SATURDAY_PATH = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'easter_saturday.csv')
WINDJAMMER_PATH = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')

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
    """Load and prepare data for training and validation."""
    print("Loading and preparing data...")
    
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
    
    # Fill missing values with forward fill and backward fill
    weather_features = weather_features.ffill().bfill()
    
    # Calculate feels_like_temperature
    weather_features['feels_like_temperature'] = weather_features['Temperatur'] - (0.2 * weather_features['Windgeschwindigkeit'])
    
    # Create is_good_weather feature
    weather_features['is_good_weather'] = (
        (weather_features['Temperatur'] >= 15) &  # Comfortable temperature
        (weather_features['Bewoelkung'] <= 5) &   # Less than 50% cloud cover
        (~weather_features.filter(like='weather_Regen').any(axis=1)) &  # No rain
        (~weather_features.filter(like='weather_Schnee').any(axis=1))   # No snow
    ).astype(int)
    
    # Create is_very_good_weather feature
    weather_features['is_very_good_weather'] = (
        (weather_features['Temperatur'] >= 20) &  # Very warm temperature
        (weather_features['Bewoelkung'] <= 3) &   # Very clear skies
        (~weather_features.filter(like='weather_Regen').any(axis=1)) &  # No rain
        (~weather_features.filter(like='weather_Schnee').any(axis=1))   # No snow
    ).astype(int)
    
    # Create is_bad_weather feature
    weather_features['is_bad_weather'] = (
        (weather_features['Temperatur'] < 10) |  # Cold temperature
        (weather_features['Bewoelkung'] > 6) |   # Heavy cloud cover
        (weather_features.filter(like='weather_Regen').any(axis=1)) |  # Rain
        (weather_features.filter(like='weather_Schnee').any(axis=1))   # Snow
    ).astype(int)
    
    # Merge weather features
    train_df = pd.merge(train_df, weather_features, on='Datum', how='left')
    
    # Fill missing values from the merge with forward fill and backward fill
    train_df = train_df.ffill().bfill()
    
    # Add features
    train_df = add_features(train_df)
    
    # Create train/validation split based on dates
    train_mask = (train_df['Datum'] >= '2013-01-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
    test_df['Warengruppe'] = test_df['id'].astype(str).str[-1].astype(int)
    
    # Merge weather data for test set
    test_df = pd.merge(test_df, weather_features, on='Datum', how='left')
    
    # Fill missing values from the merge with forward fill and backward fill
    test_df = test_df.ffill().bfill()
    
    # Add features for test data
    test_df = add_features(test_df)
    
    return train_data, val_data, test_df

def add_features(df):
    """Add features to the dataframe."""
    
    # Add date-based features
    df['Wochentag'] = df['Datum'].dt.dayofweek
    df['is_weekend'] = df['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Add cyclical features
    df['week_sin'] = np.sin(2 * np.pi * df['Datum'].dt.isocalendar().week / 52.0)
    df['week_cos'] = np.cos(2 * np.pi * df['Datum'].dt.isocalendar().week / 52.0)
    df['month_sin'] = np.sin(2 * np.pi * df['Datum'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['Datum'].dt.month / 12.0)
    
    # Add seasonal features
    df['is_summer'] = df['Datum'].dt.month.isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['Datum'].dt.month.isin([12, 1, 2]).astype(int)
    
    # Load public holidays
    public_holidays_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    public_holidays_df = pd.read_csv(public_holidays_path, sep=';')
    public_holidays_df['Datum'] = pd.to_datetime(public_holidays_df['Datum'], format='%d.%m.%Y')
    df['is_public_holiday'] = df['Datum'].isin(public_holidays_df['Datum']).astype(int)
    
    # Load Easter Saturday data
    easter_saturday_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'easter_saturday.csv')
    easter_saturday_df = pd.read_csv(easter_saturday_path, names=['Datum', 'value'], skiprows=1)
    easter_saturday_df['Datum'] = pd.to_datetime(easter_saturday_df['Datum'])
    df['is_easter_saturday'] = df['Datum'].isin(easter_saturday_df['Datum']).astype(int)
    
    # Add days before/after Easter Saturday
    df['is_day_before_easter'] = df['Datum'].isin(easter_saturday_df['Datum'] - pd.Timedelta(days=1)).astype(int)
    df['is_easter_sunday'] = df['Datum'].isin(easter_saturday_df['Datum'] + pd.Timedelta(days=1)).astype(int)
    
    # Load Windjammer data
    windjammer_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    windjammer_df = pd.read_csv(windjammer_path)
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    df['Windjammerparade'] = df['Datum'].isin(windjammer_df['Datum']).astype(int)
    
    # Add days before/after Windjammer
    df['is_day_before_windjammer'] = df['Datum'].isin(windjammer_df['Datum'] - pd.Timedelta(days=1)).astype(int)
    df['is_day_after_windjammer'] = df['Datum'].isin(windjammer_df['Datum'] + pd.Timedelta(days=1)).astype(int)
    
    # Add weather-based features
    df['is_good_weather'] = ((df['Temperatur'] > 15) & (df['Bewoelkung'] < 5)).astype(int)
    df['is_very_good_weather'] = ((df['Temperatur'] > 20) & (df['Bewoelkung'] < 3)).astype(int)
    df['is_bad_weather'] = ((df['Temperatur'] < 10) | (df['Bewoelkung'] > 6)).astype(int)
    
    # Add day-relative features
    df['is_month_start'] = (df['Datum'].dt.day <= 3).astype(int)
    df['is_month_end'] = (df['Datum'].dt.day >= 28).astype(int)
    
    return df

def create_feature_pipeline(weather_condition_columns):
    """Create a pipeline for feature preprocessing"""
    
    # Numeric features that need scaling
    numeric_features = [
        'Warengruppe', 'Wochentag', 'Temperatur', 'feels_like_temperature', 'Bewoelkung'
    ]
    
    # Binary features that don't need scaling
    binary_features = [
        'is_weekend', 'is_summer', 'is_winter', 'is_good_weather',
        'Windjammerparade', 'is_easter_saturday', 'is_public_holiday',
        'is_day_before_windjammer', 'is_day_after_windjammer',
        'is_month_start', 'is_month_end', 'is_very_good_weather',
        'is_bad_weather'
    ]
    
    # Cyclical features that need sine-cosine transformation
    cyclical_features = ['week_sin', 'week_cos', 'month_sin', 'month_cos']
    
    # Create preprocessing steps for each feature type
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    binary_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=False)),
        ('identity', FunctionTransformer(lambda x: x))
    ])
    
    cyclical_transformer = Pipeline([
        ('identity', FunctionTransformer(lambda x: x))
    ])
    
    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('cyc', cyclical_transformer, cyclical_features)
        ]
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

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
    print("\nCreating visualizations...")
    
    os.makedirs(VIZ_DIR, exist_ok=True)
    sns.set_style("whitegrid")  # Use seaborn's whitegrid style
    
    # Prepare data for plotting
    train_plot_data = pd.DataFrame({
        'Actual': train_df['Umsatz'],
        'Predicted': train_pred,
        'Dataset': 'Training',
        'Date': train_df['Datum'],
        'Product': train_df['Warengruppe'].map(get_warengruppe_name)
    })
    
    plot_data = train_plot_data.copy()
    
    if val_df is not None and val_pred is not None:
        val_plot_data = pd.DataFrame({
            'Actual': val_df['Umsatz'],
            'Predicted': val_pred,
            'Dataset': 'Validation',
            'Date': val_df['Datum'],
            'Product': val_df['Warengruppe'].map(get_warengruppe_name)
        })
        plot_data = pd.concat([plot_data, val_plot_data])
    
    # Create scatter plot of predicted vs actual values
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_data, x='Actual', y='Predicted', hue='Dataset', alpha=0.5)
    max_val = float(max(plot_data['Actual'].max(), plot_data['Predicted'].max()))
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.title('Predicted vs Actual Sales')
    plt.xlabel('Actual Sales (€)')
    plt.ylabel('Predicted Sales (€)')
    plt.savefig(os.path.join(VIZ_DIR, 'predicted_vs_actual.png'))
    plt.close()
    
    # Create time series plot for each product group
    for product in plot_data['Product'].unique():
        plt.figure(figsize=(15, 6))
        product_data = plot_data[plot_data['Product'] == product]
        
        # Plot training data
        train_product = product_data[product_data['Dataset'] == 'Training']
        plt.plot(train_product['Date'], train_product['Actual'], 
                label='Actual (Training)', alpha=0.5)
        plt.plot(train_product['Date'], train_product['Predicted'], 
                label='Predicted (Training)', alpha=0.5)
        
        # Plot validation data if available
        if val_df is not None:
            val_product = product_data[product_data['Dataset'] == 'Validation']
            plt.plot(val_product['Date'], val_product['Actual'], 
                    label='Actual (Validation)', alpha=0.5)
            plt.plot(val_product['Date'], val_product['Predicted'], 
                    label='Predicted (Validation)', alpha=0.5)
        
        plt.title(f'Time Series Plot for {product}')
        plt.xlabel('Date')
        plt.ylabel('Sales (€)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'time_series_{product}.png'))
        plt.close()

def plot_weekday_patterns(train_df, train_pred, test_df=None, test_pred=None, val_df=None, val_pred=None):
    """Plot average sales by weekday"""
    # Create output directory if it doesn't exist
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Prepare training data
    train_plot = train_df.copy()
    train_plot['Predicted_Sales'] = train_pred
    train_plot['Dataset'] = 'Training'
    
    # Initialize lists for data collection
    datasets = []
    
    # Add training data
    for product in train_plot['Warengruppe'].unique():
        train_product = train_plot[train_plot['Warengruppe'] == product]
        train_actual_avg = train_product.groupby('Wochentag')['Umsatz'].mean()
        train_pred_avg = train_product.groupby('Wochentag')['Predicted_Sales'].mean()
        
        product_name = get_warengruppe_name(product)
        
        # Training actual
        train_actual_data = pd.DataFrame({
            'Weekday': train_actual_avg.index,
            'Sales': train_actual_avg.values,
            'Type': 'Actual',
            'Dataset': 'Training',
            'Product': product_name
        })
        datasets.append(train_actual_data)
        
        # Training predicted
        train_pred_data = pd.DataFrame({
            'Weekday': train_pred_avg.index,
            'Sales': train_pred_avg.values,
            'Type': 'Predicted',
            'Dataset': 'Training',
            'Product': product_name
        })
        datasets.append(train_pred_data)
    
    # Add validation data if provided
    if val_df is not None and val_pred is not None:
        val_plot = val_df.copy()
        val_plot['Predicted_Sales'] = val_pred
        val_plot['Dataset'] = 'Validation'
        
        for product in val_plot['Warengruppe'].unique():
            val_product = val_plot[val_plot['Warengruppe'] == product]
            val_actual_avg = val_product.groupby('Wochentag')['Umsatz'].mean()
            val_pred_avg = val_product.groupby('Wochentag')['Predicted_Sales'].mean()
            
            product_name = get_warengruppe_name(product)
            
            # Validation actual
            val_actual_data = pd.DataFrame({
                'Weekday': val_actual_avg.index,
                'Sales': val_actual_avg.values,
                'Type': 'Actual',
                'Dataset': 'Validation',
                'Product': product_name
            })
            datasets.append(val_actual_data)
            
            # Validation predicted
            val_pred_data = pd.DataFrame({
                'Weekday': val_pred_avg.index,
                'Sales': val_pred_avg.values,
                'Type': 'Predicted',
                'Dataset': 'Validation',
                'Product': product_name
            })
            datasets.append(val_pred_data)
    
    # Combine all data
    plot_data = pd.concat(datasets, ignore_index=True)
    
    # Create weekday pattern plots for each product
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for product in plot_data['Product'].unique():
        plt.figure(figsize=(12, 6))
        product_data = plot_data[plot_data['Product'] == product]
        
        # Plot training data
        train_actual = product_data[(product_data['Dataset'] == 'Training') & (product_data['Type'] == 'Actual')]
        train_pred = product_data[(product_data['Dataset'] == 'Training') & (product_data['Type'] == 'Predicted')]
        
        plt.plot(train_actual['Weekday'], train_actual['Sales'], 'b-', label='Actual (Training)', marker='o')
        plt.plot(train_pred['Weekday'], train_pred['Sales'], 'b--', label='Predicted (Training)', marker='s')
        
        # Plot validation data if available
        if val_df is not None:
            val_actual = product_data[(product_data['Dataset'] == 'Validation') & (product_data['Type'] == 'Actual')]
            val_pred = product_data[(product_data['Dataset'] == 'Validation') & (product_data['Type'] == 'Predicted')]
            
            plt.plot(val_actual['Weekday'], val_actual['Sales'], 'r-', label='Actual (Validation)', marker='o')
            plt.plot(val_pred['Weekday'], val_pred['Sales'], 'r--', label='Predicted (Validation)', marker='s')
        
        plt.title(f'Average Sales by Weekday - {product}')
        plt.xlabel('Weekday')
        plt.ylabel('Average Sales (€)')
        plt.xticks(range(7), weekday_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(VIZ_DIR, f'weekday_pattern_{product.replace(" ", "_")}.png'))
        plt.close()

def plot_seasonal_patterns(train_df, train_pred, val_df=None, val_pred=None):
    """Create visualizations of seasonal patterns"""
    # Create output directory if it doesn't exist
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # Prepare training data
    train_plot = train_df.copy()
    train_plot['Predicted_Sales'] = train_pred
    train_plot['Dataset'] = 'Training'
    train_plot['Warengruppe_Name'] = train_plot['Warengruppe'].apply(get_warengruppe_name)
    
    # Create season column
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    train_plot['season'] = pd.to_datetime(train_plot['Datum']).dt.month.apply(get_season)
    
    # Calculate seasonal averages for training data
    seasonal_data = train_plot.groupby(['season', 'Warengruppe_Name']).agg({
        'Umsatz': 'mean',
        'Predicted_Sales': 'mean'
    }).reset_index()
    
    # Add validation data if provided
    if val_df is not None and val_pred is not None:
        val_plot = val_df.copy()
        val_plot['Predicted_Sales'] = val_pred
        val_plot['Dataset'] = 'Validation'
        val_plot['Warengruppe_Name'] = val_plot['Warengruppe'].apply(get_warengruppe_name)
        val_plot['season'] = pd.to_datetime(val_plot['Datum']).dt.month.apply(get_season)
        
        val_seasonal = val_plot.groupby(['season', 'Warengruppe_Name']).agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        }).reset_index()
        
        # Add dataset column for plotting
        seasonal_data['Dataset'] = 'Training'
        val_seasonal['Dataset'] = 'Validation'
        
        # Combine training and validation data
        seasonal_data = pd.concat([seasonal_data, val_seasonal])
    
    # Create seasonal pattern plots for each product
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    
    for product in seasonal_data['Warengruppe_Name'].unique():
        plt.figure(figsize=(12, 6))
        product_data = seasonal_data[seasonal_data['Warengruppe_Name'] == product]
        
        # Plot training data
        train_product = product_data[product_data['Dataset'] == 'Training']
        plt.plot(train_product['season'], train_product['Umsatz'], 
                'b-', label='Actual (Training)', marker='o')
        plt.plot(train_product['season'], train_product['Predicted_Sales'], 
                'b--', label='Predicted (Training)', marker='s')
        
        # Plot validation data if available
        if val_df is not None:
            val_product = product_data[product_data['Dataset'] == 'Validation']
            plt.plot(val_product['season'], val_product['Umsatz'], 
                    'r-', label='Actual (Validation)', marker='o')
            plt.plot(val_product['season'], val_product['Predicted_Sales'], 
                    'r--', label='Predicted (Validation)', marker='s')
        
        plt.title(f'Average Sales by Season - {product}')
        plt.xlabel('Season')
        plt.ylabel('Average Sales (€)')
        plt.xticks(range(len(seasons)), seasons, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(VIZ_DIR, f'seasonal_pattern_{product.replace(" ", "_")}.png'))
        plt.close()

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
    
    # Fill missing values with forward fill and backward fill
    weather_features = weather_features.ffill().bfill()
    
    # Calculate feels_like_temperature
    weather_features['feels_like_temperature'] = weather_features['Temperatur'] - (0.2 * weather_features['Windgeschwindigkeit'])
    
    # Create is_good_weather feature
    weather_features['is_good_weather'] = (
        (weather_features['Temperatur'] >= 15) &  # Comfortable temperature
        (weather_features['Bewoelkung'] <= 5) &   # Less than 50% cloud cover
        (~weather_features.filter(like='weather_Regen').any(axis=1)) &  # No rain
        (~weather_features.filter(like='weather_Schnee').any(axis=1))   # No snow
    ).astype(int)
    
    # Create is_very_good_weather feature
    weather_features['is_very_good_weather'] = (
        (weather_features['Temperatur'] >= 20) &  # Very warm temperature
        (weather_features['Bewoelkung'] <= 3) &   # Very clear skies
        (~weather_features.filter(like='weather_Regen').any(axis=1)) &  # No rain
        (~weather_features.filter(like='weather_Schnee').any(axis=1))   # No snow
    ).astype(int)
    
    # Create is_bad_weather feature
    weather_features['is_bad_weather'] = (
        (weather_features['Temperatur'] < 10) |  # Cold temperature
        (weather_features['Bewoelkung'] > 6) |   # Heavy cloud cover
        (weather_features.filter(like='weather_Regen').any(axis=1)) |  # Rain
        (weather_features.filter(like='weather_Schnee').any(axis=1))   # Snow
    ).astype(int)
    
    # Merge weather features
    pred_data = pd.merge(pred_data, weather_features, on='Datum', how='left')
    
    # Fill missing values from the merge with forward fill and backward fill
    pred_data = pred_data.ffill().bfill()
    
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
    
    # Cyclical features
    pred_data['week_sin'] = np.sin(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['week_cos'] = np.cos(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['month_sin'] = np.sin(2 * np.pi * pred_data['month'] / 12)
    pred_data['month_cos'] = np.cos(2 * np.pi * pred_data['month'] / 12)
    
    # Month position indicators
    pred_data['is_month_start'] = pred_data['Datum'].dt.is_month_start.astype(int)
    pred_data['is_month_end'] = pred_data['Datum'].dt.is_month_end.astype(int)
    
    # Weekend interaction features
    pred_data['summer_weekend'] = (pred_data['is_summer'].astype(int) & pred_data['is_weekend'].astype(int)).astype(int)
    pred_data['winter_weekend'] = (pred_data['is_winter'].astype(int) & pred_data['is_weekend'].astype(int)).astype(int)
    pred_data['warm_weekend'] = ((pred_data['Temperatur'] >= 20).astype(int) & pred_data['is_weekend'].astype(int)).astype(int)
    pred_data['cold_weekend'] = ((pred_data['Temperatur'] <= 10).astype(int) & pred_data['is_weekend'].astype(int)).astype(int)
    
    # Special events
    # Easter Saturday (April 8, 2023)
    pred_data['is_easter_saturday'] = ((pred_data['month'] == 4) & (pred_data['day_of_month'] == 8)).astype(int)
    pred_data['is_day_before_easter'] = ((pred_data['month'] == 4) & (pred_data['day_of_month'] == 7)).astype(int)
    pred_data['is_easter_sunday'] = ((pred_data['month'] == 4) & (pred_data['day_of_month'] == 9)).astype(int)
    
    # Windjammer Parade (June 24, 2023)
    pred_data['Windjammerparade'] = ((pred_data['month'] == 6) & (pred_data['day_of_month'] == 24)).astype(int)
    pred_data['is_day_before_windjammer'] = ((pred_data['month'] == 6) & (pred_data['day_of_month'] == 23)).astype(int)
    pred_data['is_day_after_windjammer'] = ((pred_data['month'] == 6) & (pred_data['day_of_month'] == 25)).astype(int)
    
    # Public holidays in Schleswig-Holstein 2023
    holidays_2023 = {
        '2023-01-01': 'Neujahr',
        '2023-04-07': 'Karfreitag',
        '2023-04-10': 'Ostermontag',
        '2023-05-01': 'Tag der Arbeit',
        '2023-05-18': 'Christi Himmelfahrt',
        '2023-05-29': 'Pfingstmontag',
        '2023-10-03': 'Tag der Deutschen Einheit',
        '2023-10-31': 'Reformationstag',
        '2023-12-25': 'Weihnachten',
        '2023-12-26': 'Zweiter Weihnachtstag'
    }
    
    pred_data['is_public_holiday'] = pred_data['Datum'].dt.strftime('%Y-%m-%d').isin(holidays_2023.keys()).astype(int)
    
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
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Debug: Print data info
    print("\nTraining data info:")
    print(train_df.describe())
    print("\nMissing values in training data:")
    print(train_df.isnull().sum())
    
    # Create and train model
    print("\nTraining model...")
    
    # Define weather condition features first
    weather_condition_columns = [
        'weather_Regen',
        'weather_Schnee',
        'weather_Nebel oder Eisnebel',
        'weather_Gewitter',
        'weather_Schauer',
        'weather_Sprühregen',
        'weather_Trockenereignisse'
    ]
    
    # Define feature columns for the model
    feature_columns = [
        # Basic features
        'Warengruppe', 'Wochentag', 'is_weekend',
        
        # Weather features
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'feels_like_temperature',
        'is_good_weather', 'is_very_good_weather', 'is_bad_weather',
        
        # Cyclical features
        'week_sin', 'week_cos', 'month_sin', 'month_cos',
        
        # Seasonal features
        'is_summer', 'is_winter',
        
        # Event features
        'Windjammerparade', 'is_easter_saturday', 'is_public_holiday',
        'is_day_before_windjammer', 'is_day_after_windjammer',
        
        # Day-relative features
        'is_month_start', 'is_month_end'
    ]
    
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
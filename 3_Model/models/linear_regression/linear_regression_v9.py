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
import glob
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

# Configure logging to output to a file
log_file_path = os.path.join(PROJECT_ROOT, '3_Model', 'output', 'linear_regression', 'linear_regression_v9', 'linear_regression_v9.log')

# Initialize the log file by truncating it
with open(log_file_path, 'w') as log_file:
    log_file.truncate()

# Configure logging to output to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # This keeps logging to the console as well
    ]
)

# Data paths
TRAIN_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'train.csv')
TEST_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'test.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv')
WEATHER_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
EASTER_SATURDAY_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'easter_saturday.csv')
WINDJAMMER_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
PUBLIC_HOLIDAYS_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
BRIDGE_DAYS_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'bridge_days.csv')
DAY_BEFORE_HOLIDAY_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'day_before_holiday.csv')
PAYDAY_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'payday.csv')
KIELER_WOCHE_PATH = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
SCHOOL_HOLIDAYS_DIR = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'school_holidays')

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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '3_Model', 'output', 'linear_regression', MODEL_NAME)
VIZ_DIR = os.path.join(PROJECT_ROOT, '3_Model', 'visualizations', 'linear_regression', MODEL_NAME)

for directory in [OUTPUT_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load school holiday data for Schleswig-Holstein
school_holidays_sh = pd.read_csv('/Users/admin/Dropbox/@PARA/Projects/opencampus/bakery_sales_prediction/0_DataPreparation/input/compiled_data/school_holidays/school_holidays_schleswig-holstein.csv')
school_holidays_sh['Datum'] = pd.to_datetime(school_holidays_sh['date'])

def load_and_prepare_data():
    """Load and prepare data for training and validation"""
    # Load data
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    weather_data = pd.read_csv(WEATHER_PATH)
    
    # Convert dates
    train_data['Datum'] = pd.to_datetime(train_data['Datum'])
    test_data['Datum'] = pd.to_datetime(test_data['Datum'])
    weather_data['Datum'] = pd.to_datetime(weather_data['Datum'])
    
    # Log date ranges
    logging.info("\nDate ranges:")
    logging.info("Weather data: %s to %s", weather_data['Datum'].min(), weather_data['Datum'].max())
    logging.info("Test data: %s to %s", test_data['Datum'].min(), test_data['Datum'].max())
    logging.info("Train data: %s to %s", train_data['Datum'].min(), train_data['Datum'].max())
    
    # Process weather data
    weather_codes_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'wettercode.csv')
    weather_codes = pd.read_csv(weather_codes_path, sep=';', header=None, names=['code', 'description'])
    
    # Merge weather data with codes
    weather_data = pd.merge(weather_data, weather_codes[['code', 'description']], 
                          left_on='Wettercode', right_on='code', how='left')
    
    # Create weather features
    weather_dummies = pd.get_dummies(weather_data['description'], prefix='weather')
    weather_features = pd.concat([
        weather_data[['Datum', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']],
        weather_dummies
    ], axis=1)
    
    # Split training data into train and validation sets
    train_mask = (train_data['Datum'] >= '2013-07-01') & (train_data['Datum'] <= '2017-07-31')
    val_mask = (train_data['Datum'] > '2017-07-31') & (train_data['Datum'] <= '2018-07-31')
    
    train_df = train_data[train_mask].copy()
    val_df = train_data[val_mask].copy()
    
    # Add date-based features to all datasets
    for df in [train_df, val_df, test_data]:
        df['Wochentag'] = df['Datum'].dt.dayofweek
        df['is_weekend'] = df['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
        df['week_of_year'] = df['Datum'].dt.isocalendar().week
        df['month'] = df['Datum'].dt.month
        df['day_of_month'] = df['Datum'].dt.day
        df['year'] = df['Datum'].dt.year - 2013  # Scale year relative to start
        df['quarter'] = df['Datum'].dt.quarter
        
        # Seasonal features
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # Special days
        df['is_silvester'] = ((df['month'] == 12) & (df['day_of_month'] == 31)).astype(int)
        
        # Cyclical features
        df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Calculate is_month_end as a binary feature
        df['is_month_end'] = df['Datum'].dt.is_month_end.astype(int)
    
    # Add product group names
    train_df['Warengruppe_Name'] = train_df['Warengruppe'].map(get_warengruppe_name)
    val_df['Warengruppe_Name'] = val_df['Warengruppe'].map(get_warengruppe_name)
    test_data['Warengruppe_Name'] = test_data['Warengruppe'].map(get_warengruppe_name)
    
    # Merge weather features with each dataset
    train_df = pd.merge(train_df, weather_features, on='Datum', how='left')
    val_df = pd.merge(val_df, weather_features, on='Datum', how='left')
    test_data = pd.merge(test_data, weather_features, on='Datum', how='left')
    
    # Forward fill any missing weather data
    weather_cols = weather_features.columns.drop('Datum')
    for df in [train_df, val_df, test_data]:
        for col in weather_cols:
            df[col] = df[col].fillna(df[col].mean())
    
    # Calculate feels-like temperature using temperature and wind speed
    for df in [train_df, val_df, test_data]:
        temp = df['Temperatur']
        wind = df['Windgeschwindigkeit']
        
        # Initialize feels_like_temperature with actual temperature
        df['feels_like_temperature'] = temp
        
        # Apply wind chill formula for cold temperatures (below 10°C)
        cold_mask = temp < 10
        df.loc[cold_mask, 'feels_like_temperature'] = (
            13.12 + 0.6215 * temp - 11.37 * (wind ** 0.16) + 0.3965 * temp * (wind ** 0.16)
        )
        
        # Apply heat index formula for warm temperatures (above 20°C)
        warm_mask = temp > 20
        df.loc[warm_mask, 'feels_like_temperature'] = (
            -8.78469475556 +
            1.61139411 * temp +
            2.33854883889 * 0.5 +  # Assuming 50% humidity as we don't have humidity data
            -0.14611605 * temp * 0.5 +
            -0.012308094 * temp * temp +
            -0.0164248277778 * 0.5 * 0.5 +
            0.002211732 * temp * temp * 0.5 +
            0.00072546 * temp * 0.5 * 0.5 +
            -0.000003582 * temp * temp * 0.5 * 0.5
        )
        
        # Calculate temp_base_warm as a binary feature (temperature above 20°C)
        df['temp_base_warm'] = (temp >= 20).astype(int)
        
        # Calculate temp_base_kalt as a binary feature (temperature below 10°C)
        df['temp_base_kalt'] = (temp < 10).astype(int)
        
        # Calculate temp_base_mild as a binary feature (temperature between 10°C and 20°C)
        df['temp_base_mild'] = ((temp >= 10) & (temp < 20)).astype(int)
        
        # Calculate temp_seasonal_kalt as a product of temperature and is_winter
        df['temp_seasonal_kalt'] = df['Temperatur'] * df['is_winter']
        
        # Create is_good_weather feature
        df['is_good_weather'] = (
            (df['Temperatur'] >= 15) &  # Comfortable temperature
            (df['Bewoelkung'] <= 5) &   # Less than 50% cloud cover
            ~df.filter(like='weather_Regen').any(axis=1) &  # No rain
            ~df.filter(like='weather_Schnee').any(axis=1)   # No snow
        ).astype(int)
    
    # Define feature columns
    feature_columns = [
        'Warengruppe',
        'Wochentag',
        'is_weekend',
        'week_of_year',
        'month',
        'year',
        'quarter',
        'week_sin',
        'week_cos',
        'month_sin',
        'month_cos',
        'is_summer',
        'is_winter',
        'is_spring',
        'is_fall',
        'is_silvester',
        'Bewoelkung',
        'Temperatur',
        'feels_like_temperature',
        'temp_base_warm',
        'temp_base_kalt',
        'temp_base_mild',
        'Windgeschwindigkeit',
        'is_good_weather',
        'is_public_holiday',
        'is_easter_saturday',
        'is_day_before_holiday',
        'temp_seasonal_kalt',
        'is_month_end',
        'KielerWoche'
    ]
    
    # Add selected weather condition features based on importance
    selected_weather_conditions = [
        'weather_Regen',
        'weather_Schnee',
        'weather_Nebel oder Eisnebel',
        'weather_Gewitter',
        'weather_Trockenereignisse'
    ]
    
    # Only add weather conditions that exist in the data
    existing_weather_cols = [col for col in selected_weather_conditions if col in weather_dummies.columns]
    feature_columns.extend(existing_weather_cols)
    
    # Load public holidays data to define holidays_dates
    holidays_data = pd.read_csv(PUBLIC_HOLIDAYS_PATH, sep=';')
    holidays_data['Datum'] = pd.to_datetime(holidays_data['Datum'], format='%d.%m.%Y')
    holidays_dates = set(holidays_data['Datum'].dt.date)

    # Add is_day_before_holiday
    def is_day_before_holiday(date):
        next_day = date.date() + pd.Timedelta(days=1)
        return 1 if next_day in holidays_dates else 0

    train_df['is_day_before_holiday'] = train_df['Datum'].apply(is_day_before_holiday)
    val_df['is_day_before_holiday'] = val_df['Datum'].apply(is_day_before_holiday)
    
    # Merge public holidays data
    public_holidays_data = pd.read_csv(PUBLIC_HOLIDAYS_PATH, sep=';')
    public_holidays_data['Datum'] = pd.to_datetime(public_holidays_data['Datum'], format='%d.%m.%Y')
    
    train_df = pd.merge(train_df, public_holidays_data[['Datum', 'Feiertag']], on='Datum', how='left')
    val_df = pd.merge(val_df, public_holidays_data[['Datum', 'Feiertag']], on='Datum', how='left')
    
    # Create is_public_holiday feature
    train_df['is_public_holiday'] = train_df['Feiertag'].notna().astype(int)
    val_df['is_public_holiday'] = val_df['Feiertag'].notna().astype(int)
    
    # Drop the Feiertag column after use
    train_df.drop('Feiertag', axis=1, inplace=True)
    val_df.drop('Feiertag', axis=1, inplace=True)
    
    # Load Easter Saturday data
    easter_saturday_data = pd.read_csv(EASTER_SATURDAY_PATH, skiprows=1, names=['Datum', 'is_easter_saturday'])
    easter_saturday_data['Datum'] = pd.to_datetime(easter_saturday_data['Datum'], format='%Y-%m-%d')
    train_df = pd.merge(train_df, easter_saturday_data, on='Datum', how='left')
    val_df = pd.merge(val_df, easter_saturday_data, on='Datum', how='left')
    train_df['is_easter_saturday'] = train_df['is_easter_saturday'].fillna(0).astype(int)
    val_df['is_easter_saturday'] = val_df['is_easter_saturday'].fillna(0).astype(int)
    
    # Load KielerWoche data
    kiwo_data = pd.read_csv(KIELER_WOCHE_PATH)
    kiwo_data['Datum'] = pd.to_datetime(kiwo_data['Datum'])

    # Add KielerWoche feature
    def add_kielerwoche_feature(df):
        df['KielerWoche'] = df['Datum'].isin(kiwo_data['Datum']).astype(int)

    add_kielerwoche_feature(train_df)
    add_kielerwoche_feature(val_df)
    add_kielerwoche_feature(test_data)
    
    # Load school holiday data for Schleswig-Holstein
    school_holidays_sh = pd.read_csv('/Users/admin/Dropbox/@PARA/Projects/opencampus/bakery_sales_prediction/0_DataPreparation/input/compiled_data/school_holidays/school_holidays_schleswig-holstein.csv')
    school_holidays_sh['Datum'] = pd.to_datetime(school_holidays_sh['date'])

    # Add school holiday feature
    def add_school_holiday_feature(df):
        df['is_school_holiday_schleswig_holstein'] = df['Datum'].isin(school_holidays_sh['Datum']).astype(int)

    add_school_holiday_feature(train_df)
    add_school_holiday_feature(val_df)
    add_school_holiday_feature(test_data)
    
    return train_df, val_df, test_data, existing_weather_cols, feature_columns

def create_feature_pipeline(weather_condition_columns):
    """Create a pipeline with feature preprocessing"""
    # Create transformers for different feature types
    numeric_features = [
        'Warengruppe',
        'Wochentag',
        'week_of_year',
        'month',
        'year',
        'quarter',
        'Bewoelkung',
        'Temperatur',
        'feels_like_temperature',
        'temp_base_warm',
        'Windgeschwindigkeit'
    ]
    
    binary_features = [
        'is_weekend',
        'is_summer',
        'is_winter',
        'is_spring',
        'is_fall',
        'is_silvester'
    ]
    
    cyclical_features = [
        'week_sin',
        'week_cos',
        'month_sin',
        'month_cos'
    ]
    
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    binary_transformer = 'passthrough'  # Binary features are already encoded
    cyclical_transformer = 'passthrough'  # Cyclical features are already transformed
    weather_transformer = 'passthrough'  # Weather features are already one-hot encoded
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', binary_transformer, binary_features),
            ('cyc', cyclical_transformer, cyclical_features),
            ('weather', weather_transformer, weather_condition_columns)
        ],
        remainder='drop'  # Drop any columns not specified in the transformers
    )
    
    # Create pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    return model

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
    
    logging.info(f"\nMetrics for {dataset_name}:")
    logging.info(f"RMSE: {rmse:.2f}€")
    logging.info(f"MAE: {mae:.2f}€")
    logging.info(f"R²: {r2:.3f}")
    
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
    logging.info("\nCreating seasonal pattern visualizations...")
    
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

def generate_submission_predictions(model, feature_columns):
    """Generate predictions for submission"""
    logging.info("\nGenerating predictions for submission...")
    
    # Load submission template
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Create prediction data with all necessary features
    pred_data = submission_df.copy()
    
    # Extract date and product group from ID
    pred_data['Datum'] = pd.to_datetime('20' + pred_data['id'].astype(str).str[:6], format='%Y%m%d')
    pred_data['Warengruppe'] = pred_data['id'].astype(str).str[-1].astype(int)
    
    # Add date-based features
    pred_data['Wochentag'] = pred_data['Datum'].dt.dayofweek
    pred_data['is_weekend'] = pred_data['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
    pred_data['week_of_year'] = pred_data['Datum'].dt.isocalendar().week
    pred_data['month'] = pred_data['Datum'].dt.month
    pred_data['day_of_month'] = pred_data['Datum'].dt.day
    pred_data['year'] = pred_data['Datum'].dt.year - 2013  # Scale year relative to start
    pred_data['quarter'] = pred_data['Datum'].dt.quarter
    
    # Special days
    pred_data['is_silvester'] = ((pred_data['month'] == 12) & (pred_data['day_of_month'] == 31)).astype(int)
    
    # Seasonal features
    pred_data['is_summer'] = pred_data['month'].isin([6, 7, 8]).astype(int)
    pred_data['is_winter'] = pred_data['month'].isin([12, 1, 2]).astype(int)
    pred_data['is_spring'] = pred_data['month'].isin([3, 4, 5]).astype(int)
    pred_data['is_fall'] = pred_data['month'].isin([9, 10, 11]).astype(int)
    
    # Cyclical features
    pred_data['week_sin'] = np.sin(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['week_cos'] = np.cos(2 * np.pi * pred_data['week_of_year'] / 52)
    pred_data['month_sin'] = np.sin(2 * np.pi * pred_data['month'] / 12)
    pred_data['month_cos'] = np.cos(2 * np.pi * pred_data['month'] / 12)
    
    # Process weather data
    weather_codes_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'wettercode.csv')
    weather_codes = pd.read_csv(weather_codes_path, sep=';', header=None, names=['code', 'description'])
    
    weather_data = pd.read_csv(WEATHER_PATH)
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
    
    # Merge weather features
    pred_data = pd.merge(pred_data, weather_features, on='Datum', how='left')
    
    # Fill missing weather data with mean values
    weather_cols = weather_features.columns.drop('Datum')
    for col in weather_cols:
        pred_data[col] = pred_data[col].fillna(pred_data[col].mean())
    
    # Calculate feels-like temperature using temperature and wind speed
    temp = pred_data['Temperatur']
    wind = pred_data['Windgeschwindigkeit']
    
    # Initialize feels_like_temperature with actual temperature
    pred_data['feels_like_temperature'] = temp
    
    # Apply wind chill formula for cold temperatures (below 10°C)
    cold_mask = temp < 10
    pred_data.loc[cold_mask, 'feels_like_temperature'] = (
        13.12 + 0.6215 * temp - 11.37 * (wind ** 0.16) + 0.3965 * temp * (wind ** 0.16)
    )
    
    # Apply heat index formula for warm temperatures (above 20°C)
    warm_mask = temp > 20
    pred_data.loc[warm_mask, 'feels_like_temperature'] = (
        -8.78469475556 +
        1.61139411 * temp +
        2.33854883889 * 0.5 +  # Assuming 50% humidity as we don't have humidity data
        -0.14611605 * temp * 0.5 +
        -0.012308094 * temp * temp +
        -0.0164248277778 * 0.5 * 0.5 +
        0.002211732 * temp * temp * 0.5 +
        0.00072546 * temp * 0.5 * 0.5 +
        -0.000003582 * temp * temp * 0.5 * 0.5
    )
    
    # Calculate temp_base_warm as a binary feature (temperature above 20°C)
    pred_data['temp_base_warm'] = (temp >= 20).astype(int)
    
    # Calculate temp_base_kalt as a binary feature (temperature below 10°C)
    pred_data['temp_base_kalt'] = (temp < 10).astype(int)
    
    # Calculate temp_base_mild as a binary feature (temperature between 10°C and 20°C)
    pred_data['temp_base_mild'] = ((temp >= 10) & (temp < 20)).astype(int)
    
    # Calculate temp_seasonal_kalt as a product of temperature and is_winter
    pred_data['temp_seasonal_kalt'] = pred_data['Temperatur'] * pred_data['is_winter']
    
    # Create is_good_weather feature
    pred_data['is_good_weather'] = (
        (pred_data['Temperatur'] >= 15) &  # Comfortable temperature
        (pred_data['Bewoelkung'] <= 5) &   # Less than 50% cloud cover
        ~pred_data.filter(like='weather_Regen').any(axis=1) &  # No rain
        ~pred_data.filter(like='weather_Schnee').any(axis=1)   # No snow
    ).astype(int)
    
    # Merge public holidays data
    public_holidays_data = pd.read_csv(PUBLIC_HOLIDAYS_PATH, sep=';')
    public_holidays_data['Datum'] = pd.to_datetime(public_holidays_data['Datum'], format='%d.%m.%Y')
    
    pred_data = pd.merge(pred_data, public_holidays_data[['Datum', 'Feiertag']], on='Datum', how='left')
    
    # Create is_public_holiday feature
    pred_data['is_public_holiday'] = pred_data['Feiertag'].notna().astype(int)
    
    # Drop the Feiertag column after use
    pred_data.drop('Feiertag', axis=1, inplace=True)
    
    # Load public holidays data to define holidays_dates
    holidays_data = pd.read_csv(PUBLIC_HOLIDAYS_PATH, sep=';')
    holidays_data['Datum'] = pd.to_datetime(holidays_data['Datum'], format='%d.%m.%Y')
    holidays_dates = set(holidays_data['Datum'].dt.date)

    # Add is_day_before_holiday
    def is_day_before_holiday(date):
        next_day = date.date() + pd.Timedelta(days=1)
        return 1 if next_day in holidays_dates else 0

    pred_data['is_day_before_holiday'] = pred_data['Datum'].apply(is_day_before_holiday)
    
    # Load Easter Saturday data
    easter_saturday_data = pd.read_csv(EASTER_SATURDAY_PATH, skiprows=1, names=['Datum', 'is_easter_saturday'])
    easter_saturday_data['Datum'] = pd.to_datetime(easter_saturday_data['Datum'], format='%Y-%m-%d')
    pred_data = pd.merge(pred_data, easter_saturday_data, on='Datum', how='left')
    pred_data['is_easter_saturday'] = pred_data['is_easter_saturday'].fillna(0).astype(int)
    
    # Calculate is_month_end as a binary feature
    pred_data['is_month_end'] = pred_data['Datum'].dt.is_month_end.astype(int)
    
    # Load KielerWoche data
    kieler_woche_data = pd.read_csv(KIELER_WOCHE_PATH)
    kieler_woche_data['Datum'] = pd.to_datetime(kieler_woche_data['Datum'])
    pred_data = pd.merge(pred_data, kieler_woche_data[['Datum', 'KielerWoche']], on='Datum', how='left')
    pred_data['KielerWoche'] = pred_data['KielerWoche'].fillna(0).astype(int)
    
    # Add school holiday feature for Schleswig-Holstein
    pred_data['is_school_holiday_schleswig_holstein'] = pred_data['Datum'].isin(school_holidays_sh['Datum']).astype(int)
    
    # Make predictions
    predictions = model.predict(pred_data[feature_columns])
    
    # Debug: Print prediction statistics
    logging.info("\nSubmission predictions statistics:")
    logging.info(pd.Series(predictions).describe())
    
    # Add predictions to submission dataframe
    submission_df['Umsatz'] = predictions.clip(min=0)  # Ensure no negative sales predictions
    
    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    logging.info(f"Saved submission to {submission_path}")
    
    return submission_df

def check_missing_columns(pred_data, feature_columns):
    """Log missing columns in prediction data."""
    missing_cols = set(feature_columns) - set(pred_data.columns)
    if missing_cols:
        logging.warning(f'Missing columns in prediction data: {missing_cols}')
    else:
        logging.info('No missing columns in prediction data.')

def verify_data_integrity(pred_data, feature_columns):
    """Print sample and statistics of prediction data."""
    logging.info('Sample of prediction data:')
    logging.info(pred_data[feature_columns].head())
    logging.info('Statistics of prediction data:')
    logging.info(pred_data[feature_columns].describe())

def check_negative_predictions(predictions):
    """Log raw predictions and their statistics."""
    logging.info('Raw predictions:')
    logging.info(predictions)
    logging.info('Statistics of raw predictions:')
    logging.info(pd.Series(predictions).describe())

def validate_date_alignments(df_list):
    """Check date alignments in dataframes."""
    for df in df_list:
        logging.info(f'Date range for dataframe: {df["date"].min()} to {df["date"].max()}')

def confirm_external_csv_merges(df_list):
    """Log date ranges of external CSVs and confirm merge logic."""
    for df in df_list:
        logging.info(f'Date range for external CSV: {df["date"].min()} to {df["date"].max()}')

def review_year_scaling(model, feature_columns):
    """Inspect model coefficients for the year feature."""
    year_coeff = model.coef_[feature_columns.index('year')]
    logging.info(f'Year feature coefficient: {year_coeff}')

def verify_final_submission(submission_df, predictions):
    """Ensure final submission DataFrame aligns with predictions."""
    if len(submission_df) != len(predictions):
        logging.warning('Mismatch between submission DataFrame and predictions length.')
    else:
        logging.info('Submission DataFrame and predictions are aligned.')

def main():
    """Main function to run the combined model with validation"""
    logging.info("Starting combined model training with validation...")
    
    # Load and prepare data
    logging.info("\nLoading and preparing data...")
    train_df, val_df, test_df, weather_condition_columns, feature_columns = load_and_prepare_data()
    
    # Debug: Print data info
    logging.info("\nTraining data info:")
    logging.info(train_df.describe())
    logging.info("\nMissing values in training data:")
    logging.info(train_df.isnull().sum())
    
    # Create and train model
    logging.info("\nTraining model...")
    model = create_feature_pipeline(weather_condition_columns)
    
    # Fit model and make predictions
    model.fit(train_df[feature_columns], train_df['Umsatz'])
    train_pred = model.predict(train_df[feature_columns])
    val_pred = model.predict(val_df[feature_columns])
    
    # Debug: Print prediction statistics
    logging.info("\nPrediction statistics:")
    logging.info("Training predictions: %s", pd.Series(train_pred).describe())
    logging.info("Validation predictions: %s", pd.Series(val_pred).describe())
    logging.info("Actual values: %s", train_df['Umsatz'].describe())
    
    # Save predictions
    train_predictions_df = pd.DataFrame({
        'predicted_sales': train_pred,
        'actual_sales': train_df['Umsatz'],
        'Warengruppe': train_df['Warengruppe'],
        'Datum': train_df['Datum']
    })
    train_predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'train_predictions.csv'), index=False)
    
    val_predictions_df = pd.DataFrame({
        'predicted_sales': val_pred,
        'actual_sales': val_df['Umsatz'],
        'Warengruppe': val_df['Warengruppe'],
        'Datum': val_df['Datum']
    })
    val_predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'val_predictions.csv'), index=False)
    
    # Evaluate model
    logging.info("\nEvaluating model performance...")
    train_rmse = np.sqrt(mean_squared_error(train_df['Umsatz'], train_pred))
    train_mae = mean_absolute_error(train_df['Umsatz'], train_pred)
    train_r2 = r2_score(train_df['Umsatz'], train_pred)
    logging.info("\nMetrics for Training:")
    logging.info("RMSE: %.2f€", train_rmse)
    logging.info("MAE: %.2f€", train_mae)
    logging.info("R²: %.3f", train_r2)

    val_rmse = np.sqrt(mean_squared_error(val_df['Umsatz'], val_pred))
    val_mae = mean_absolute_error(val_df['Umsatz'], val_pred)
    val_r2 = r2_score(val_df['Umsatz'], val_pred)
    logging.info("\nMetrics for Validation:")
    logging.info("RMSE: %.2f€", val_rmse)
    logging.info("MAE: %.2f€", val_mae)
    logging.info("R²: %.3f", val_r2)
    
    # Create visualizations
    logging.info("\nCreating visualizations...")
    plot_predictions(train_df, train_pred, val_df, val_pred)
    plot_weekday_patterns(train_df, train_pred, test_df, None, val_df, val_pred)
    plot_seasonal_patterns(train_df, train_pred, val_df, val_pred)
    
    # Generate submission predictions
    submission_df = generate_submission_predictions(model, feature_columns)
    
    logging.info("\nModel training and evaluation complete!")
    return model, submission_df

if __name__ == "__main__":
    main()
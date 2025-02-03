#!/usr/bin/env python3
# linear_regression_v11.py
"""
Linear Regression Model (v11) - Combining best practices from v8 and v9:
 - More structured feature pipeline
 - Reduced, high-importance feature set
 - Proper scaling & encoding
 - Simpler merges (weather + minimal external data)
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MODEL_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Output directories and files
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '3_Model', 'output', 'linear_regression', MODEL_NAME)
VIZ_DIR = os.path.join(PROJECT_ROOT, '3_Model', 'visualizations', 'linear_regression', MODEL_NAME)
LOG_FILE = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.log")
SEASONAL_STATS_FILE = os.path.join(OUTPUT_DIR, "seasonal_statistics.csv")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission.csv")
TRAIN_PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "train_predictions.csv")
VAL_PREDICTIONS_FILE = os.path.join(OUTPUT_DIR, "val_predictions.csv")
VAL_SEASONAL_STATS_FILE = os.path.join(OUTPUT_DIR, "validation_seasonal_statistics.csv")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# Initialize logging
with open(LOG_FILE, 'w') as log_file:
    log_file.truncate()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Data paths
TRAIN_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "competition_data", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "competition_data", "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "competition_data", "sample_submission.csv")
WEATHER_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "competition_data", "wetter.csv")
WEATHER_CODES_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "wettercode.csv")
PUBLIC_HOLIDAYS_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "Feiertage-SH.csv")
EASTER_SATURDAY_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "easter_saturday.csv")
WINDJAMMER_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "windjammer.csv")
KIELER_WOCHE_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "kieler_woche.csv")

# Product group dictionary
WARENGRUPPEN = {
    1: "Brot",
    2: "Brötchen",
    3: "Croissant",
    4: "Konditorei",
    5: "Kuchen",
    6: "Saisonbrot"
}

# ----------- Helper functions -----------
def get_warengruppe_name(code):
    """Get the name of a product group by its code."""
    return WARENGRUPPEN.get(code, f"Unknown_{code}")

def load_and_prepare_data():
    """Load and prepare training, validation, and test data with a simpler approach."""
    
    # 1) Load train data
    train_df = pd.read_csv(TRAIN_PATH)
    train_df["Datum"] = pd.to_datetime(train_df["Datum"])
    
    # 2) Basic train/val split
    train_mask = (train_df["Datum"] >= "2013-07-01") & (train_df["Datum"] <= "2017-07-31")
    val_mask = (train_df["Datum"] >= "2017-08-01") & (train_df["Datum"] <= "2018-07-31")
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # 3) Load test data
    test_df = pd.read_csv(TEST_PATH)
    test_df["Datum"] = pd.to_datetime("20" + test_df["id"].astype(str).str[:6], format="%Y%m%d")
    test_df["Warengruppe"] = test_df["id"].astype(str).str[-1].astype(int)
    
    # ---- Load weather & minimal merges ----
    weather_df = pd.read_csv(WEATHER_PATH)
    weather_df["Datum"] = pd.to_datetime(weather_df["Datum"])
    
    # Merge with codes to get weather descriptions
    weather_codes = pd.read_csv(WEATHER_CODES_PATH, sep=";", header=None, names=["code", "description"])
    weather_df = pd.merge(weather_df, weather_codes[["code", "description"]], 
                          left_on="Wettercode", right_on="code", how="left")
    
    # Create weather dummies
    weather_dummies = pd.get_dummies(weather_df["description"], prefix="weather")
    weather_features = pd.concat([
        weather_df[["Datum", "Bewoelkung", "Temperatur", "Windgeschwindigkeit"]],
        weather_dummies
    ], axis=1)
    
    # Feels-like temperature
    weather_features["feels_like_temperature"] = (
        weather_features["Temperatur"] - 0.2 * weather_features["Windgeschwindigkeit"]
    )
    
    # Temperature base features (high importance)
    weather_features["temp_base_warm"] = (weather_features["Temperatur"] >= 20).astype(int)
    weather_features["temp_base_mild"] = ((weather_features["Temperatur"] >= 10) & (weather_features["Temperatur"] < 20)).astype(int)
    weather_features["temp_base_kalt"] = (weather_features["Temperatur"] < 10).astype(int)
    
    # is_good_weather (moderate importance)
    weather_features["is_good_weather"] = (
        (weather_features["Temperatur"] >= 15)
        & (weather_features["Bewoelkung"] <= 5)
        & (~weather_features.filter(like="weather_Regen").any(axis=1))
        & (~weather_features.filter(like="weather_Schnee").any(axis=1))
    ).astype(int)
    
    # Merge weather_features into train_data, val_data, test_df
    train_data = train_data.merge(weather_features, on="Datum", how="left")
    val_data = val_data.merge(weather_features, on="Datum", how="left")
    test_df = test_df.merge(weather_features, on="Datum", how="left")
    
    # Forward/backward fill
    train_data = train_data.ffill().bfill().infer_objects()
    val_data = val_data.ffill().bfill().infer_objects()
    test_df = test_df.ffill().bfill().infer_objects()
    
    # Load school holidays (high importance)
    school_holidays_states = [
        "thueringen", "berlin", "sachsen", "hessen", "niedersachsen",
        "schleswig-holstein", "hamburg", "bremen", "saarland", "rheinland-pfalz",
        "brandenburg", "sachsen-anhalt", "nordrhein-westfalen",
        "mecklenburg-vorpommern", "baden-wuerttemberg", "bayern"
    ]
    
    for state in school_holidays_states:
        holiday_file = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", f"school_holidays_{state}.csv")
        if os.path.exists(holiday_file):
            holidays_df = pd.read_csv(holiday_file)
            holidays_df["Datum"] = pd.to_datetime(holidays_df["Datum"])
            col_name = f"is_school_holiday_{state}"
            
            train_data = train_data.merge(holidays_df[["Datum", "is_holiday"]], on="Datum", how="left")
            val_data = val_data.merge(holidays_df[["Datum", "is_holiday"]], on="Datum", how="left")
            test_df = test_df.merge(holidays_df[["Datum", "is_holiday"]], on="Datum", how="left")
            
            train_data[col_name] = train_data["is_holiday"].fillna(0)
            val_data[col_name] = val_data["is_holiday"].fillna(0)
            test_df[col_name] = test_df["is_holiday"].fillna(0)
            
            train_data.drop("is_holiday", axis=1, inplace=True)
            val_data.drop("is_holiday", axis=1, inplace=True)
            test_df.drop("is_holiday", axis=1, inplace=True)
    
    # Public holidays
    holidays_df = pd.read_csv(PUBLIC_HOLIDAYS_PATH, sep=";")
    holidays_df["Datum"] = pd.to_datetime(holidays_df["Datum"], format="%d.%m.%Y")
    
    train_data = train_data.merge(holidays_df[["Datum", "Feiertag"]], on="Datum", how="left")
    val_data = val_data.merge(holidays_df[["Datum", "Feiertag"]], on="Datum", how="left")
    test_df = test_df.merge(holidays_df[["Datum", "Feiertag"]], on="Datum", how="left")
    
    train_data["is_public_holiday"] = train_data["Feiertag"].notna().astype(int)
    val_data["is_public_holiday"] = val_data["Feiertag"].notna().astype(int)
    test_df["is_public_holiday"] = test_df["Feiertag"].notna().astype(int)
    
    # Add is_day_before_holiday (moderate importance)
    train_data["is_day_before_holiday"] = train_data["Feiertag"].shift(-1).notna().astype(int)
    val_data["is_day_before_holiday"] = val_data["Feiertag"].shift(-1).notna().astype(int)
    test_df["is_day_before_holiday"] = test_df["Feiertag"].shift(-1).notna().astype(int)
    
    train_data.drop("Feiertag", axis=1, inplace=True)
    val_data.drop("Feiertag", axis=1, inplace=True)
    test_df.drop("Feiertag", axis=1, inplace=True)
    
    # Easter Saturday data
    easter_df = pd.read_csv(EASTER_SATURDAY_PATH, skiprows=1, names=["Datum", "is_easter_saturday"])
    easter_df["Datum"] = pd.to_datetime(easter_df["Datum"], format="%Y-%m-%d")
    
    train_data = train_data.merge(easter_df, on="Datum", how="left")
    val_data = val_data.merge(easter_df, on="Datum", how="left")
    test_df = test_df.merge(easter_df, on="Datum", how="left")
    
    train_data["is_easter_saturday"] = train_data["is_easter_saturday"].fillna(0).astype(int)
    val_data["is_easter_saturday"] = val_data["is_easter_saturday"].fillna(0).astype(int)
    test_df["is_easter_saturday"] = test_df["is_easter_saturday"].fillna(0).astype(int)
    
    # Windjammer
    windjammer_df = pd.read_csv(WINDJAMMER_PATH)
    windjammer_df["Datum"] = pd.to_datetime(windjammer_df["Datum"])
    
    train_data = train_data.merge(windjammer_df[["Datum", "Windjammerparade"]], on="Datum", how="left")
    val_data = val_data.merge(windjammer_df[["Datum", "Windjammerparade"]], on="Datum", how="left")
    test_df = test_df.merge(windjammer_df[["Datum", "Windjammerparade"]], on="Datum", how="left")
    
    train_data["is_windjammer"] = train_data["Windjammerparade"].fillna(0).astype(int)
    val_data["is_windjammer"] = val_data["Windjammerparade"].fillna(0).astype(int)
    test_df["is_windjammer"] = test_df["Windjammerparade"].fillna(0).astype(int)
    
    train_data.drop("Windjammerparade", axis=1, inplace=True)
    val_data.drop("Windjammerparade", axis=1, inplace=True)
    test_df.drop("Windjammerparade", axis=1, inplace=True)
    
    # Add Kieler Woche data
    kieler_woche_path = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "kieler_woche.csv")
    if os.path.exists(kieler_woche_path):
        kieler_woche_df = pd.read_csv(kieler_woche_path)
        kieler_woche_df["Datum"] = pd.to_datetime(kieler_woche_df["Datum"])
        
        train_data = train_data.merge(kieler_woche_df[["Datum", "KielerWoche"]], on="Datum", how="left")
        val_data = val_data.merge(kieler_woche_df[["Datum", "KielerWoche"]], on="Datum", how="left")
        test_df = test_df.merge(kieler_woche_df[["Datum", "KielerWoche"]], on="Datum", how="left")
        
        train_data["is_kieler_woche"] = train_data["KielerWoche"].fillna(0).astype(int)
        val_data["is_kieler_woche"] = val_data["KielerWoche"].fillna(0).astype(int)
        test_df["is_kieler_woche"] = test_df["KielerWoche"].fillna(0).astype(int)
        
        train_data.drop("KielerWoche", axis=1, inplace=True)
        val_data.drop("KielerWoche", axis=1, inplace=True)
        test_df.drop("KielerWoche", axis=1, inplace=True)
    
    # Add date-based features
    for df in [train_data, val_data, test_df]:
        # Day of week (high importance)
        df['day_of_week'] = df['Datum'].dt.dayofweek
        df['is_weekend'] = df['Datum'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Month and year features (moderate importance)
        df['month'] = df['Datum'].dt.month
        df['year'] = df['Datum'].dt.year - 2013  # Scale year relative to start
        df['quarter'] = df['Datum'].dt.quarter
        
        # Month position features
        df['is_month_end'] = df['Datum'].dt.is_month_end.astype(int)
        
        # Seasonal features (varying importance)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # Special days (high importance)
        df['is_silvester'] = ((df['month'] == 12) & (df['Datum'].dt.day == 31)).astype(int)
        
        # Add Warengruppe_Name mapping
        df['Warengruppe_Name'] = df['Warengruppe'].map(get_warengruppe_name)
    
    return train_data, val_data, test_df

def create_pipeline(weather_condition_cols, available_features):
    """Create the pipeline with structured numeric/categorical/binary features."""
    # High importance numeric features
    numeric_features = [
        "day_of_week",
        "Temperatur",
        "feels_like_temperature",
        "Bewoelkung",
        "year",
        "quarter"
    ]
    
    # One-hot encode Warengruppe (high importance)
    categorical_features = ["Warengruppe"]
    
    # Binary features ordered by importance
    binary_features = [
        # Very high importance
        "is_silvester",
        "is_summer",
        "temp_base_warm",
        "is_weekend",
        "is_good_weather",
    ]
    
    # School holiday features (high importance) - only add if available
    school_holiday_features = [
        "is_school_holiday_thueringen",
        "is_school_holiday_berlin",
        "is_school_holiday_sachsen",
        "is_school_holiday_hessen",
        "is_school_holiday_niedersachsen",
        "is_school_holiday_schleswig-holstein",
        "is_school_holiday_hamburg",
        "is_school_holiday_bremen",
        "is_school_holiday_saarland",
        "is_school_holiday_rheinland-pfalz",
        "is_school_holiday_brandenburg",
        "is_school_holiday_sachsen-anhalt",
        "is_school_holiday_nordrhein-westfalen",
        "is_school_holiday_mecklenburg-vorpommern",
        "is_school_holiday_baden-wuerttemberg",
        "is_school_holiday_bayern"
    ]
    
    # Add school holiday features that exist in the data
    binary_features.extend([f for f in school_holiday_features if f in available_features])
    
    # Moderate importance features
    moderate_importance = [
        "is_winter",
        "temp_base_kalt",
        "temp_base_mild",
        "is_day_before_holiday",
        "is_easter_saturday",
        "is_month_end",
        "is_kieler_woche"  # Added Kieler Woche feature (score: 44.86)
    ]
    
    # Add moderate importance features that exist in the data
    binary_features.extend([f for f in moderate_importance if f in available_features])
    
    # Lower importance but still relevant
    low_importance = [
        "is_spring",
        "is_fall",
        "is_public_holiday",
        "is_windjammer"
    ]
    
    # Add low importance features that exist in the data
    binary_features.extend([f for f in low_importance if f in available_features])
    
    # Selected weather conditions (low but relevant importance)
    selected_weather_conditions = [
        "weather_Schnee",
        "weather_Nebel oder Eisnebel",
        "weather_Regen",
        "weather_Gewitter"
    ]
    
    # Only keep columns that exist in the data
    valid_weather_cols = [c for c in weather_condition_cols if c in selected_weather_conditions and c in available_features]
    binary_features.extend(valid_weather_cols)
    
    # Verify all features exist in the data
    numeric_features = [f for f in numeric_features if f in available_features]
    categorical_features = [f for f in categorical_features if f in available_features]
    
    # Preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", "passthrough", binary_features)
        ],
        remainder="drop"
    )
    
    # Full pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    # Define final feature_columns list for convenience
    feature_columns = numeric_features + categorical_features + binary_features
    
    return model, feature_columns

def evaluate_model(y_true, y_pred, dataset_name=""):
    """Compute RMSE, MAE, R^2 and print them."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    logging.info(f"\nMetrics for {dataset_name}:")
    logging.info(f"  RMSE: {rmse:.2f}")
    logging.info(f"  MAE: {mae:.2f}")
    logging.info(f"  R² : {r2:.3f}")

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
    test_data['Predicted_Sales'] = test_pred if test_pred is not None else None
    
    if val_df is not None and val_pred is not None:
        val_data = val_df.copy()
        val_data['Predicted_Sales'] = val_pred
    
    # Create a figure with subplots for each product group
    n_products = len(WARENGRUPPEN)
    fig, axes = plt.subplots(n_products, 1, figsize=(12, 4*n_products))
    if n_products == 1:
        axes = [axes]
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    
    for (product_group, product_name), ax in zip(WARENGRUPPEN.items(), axes):
        # Get data for this product
        train_product = train_data[train_data['Warengruppe'] == product_group]
        test_product = test_data[test_data['Warengruppe'] == product_group]
        
        # Calculate average sales by weekday
        train_weekday_avg = train_product.groupby('day_of_week').agg({
            'Umsatz': 'mean',
            'Predicted_Sales': 'mean'
        })
        
        # Plot weekday patterns
        ax.plot(weekday_names, train_weekday_avg['Umsatz'], 
                marker='o', label='Actual (Train)', color='blue')
        ax.plot(weekday_names, train_weekday_avg['Predicted_Sales'], 
                marker='o', linestyle='--', label='Predicted (Train)', color='red')
        
        # Add validation data if available
        if val_df is not None and val_pred is not None:
            val_product = val_data[val_data['Warengruppe'] == product_group]
            val_weekday_avg = val_product.groupby('day_of_week').agg({
                'Umsatz': 'mean',
                'Predicted_Sales': 'mean'
            })
            ax.plot(weekday_names, val_weekday_avg['Umsatz'],
                   marker='o', label='Actual (Val)', color='green')
            ax.plot(weekday_names, val_weekday_avg['Predicted_Sales'],
                   marker='o', linestyle='--', label='Predicted (Val)', color='orange')
        
        if test_pred is not None:
            test_weekday_avg = test_product.groupby('day_of_week')['Predicted_Sales'].mean()
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
    
    seasonal_data.to_csv(SEASONAL_STATS_FILE)
    
    if val_df is not None and val_pred is not None:
        val_seasonal = val_plot.groupby(['season', 'Warengruppe_Name']).agg({
            'Umsatz': ['mean', 'std'],
            'Predicted_Sales': ['mean', 'std']
        }).round(2)
        val_seasonal.to_csv(VAL_SEASONAL_STATS_FILE)
    
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

def main():
    # 1) Load & prepare data
    logging.info("Loading and preparing data (v11)...")
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Debug: Print data info
    logging.info("\nTraining data info:")
    logging.info(train_df.describe())
    logging.info("\nMissing values in training data:")
    logging.info(train_df.isnull().sum())
    
    # 2) Create and train model
    logging.info("\nTraining model...")
    
    # Get list of available features
    available_features = train_df.columns.tolist()
    
    # Get weather condition columns
    weather_cols = [col for col in train_df.columns if col.startswith('weather_')]
    
    # Create model with available features
    model, feature_columns = create_pipeline(weather_cols, available_features)
    
    # 3) Fit model and make predictions
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
    train_predictions_df.to_csv(TRAIN_PREDICTIONS_FILE, index=False)
    
    val_predictions_df = pd.DataFrame({
        'predicted_sales': val_pred,
        'actual_sales': val_df['Umsatz'],
        'Warengruppe': val_df['Warengruppe'],
        'Datum': val_df['Datum']
    })
    val_predictions_df.to_csv(VAL_PREDICTIONS_FILE, index=False)
    
    # Evaluate model
    logging.info("\nEvaluating model performance...")
    evaluate_model(train_df['Umsatz'], train_pred, "Training")
    evaluate_model(val_df['Umsatz'], val_pred, "Validation")
    
    # Create visualizations
    logging.info("\nCreating visualizations...")
    plot_predictions(train_df, train_pred, val_df, val_pred)
    plot_weekday_patterns(train_df, train_pred, test_df, None, val_df, val_pred)
    plot_seasonal_patterns(train_df, train_pred, val_df, val_pred)
    
    # Generate submission predictions
    generate_submission_predictions(model, feature_columns, test_df)
    
    logging.info("\nModel training and evaluation complete!")
    return model

def generate_submission_predictions(model, feature_columns, test_df):
    """Use the fitted model to predict on the test set, then save to submission.csv."""
    
    # Load sample submission
    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Predict
    preds = model.predict(test_df[feature_columns])
    
    # Clip negative predictions
    submission["Umsatz"] = np.clip(preds, 0, None)
    
    out_path = os.path.join(OUTPUT_DIR, "submission_v11.csv")
    submission.to_csv(out_path, index=False)
    print(f"\nSaved final predictions to {out_path}")

if __name__ == "__main__":
    main()
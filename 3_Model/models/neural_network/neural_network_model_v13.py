"""
Neural Network Model for Bakery Sales Prediction - Version 13

Updates from v12:
- Integrated all features from linear_regression_v10
- Maintained feature scaling and preprocessing pipeline
- Optimized neural network architecture for sales prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import traceback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers, losses
import math

import joblib
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)

# Product group configurations
WARENGRUPPEN = {
    1: 'Brot',
    2: 'Brötchen',
    3: 'Croissant',
    4: 'Konditorei',
    5: 'Kuchen',
    6: 'Saisonbrot'
}

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'neural_network_model_v13'

# Create output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'neural_network', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'visualizations', 'neural_network', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Define data paths
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'competition_data', 'train.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'competition_data', 'test.csv')
WEATHER_DATA_PATH = os.path.join(DATA_DIR, 'competition_data', 'wetter.csv')
WEATHER_CODES_PATH = os.path.join(DATA_DIR, 'compiled_data', 'wettercode.csv')
EASTER_SATURDAY_PATH = os.path.join(DATA_DIR, 'compiled_data', 'easter_saturday.csv')
PUBLIC_HOLIDAYS_PATH = os.path.join(DATA_DIR, 'compiled_data', 'Feiertage-SH.csv')
KIELER_WOCHE_PATH = os.path.join(DATA_DIR, 'competition_data', 'kiwo.csv')
WINDJAMMER_PATH = os.path.join(DATA_DIR, 'compiled_data', 'windjammer.csv')
SCHOOL_HOLIDAYS_SH_PATH = os.path.join(DATA_DIR, 'compiled_data', 'Schulferientage-SH.csv')

def add_cyclical_features(df):
    """Add cyclical encoding for temporal features."""
    # Encode day of week (0-6)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Encode month (1-12)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Add week of year cyclical encoding
    df['week_of_year'] = df['Datum'].dt.isocalendar().week
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    # Add day of month cyclical encoding
    df['day_of_month'] = df['Datum'].dt.day
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    
    return df

def add_binary_features(df):
    """Add binary features to the dataframe."""
    # Weekend and holiday features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_silvester'] = ((df['Datum'].dt.month == 12) & (df['Datum'].dt.day == 31)).astype(int)
    
    # Season features
    df['month'] = df['Datum'].dt.month
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
    
    # Special day features
    df['is_month_start'] = df['Datum'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Datum'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Datum'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Datum'].dt.is_quarter_end.astype(int)
    
    return df

def add_weather_features(df):
    """Add weather-based features after weather data is merged."""
    # Temperature features
    df['temp_base_warm'] = (df['Temperatur'] >= 15).astype(int)
    df['temp_base_hot'] = (df['Temperatur'] >= 25).astype(int)
    df['temp_base_cold'] = (df['Temperatur'] <= 5).astype(int)
    
    # Temperature ranges
    df['temp_range'] = pd.qcut(df['Temperatur'], q=5, labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
    temp_dummies = pd.get_dummies(df['temp_range'], prefix='temp_range')
    df = pd.concat([df, temp_dummies], axis=1)
    df.drop('temp_range', axis=1, inplace=True)
    
    # Weather interaction features
    df['temp_wind_interaction'] = df['Temperatur'] * df['Windgeschwindigkeit']
    df['temp_cloud_interaction'] = df['Temperatur'] * df['Bewoelkung']
    df['wind_cloud_interaction'] = df['Windgeschwindigkeit'] * df['Bewoelkung']
    
    # Complex weather features
    df['weather_comfort'] = (
        (df['Temperatur'].between(18, 25)) &
        (df['Bewoelkung'] <= 5) &
        (df['Windgeschwindigkeit'] <= 15)
    ).astype(int)
    
    df['weather_discomfort'] = (
        (df['Temperatur'] < 5) | 
        (df['Temperatur'] > 30) |
        (df['Bewoelkung'] >= 7) |
        (df['Windgeschwindigkeit'] >= 25)
    ).astype(int)
    
    return df

def prepare_data(df, skip_weather_merge=False):
    """Prepare data for model training."""
    logging.info("Shape before feature engineering: %s", df.shape)
    logging.info("\nColumns before feature engineering: %s", df.columns.tolist())
    logging.info("\nSample of data before feature engineering:\n%s", df.head())
    logging.info("\nData info:")
    df.info()
    logging.info("\nChecking for NaN values:")
    logging.info(df.isnull().sum())

    # Extract date features
    df['Datum'] = pd.to_datetime(df['Datum'])
    df['day_of_week'] = df['Datum'].dt.dayofweek
    df['month'] = df['Datum'].dt.month
    df['year'] = df['Datum'].dt.year
    
    # Add cyclical features
    df = add_cyclical_features(df)
    
    # Add basic binary features (non-weather dependent)
    df = add_binary_features(df)
    
    # Merge weather data if not skipped
    if not skip_weather_merge:
        df = merge_weather_data(df)
        
        # Handle missing values in weather features
        weather_cols = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'feels_like_temperature']
        for col in weather_cols:
            # Create missing indicator
            df[f'{col}_missing'] = df[col].isna().astype(int)
            # Fill missing values with median
            df[col] = df[col].fillna(df[col].median())
            
        # Create interaction features
        df['temp_humidity'] = df['Temperatur'] * df['Bewoelkung']
        df['wind_temp'] = df['Windgeschwindigkeit'] * df['Temperatur']
        
        # Add weather-dependent features
        df = add_weather_features(df)
    
    # Select features for model
    feature_cols = [
        'day_of_week', 'month', 'year',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos',
        'week_sin', 'week_cos',
        'day_of_month_sin', 'day_of_month_cos',
        'is_weekend', 'is_silvester', 
        'is_summer', 'is_winter', 'is_spring', 'is_autumn',
        'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
        'Warengruppe'
    ]
    
    if not skip_weather_merge:
        feature_cols.extend([
            'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'feels_like_temperature',
            'Bewoelkung_missing', 'Temperatur_missing', 
            'Windgeschwindigkeit_missing', 'feels_like_temperature_missing',
            'temp_humidity', 'wind_temp',
            'temp_base_warm', 'temp_base_hot', 'temp_base_cold',
            'temp_range_very_cold', 'temp_range_cold', 'temp_range_mild', 'temp_range_warm', 'temp_range_hot',
            'temp_wind_interaction', 'temp_cloud_interaction', 'wind_cloud_interaction',
            'weather_comfort', 'weather_discomfort'
        ])
    
    X = df[feature_cols].copy()
    y = df['Umsatz'] if 'Umsatz' in df.columns else None
    
    # One-hot encode Warengruppe
    X = pd.get_dummies(X, columns=['Warengruppe'], prefix=['product_group'])
    
    return X, y

def build_model(input_dim):
    """Build neural network model with residual connections and advanced architecture."""
    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='input_layer')
    x = layers.BatchNormalization()(inputs)
    
    # First dense block
    x1 = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), 
                     kernel_initializer='he_normal')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.PReLU(shared_axes=[1])(x1)
    x1 = layers.Dropout(0.2)(x1)
    
    # Second dense block with residual
    x2 = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001),
                     kernel_initializer='he_normal')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.PReLU(shared_axes=[1])(x2)
    x2 = layers.Dropout(0.2)(x2)
    x2 = layers.Add()([x1, x2])
    
    # Third dense block
    x3 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),
                     kernel_initializer='he_normal')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.PReLU(shared_axes=[1])(x3)
    x3 = layers.Dropout(0.1)(x3)
    
    # Fourth dense block
    x4 = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001),
                     kernel_initializer='he_normal')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.PReLU(shared_axes=[1])(x4)
    x4 = layers.Dropout(0.1)(x4)
    x4 = layers.Add()([x3, x4])
    
    # Fifth dense block
    x5 = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001),
                     kernel_initializer='he_normal')(x4)
    x5 = layers.BatchNormalization()(x5)
    x5 = layers.PReLU(shared_axes=[1])(x5)
    
    # Final output layer with positive bias
    outputs = layers.Dense(1, activation='softplus', name='output',
                         bias_initializer=tf.keras.initializers.Constant(2.0))(x5)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='bakery_sales_model')
    
    # Custom loss function with stronger underprediction penalty
    def asymmetric_huber(y_true, y_pred):
        error = y_true - y_pred
        is_under = tf.cast(error > 0, tf.float32)
        is_over = tf.cast(error <= 0, tf.float32)
        
        # Much stronger penalty for underprediction
        under_weight = 2.5  # Increased further
        over_weight = 0.7   # Decreased further
        
        squared_loss = 0.5 * tf.square(error)
        linear_loss = tf.abs(error) - 0.5
        
        # Add bias term to encourage higher predictions
        prediction_scale = 0.2 * tf.reduce_mean(y_pred)  # Increased scale
        
        # Combine losses with weights and bias
        weighted_loss = (
            under_weight * is_under * tf.where(tf.abs(error) <= 1, squared_loss, linear_loss) +
            over_weight * is_over * tf.where(tf.abs(error) <= 1, squared_loss, linear_loss)
        )
        
        return tf.reduce_mean(weighted_loss) - prediction_scale
    
    # Use AdamW optimizer with weight decay
    optimizer = optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.001,
        clipnorm=1.0,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # Compile with custom loss
    model.compile(
        optimizer=optimizer,
        loss=asymmetric_huber,
        metrics=['mae']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=200):
    """Train the model with advanced callbacks and learning rate scheduling."""
    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor='val_mae',
        patience=30,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, 'best_model.keras'),
        monitor='val_mae',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Custom learning rate scheduler
    def lr_schedule(epoch):
        initial_lr = 0.001
        if epoch < 10:
            return float(initial_lr)
        else:
            return float(initial_lr * np.exp(-0.1 * (epoch - 10)))
    
    lr_scheduler = callbacks.LearningRateScheduler(lr_schedule)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint, lr_scheduler],
        verbose=1
    )
    
    # Load best weights
    model.load_weights(os.path.join(OUTPUT_DIR, 'best_model.keras'))
    
    return history

def plot_training_history(history) -> None:
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history.png'))
    plt.close()

def load_and_prepare_data():
    """Load and prepare training, validation, and test data."""
    logging.info("Loading and preparing data...")
    
    # Load train data
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df["Datum"] = pd.to_datetime(train_df["Datum"])
    
    # Add date-based features
    train_df['day_of_week'] = train_df['Datum'].dt.dayofweek
    train_df['year'] = train_df['Datum'].dt.year
    train_df['quarter'] = train_df['Datum'].dt.quarter
    train_df['month'] = train_df['Datum'].dt.month
    
    # Add Warengruppe names
    train_df['Warengruppe_Name'] = train_df['Warengruppe'].map(lambda x: WARENGRUPPEN.get(x, f'Unknown_{x}'))
    
    # Basic train/val split
    train_mask = (train_df["Datum"] >= "2013-07-01") & (train_df["Datum"] <= "2017-07-31")
    val_mask = (train_df["Datum"] >= "2017-08-01") & (train_df["Datum"] <= "2018-07-31")
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["Datum"] = pd.to_datetime("20" + test_df["id"].astype(str).str[:6], format="%Y%m%d")
    test_df["Warengruppe"] = test_df["id"].astype(str).str[-1].astype(int)
    
    # Add date-based features to test data
    test_df['day_of_week'] = test_df['Datum'].dt.dayofweek
    test_df['year'] = test_df['Datum'].dt.year
    test_df['quarter'] = test_df['Datum'].dt.quarter
    test_df['month'] = test_df['Datum'].dt.month
    
    test_df['Warengruppe_Name'] = test_df['Warengruppe'].map(lambda x: WARENGRUPPEN.get(x, f'Unknown_{x}'))
    
    # Load weather data
    weather_df = pd.read_csv(WEATHER_DATA_PATH)
    weather_df["Datum"] = pd.to_datetime(weather_df["Datum"])
    
    # Merge with codes to get weather descriptions
    weather_codes = pd.read_csv(WEATHER_CODES_PATH, sep=";", header=None, names=["code", "description"])
    weather_df = pd.merge(weather_df, weather_codes[["code", "description"]], 
                       left_on="Wettercode", right_on="code", how="left")
    
    # Create selected weather dummies
    selected_conditions = ["Regen", "Schnee", "Nebel oder Eisnebel", "Gewitter"]
    weather_dummies = pd.get_dummies(weather_df["description"], prefix="weather")
    selected_weather_dummies = weather_dummies[[col for col in weather_dummies.columns 
                                              if any(cond in col for cond in selected_conditions)]]
    
    weather_features = pd.concat([
        weather_df[["Datum", "Bewoelkung", "Temperatur", "Windgeschwindigkeit"]],
        selected_weather_dummies
    ], axis=1)
    
    # Feels-like temperature
    weather_features["feels_like_temperature"] = (
        weather_features["Temperatur"] - 0.2 * weather_features["Windgeschwindigkeit"]
    )
    
    # Merge weather data with train/val/test data
    train_data = pd.merge(train_data, weather_features, on="Datum", how="left")
    val_data = pd.merge(val_data, weather_features, on="Datum", how="left")
    test_df = pd.merge(test_df, weather_features, on="Datum", how="left")
    
    # Add binary features
    train_data = add_binary_features(train_data)
    val_data = add_binary_features(val_data)
    test_df = add_binary_features(test_df)
    
    print("\nShape before feature engineering:", train_data.shape)
    print("\nColumns before feature engineering:", train_data.columns.tolist())
    print("\nSample of data before feature engineering:")
    print(train_data.head())
    print("\nData info:")
    print(train_data.info())
    print("\nChecking for NaN values:")
    print(train_data.isna().sum())

    print("\nShape before feature engineering:", val_data.shape)
    print("\nColumns before feature engineering:", val_data.columns.tolist())
    print("\nSample of data before feature engineering:")
    print(val_data.head())
    print("\nData info:")
    print(val_data.info())
    print("\nChecking for NaN values:")
    print(val_data.isna().sum())
    
    print("\nShape before feature engineering:", test_df.shape)
    print("\nColumns before feature engineering:", test_df.columns.tolist())
    print("\nSample of data before feature engineering:")
    print(test_df.head())
    print("\nData info:")
    print(test_df.info())
    print("\nChecking for NaN values:")
    print(test_df.isna().sum())
    
    # Prepare data
    X_train, y_train = prepare_data(train_data)
    X_val, y_val = prepare_data(val_data)
    X_test, _ = prepare_data(test_df)
    
    return X_train, X_val, X_test, y_train, y_val

def merge_weather_data(df):
    """Merge weather data with the given dataframe."""
    # Load weather data
    weather_df = pd.read_csv(WEATHER_DATA_PATH)
    weather_df["Datum"] = pd.to_datetime(weather_df["Datum"])
    
    # Merge with codes to get weather descriptions
    weather_codes = pd.read_csv(WEATHER_CODES_PATH, sep=";", header=None, names=["code", "description"])
    weather_df = pd.merge(weather_df, weather_codes[["code", "description"]], 
                       left_on="Wettercode", right_on="code", how="left")
    
    # Create selected weather dummies
    selected_conditions = ["Regen", "Schnee", "Nebel oder Eisnebel", "Gewitter"]
    weather_dummies = pd.get_dummies(weather_df["description"], prefix="weather")
    selected_weather_dummies = weather_dummies[[col for col in weather_dummies.columns 
                                              if any(cond in col for cond in selected_conditions)]]
    
    weather_features = pd.concat([
        weather_df[["Datum", "Bewoelkung", "Temperatur", "Windgeschwindigkeit"]],
        selected_weather_dummies
    ], axis=1)
    
    # Feels-like temperature
    weather_features["feels_like_temperature"] = (
        weather_features["Temperatur"] - 0.2 * weather_features["Windgeschwindigkeit"]
    )
    
    # Merge weather data with the given dataframe
    df = pd.merge(df, weather_features, on="Datum", how="left")
    
    return df

def plot_predictions(train_df, train_pred, val_df=None, val_pred=None):
    """Create visualizations of the model's predictions."""
    plt.figure(figsize=(15, 8))
    plt.plot(train_df['Datum'], train_df['Umsatz'], label='Actual Train', alpha=0.6)
    plt.plot(train_df['Datum'], train_pred, label='Predicted Train', alpha=0.6)
    
    if val_df is not None and val_pred is not None:
        plt.plot(val_df['Datum'], val_df['Umsatz'], label='Actual Validation', alpha=0.6)
        plt.plot(val_df['Datum'], val_pred, label='Predicted Validation', alpha=0.6)
    
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'predictions_comparison.png'))
    plt.close()

def plot_weekday_patterns(train_data, train_pred, test_data, test_pred, val_data=None, val_pred=None):
    """Create plots showing weekly patterns for each product group."""
    # Add predictions to dataframes
    train_data = train_data.copy()
    train_data['Predicted'] = train_pred
    
    if val_data is not None and val_pred is not None:
        val_data = val_data.copy()
        val_data['Predicted'] = val_pred
    
    # Calculate average sales by day of week for each Warengruppe
    train_weekday = train_data.groupby(['Warengruppe', 'day_of_week'])[['Umsatz', 'Predicted']].mean().reset_index()
    
    if val_data is not None:
        val_weekday = val_data.groupby(['Warengruppe', 'day_of_week'])[['Umsatz', 'Predicted']].mean().reset_index()
    
    # Plot patterns for each Warengruppe
    for wg in train_weekday['Warengruppe'].unique():
        plt.figure(figsize=(12, 6))
        
        # Training data
        wg_train = train_weekday[train_weekday['Warengruppe'] == wg]
        plt.plot(wg_train['day_of_week'], wg_train['Umsatz'], 'b-', label='Actual (Train)', alpha=0.7)
        plt.plot(wg_train['day_of_week'], wg_train['Predicted'], 'b--', label='Predicted (Train)', alpha=0.7)
        
        # Validation data
        if val_data is not None:
            wg_val = val_weekday[val_weekday['Warengruppe'] == wg]
            plt.plot(wg_val['day_of_week'], wg_val['Umsatz'], 'g-', label='Actual (Val)', alpha=0.7)
            plt.plot(wg_val['day_of_week'], wg_val['Predicted'], 'g--', label='Predicted (Val)', alpha=0.7)
        
        plt.title(f'Weekly Sales Pattern - {WARENGRUPPEN[wg]}')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Sales')
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(VISUALIZATION_DIR, f'weekday_pattern_wg{wg}.png'))
        plt.close()

def plot_seasonal_patterns(train_data, train_pred, val_data, val_pred):
    """Plot seasonal patterns in the data."""
    # Prepare data
    train_plot = train_data.copy()
    train_plot['Predicted'] = train_pred
    val_plot = val_data.copy()
    val_plot['Predicted'] = val_pred
    
    # Calculate monthly averages
    train_plot['Month'] = train_plot['Datum'].dt.month
    val_plot['Month'] = val_plot['Datum'].dt.month
    
    train_monthly_actual = train_plot.groupby('Month')['Umsatz'].mean()
    train_monthly_pred = train_plot.groupby('Month')['Predicted'].mean()
    val_monthly_actual = val_plot.groupby('Month')['Umsatz'].mean()
    val_monthly_pred = val_plot.groupby('Month')['Predicted'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_monthly_actual.index, train_monthly_actual.values, 'b-', label='Actual (Train)', alpha=0.7)
    plt.plot(train_monthly_pred.index, train_monthly_pred.values, 'b--', label='Predicted (Train)', alpha=0.7)
    plt.plot(val_monthly_actual.index, val_monthly_actual.values, 'g-', label='Actual (Val)', alpha=0.7)
    plt.plot(val_monthly_pred.index, val_monthly_pred.values, 'g--', label='Predicted (Val)', alpha=0.7)
    
    plt.title('Monthly Sales Patterns')
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'seasonal_patterns.png'))
    plt.close()

def clip_predictions(predictions):
    """Clip predictions to ensure non-negative values."""
    return np.clip(predictions, 0, None)

def prepare_test_data():
    """Prepare test data for predictions"""
    logging.info("\nPreparing test data...")
    
    # Load test data template and additional data
    sample_submission_path = os.path.join(DATA_DIR, 'competition_data', 'sample_submission.csv')
    kiwo_path = os.path.join(DATA_DIR, 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(DATA_DIR, 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(DATA_DIR, 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(DATA_DIR, 'compiled_data', 'Schulferientage-SH.csv')
    
    submission_template = pd.read_csv(sample_submission_path)
    kiwo_df = pd.read_csv(kiwo_path)
    windjammer_df = pd.read_csv(windjammer_path)
    holidays_df = pd.read_csv(holidays_path, sep=';')
    school_holidays_df = pd.read_csv(school_holidays_path, sep=';')
    
    # Convert date columns
    kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    holidays_df['Datum'] = pd.to_datetime(holidays_df['Datum'], format='%d.%m.%Y')
    school_holidays_df['Datum'] = pd.to_datetime(school_holidays_df['Datum'], format='%d.%m.%Y')
    
    # Create test DataFrame from submission template
    test_df = submission_template.copy()
    test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
    test_df['Warengruppe'] = test_df['id'].astype(str).str[6:].astype(int)
    
    # Merge with all data sources
    test_df = pd.merge(test_df, kiwo_df, on='Datum', how='left')
    test_df = pd.merge(test_df, windjammer_df, on='Datum', how='left')
    test_df = pd.merge(test_df, holidays_df[['Datum', 'Feiertag']], on='Datum', how='left')
    test_df = pd.merge(test_df, school_holidays_df[['Datum', 'Ferientag']], on='Datum', how='left')
    
    # Prepare features
    X_test, _ = prepare_data(test_df)
    
    return X_test

def generate_submission_predictions(model, X_test):
    """Generate predictions for submission."""
    logging.info("Generating submission predictions...")
    print("Generating submission predictions...")

    try:
        # Generate predictions
        predictions = model.predict(X_test)

        # Clip predictions to ensure non-negative values
        predictions = clip_predictions(predictions)

        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': pd.read_csv(TEST_DATA_PATH)['id'],
            'Umsatz': predictions.flatten()
        })

        # Save predictions
        submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
        logging.info("Submission predictions saved successfully.")
        print("Submission predictions saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        print(traceback.format_exc())

def main():
    """Main function to run the neural network model."""
    logging.info("Starting neural network model training...")
    logging.info("Loading and preparing data...")

    # Load data
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    print("\nShape before feature engineering:", train_data.shape)
    print("\nColumns before feature engineering:", train_data.columns.tolist())
    print("\nSample of data before feature engineering:")
    print(train_data.head())
    print("\nData info:")
    print(train_data.info())
    print("\nChecking for NaN values:")
    print(train_data.isna().sum())
    
    train_data['Datum'] = pd.to_datetime(train_data['Datum'])
    
    # Split data into train and validation
    validation_mask = train_data['Datum'] >= "2017-08-01"
    train_subset = train_data[~validation_mask].copy()
    val_subset = train_data[validation_mask].copy()
    
    # Prepare features
    X_train, y_train = prepare_data(train_subset)
    X_val, y_val = prepare_data(val_subset)
    
    # Build and train model
    logging.info("Building and training model...")
    model = build_model(X_train.shape[1])
    logging.info("Model architecture:")
    model.summary(print_fn=logging.info)
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Make predictions
    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    
    # Clip predictions to ensure non-negative values
    train_predictions = clip_predictions(train_predictions)
    val_predictions = clip_predictions(val_predictions)
    
    # Save model
    model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))
    
    # Plot training history
    plot_training_history(history)
    
    # Plot predictions
    plot_predictions(train_subset, train_predictions, val_subset, val_predictions)
    
    # Plot patterns
    plot_seasonal_patterns(train_subset, train_predictions, val_subset, val_predictions)
    plot_weekday_patterns(train_subset, train_predictions, None, None, val_subset, val_predictions)
    
    # Save evaluation metrics
    train_mae = np.mean(np.abs(y_train - train_predictions))
    val_mae = np.mean(np.abs(y_val - val_predictions))
    
    metrics = {
        'train_mae': float(train_mae),
        'val_mae': float(val_mae),
        'final_epoch': len(history.history['loss']),
        'best_val_mae': float(min(history.history['val_mae'])),
        'best_epoch': np.argmin(history.history['val_mae']) + 1
    }
    
    with open(os.path.join(OUTPUT_DIR, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f'{metric}: {value}\n')
    
    logging.info("Training completed successfully!")
    logging.info(f"Final Training MAE: {train_mae:.2f}")
    logging.info(f"Final Validation MAE: {val_mae:.2f}")
    
    # Generate submission predictions
    X_test = prepare_test_data()
    generate_submission_predictions(model, X_test)

if __name__ == "__main__":
    main()
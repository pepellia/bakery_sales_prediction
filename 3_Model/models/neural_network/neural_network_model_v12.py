"""
Neural Network Model for Bakery Sales Prediction - Version 12

Updates from v11:
- Added all features from linear_regression_v8
- Added Easter Saturday, Windjammer, and public holiday features
- Maintained feature scaling and preprocessing pipeline
- Optimized for special event prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers, regularizers
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import tensorflow as tf
from typing import Tuple, List
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # 3_Model directory
MODEL_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Data paths
DATA_DIR = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'competition_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# Additional data paths
EASTER_SATURDAY_PATH = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'easter_saturday.csv')
WINDJAMMER_PATH = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
PUBLIC_HOLIDAYS_PATH = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')

# Create output directories
OUTPUT_DIR = os.path.join(MODEL_ROOT, 'output', 'neural_network', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(MODEL_ROOT, 'visualizations', 'neural_network', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Global variables
weather_condition_columns = []

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

def add_features(data, weather_data):
    """Add engineered features to the dataset"""
    # Merge weather data
    data = data.merge(weather_data, on='Datum', how='left')
    
    # Fill NaN values in weather features with appropriate strategies
    weather_numeric = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'feels_like_temperature']
    weather_categorical = [col for col in data.columns if col.startswith('weather_')]
    
    # For numeric weather features, use mean imputation
    for col in weather_numeric:
        data[col] = data[col].fillna(data[col].mean())
    
    # For categorical weather features, fill with 0
    data[weather_categorical] = data[weather_categorical].fillna(0)
    
    # Load Easter Saturday data
    easter_saturday_data = pd.read_csv(EASTER_SATURDAY_PATH, skiprows=1, names=['Datum', 'is_easter_saturday'])
    easter_saturday_data['Datum'] = pd.to_datetime(easter_saturday_data['Datum'], format='%Y-%m-%d')
    data = pd.merge(data, easter_saturday_data, on='Datum', how='left')
    data['is_easter_saturday'] = data['is_easter_saturday'].fillna(0).astype(int)
    
    # Load public holidays data
    public_holidays_data = pd.read_csv(PUBLIC_HOLIDAYS_PATH, sep=';')
    public_holidays_data['Datum'] = pd.to_datetime(public_holidays_data['Datum'], format='%d.%m.%Y')
    data = pd.merge(data, public_holidays_data[['Datum', 'Feiertag']], on='Datum', how='left')
    data['is_public_holiday'] = data['Feiertag'].fillna(0).astype(int)
    data.drop('Feiertag', axis=1, inplace=True)
    
    # Load Windjammer data
    windjammer_data = pd.read_csv(WINDJAMMER_PATH)
    windjammer_data['Datum'] = pd.to_datetime(windjammer_data['Datum'])
    data = pd.merge(data, windjammer_data[['Datum', 'Windjammerparade']], on='Datum', how='left')
    data['is_windjammer'] = data['Windjammerparade'].fillna(0).astype(int)
    data.drop('Windjammerparade', axis=1, inplace=True)
    
    # Extract date features
    data['Wochentag'] = data['Datum'].dt.dayofweek
    data['is_weekend'] = data['Wochentag'].isin([5, 6]).astype(int)
    data['week_of_year'] = data['Datum'].dt.isocalendar().week
    data['month'] = data['Datum'].dt.month
    data['day_of_month'] = data['Datum'].dt.day
    data['year'] = data['Datum'].dt.year
    
    # Seasonal features
    data['is_summer'] = data['month'].isin([6, 7, 8]).astype(int)
    data['is_winter'] = data['month'].isin([12, 1, 2]).astype(int)
    data['is_spring'] = data['month'].isin([3, 4, 5]).astype(int)
    data['is_fall'] = data['month'].isin([9, 10, 11]).astype(int)
    
    # Special days
    data['is_silvester'] = ((data['month'] == 12) & (data['day_of_month'] == 31)).astype(int)
    data['is_month_end'] = (data['day_of_month'] == data['Datum'].dt.days_in_month).astype(int)
    
    # Cyclical encoding for week and month
    data['week_sin'] = np.sin(2 * np.pi * data['week_of_year']/52.0)
    data['week_cos'] = np.cos(2 * np.pi * data['week_of_year']/52.0)
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12.0)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12.0)
    
    # Quarter
    data['quarter'] = data['month'].apply(lambda x: (x-1)//3 + 1)
    
    return data

def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load and prepare data with enhanced features"""
    print("Loading and preparing data...")
    
    # Load training data
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
    global weather_condition_columns
    weather_dummies = pd.get_dummies(weather_data['description'], prefix='weather')
    weather_condition_columns = weather_dummies.columns.tolist()
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
    
    # Create train/validation split based on dates
    train_mask = (train_df['Datum'] >= '2013-07-01') & (train_df['Datum'] <= '2017-07-31')
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    
    train_data = train_df[train_mask].copy()
    val_data = train_df[val_mask].copy()
    
    # Add features to training and validation data
    train_data = add_features(train_data, weather_features)
    val_data = add_features(val_data, weather_features)
    
    # Print data info for debugging
    print("\nData Info:")
    print("Train data shape:", train_data.shape)
    print("Validation data shape:", val_data.shape)
    print("\nTrain data statistics:")
    print(train_data.describe())
    print("\nChecking for NaN values in train data:")
    print(train_data.isna().sum())
    
    # Define feature columns
    numeric_features = [
        'Temperatur',
        'feels_like_temperature',
        'Bewoelkung',
        'Wochentag',
        'week_of_year',
        'month',
        'day_of_month',
        'year',
        'week_sin',
        'week_cos',
        'month_sin',
        'month_cos',
        'quarter'
    ]
    
    binary_features = [
        'is_weekend',
        'is_summer', 'is_winter', 'is_spring', 'is_fall',
        'is_silvester', 'is_month_end',
        'is_good_weather',
        'is_easter_saturday',
        'is_public_holiday',
        'is_windjammer'
    ] + weather_condition_columns
    
    feature_columns = numeric_features + binary_features
    
    # Create scalers
    numeric_scaler = StandardScaler()
    feature_scaler = StandardScaler()
    
    # Prepare training data
    X_train = pd.DataFrame(index=train_data.index)
    X_train[numeric_features] = numeric_scaler.fit_transform(train_data[numeric_features])
    X_train[binary_features] = train_data[binary_features].astype(float)
    
    X_train_scaled = feature_scaler.fit_transform(X_train)
    
    y_train = train_data['Umsatz'].values
    
    # Normalize target variable
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train_normalized = (y_train - y_mean) / y_std
    
    # Check for NaN values in scaled data
    print("\nChecking for NaN values in scaled data:")
    print("X_train_scaled NaN count:", np.isnan(X_train_scaled).sum())
    print("y_train NaN count:", np.isnan(y_train_normalized).sum())
    
    # Prepare validation data
    X_val = pd.DataFrame(index=val_data.index)
    X_val[numeric_features] = numeric_scaler.transform(val_data[numeric_features])
    X_val[binary_features] = val_data[binary_features].astype(float)
    
    X_val_scaled = feature_scaler.transform(X_val)
    
    y_val = val_data['Umsatz'].values
    y_val_normalized = (y_val - y_mean) / y_std
    
    # Check for NaN values in scaled data
    print("\nChecking for NaN values in scaled data:")
    print("X_val_scaled NaN count:", np.isnan(X_val_scaled).sum())
    print("y_val NaN count:", np.isnan(y_val_normalized).sum())
    
    # Save scalers and normalizer parameters
    joblib.dump(numeric_scaler, os.path.join(OUTPUT_DIR, 'numeric_scaler.joblib'))
    joblib.dump(feature_scaler, os.path.join(OUTPUT_DIR, 'feature_scaler.joblib'))
    joblib.dump({'mean': y_mean, 'std': y_std}, os.path.join(OUTPUT_DIR, 'target_normalizer.joblib'))
    
    return X_train_scaled, X_val_scaled, y_train_normalized, y_val_normalized, feature_columns

def build_model(input_shape, learning_rate=0.0001):
    """Build the neural network model"""
    initializer = tf.keras.initializers.HeNormal()
    
    model = Sequential([
        Input(shape=input_shape),
        
        Dense(128, 
              kernel_initializer=initializer,
              kernel_regularizer=regularizers.l2(0.01),
              kernel_constraint=tf.keras.constraints.MaxNorm(3)),
        BatchNormalization(),
        ReLU(),
        Dropout(0.3),
        
        Dense(64,
              kernel_initializer=initializer,
              kernel_regularizer=regularizers.l2(0.01),
              kernel_constraint=tf.keras.constraints.MaxNorm(3)),
        BatchNormalization(),
        ReLU(),
        Dropout(0.2),
        
        Dense(32,
              kernel_initializer=initializer,
              kernel_regularizer=regularizers.l2(0.01),
              kernel_constraint=tf.keras.constraints.MaxNorm(3)),
        BatchNormalization(),
        ReLU(),
        Dropout(0.1),
        
        Dense(1, kernel_initializer=initializer,
             kernel_constraint=tf.keras.constraints.MaxNorm(3))
    ])
    
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    
    return model

def train_model(model, X_train, X_val, y_train, y_val, batch_size=32, epochs=200):
    """Train the model with early stopping and learning rate reduction"""
    print("\nTraining model...")
    
    # Print shapes for debugging
    print("Training data shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    
    # Check for infinite or NaN values
    print("\nChecking for infinite values:")
    print("X_train inf count:", np.isinf(X_train).sum())
    print("y_train inf count:", np.isinf(y_train).sum())
    print("X_val inf count:", np.isinf(X_val).sum())
    print("y_val inf count:", np.isinf(y_val).sum())
    
    # Replace any remaining inf values with large finite values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=1e-4,
            mode='min'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            min_delta=1e-4,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(OUTPUT_DIR, 'logs'),
            histogram_freq=1
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_training_history(history: keras.callbacks.History):
    """Plot training history including loss and metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history.png'))
    plt.close()

def prepare_test_data():
    """Prepare test data for predictions"""
    print("\nPreparing test data...")
    
    # Load test data
    test_df = pd.read_csv(TEST_PATH)
    test_df['Datum'] = pd.to_datetime('20' + test_df['id'].astype(str).str[:6], format='%Y%m%d')
    test_df['Warengruppe'] = test_df['id'].astype(str).str[-1].astype(int)
    
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
    global weather_condition_columns
    weather_dummies = pd.get_dummies(weather_data['description'], prefix='weather')
    weather_condition_columns = weather_dummies.columns.tolist()
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
    
    # Add features
    test_data = add_features(test_df, weather_features)
    
    # Fill any missing values that might have occurred during feature engineering
    numeric_features = [
        'Temperatur',
        'feels_like_temperature',
        'Bewoelkung',
        'Wochentag',
        'week_of_year',
        'month',
        'day_of_month',
        'year',
        'week_sin',
        'week_cos',
        'month_sin',
        'month_cos',
        'quarter'
    ]
    
    binary_features = [
        'is_weekend',
        'is_summer', 'is_winter', 'is_spring', 'is_fall',
        'is_silvester', 'is_month_end',
        'is_good_weather',
        'is_easter_saturday',
        'is_public_holiday',
        'is_windjammer'
    ] + weather_condition_columns
    
    # Fill missing numeric features with mean values
    for col in numeric_features:
        if col in test_data.columns and test_data[col].isnull().any():
            test_data[col].fillna(test_data[col].mean(), inplace=True)
    
    # Fill missing binary features with 0
    for col in binary_features:
        if col in test_data.columns and test_data[col].isnull().any():
            test_data[col].fillna(0, inplace=True)
    
    # Ensure all weather condition columns exist
    for col in weather_condition_columns:
        if col not in test_data.columns:
            test_data[col] = 0
    
    # Load scalers and normalizer parameters
    numeric_scaler = joblib.load(os.path.join(OUTPUT_DIR, 'numeric_scaler.joblib'))
    feature_scaler = joblib.load(os.path.join(OUTPUT_DIR, 'feature_scaler.joblib'))
    normalizer = joblib.load(os.path.join(OUTPUT_DIR, 'target_normalizer.joblib'))
    
    # Prepare features
    numeric_features = [
        'Temperatur',
        'feels_like_temperature',
        'Bewoelkung',
        'Wochentag',
        'week_of_year',
        'month',
        'day_of_month',
        'year',
        'week_sin',
        'week_cos',
        'month_sin',
        'month_cos',
        'quarter'
    ]
    
    binary_features = [
        'is_weekend',
        'is_summer', 'is_winter', 'is_spring', 'is_fall',
        'is_silvester', 'is_month_end',
        'is_good_weather',
        'is_easter_saturday',
        'is_public_holiday',
        'is_windjammer'
    ] + weather_condition_columns
    
    X_test = pd.DataFrame(index=test_data.index)
    X_test[numeric_features] = numeric_scaler.transform(test_data[numeric_features])
    X_test[binary_features] = test_data[binary_features].astype(float)
    
    X_test_scaled = feature_scaler.transform(X_test)
    
    return X_test_scaled, test_df

def predict_and_save(model, test_data, test_df):
    """Make predictions and save to submission file"""
    print("\nPreparing test data...")
    
    X_test_scaled = test_data
    
    predictions = model.predict(X_test_scaled)
    
    # Load normalizer parameters and denormalize predictions
    normalizer = joblib.load(os.path.join(OUTPUT_DIR, 'target_normalizer.joblib'))
    predictions_denormalized = predictions * normalizer['std'] + normalizer['mean']
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],  # Use the actual test data IDs
        'Umsatz': predictions_denormalized.flatten()
    })
    
    # Save submission
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved submission to {submission_path}")

def main():
    """Main function to run the model"""
    print(f"\nStarting {MODEL_NAME} training...")
    
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, X_val, y_train, y_val, feature_columns = load_and_prepare_data()
    
    # Build and train model
    model = build_model(input_shape=(X_train.shape[1],))
    model, history = train_model(model, X_train, X_val, y_train, y_val)
    
    # Evaluate model
    print("\nEvaluating model...")
    train_metrics = model.evaluate(X_train, y_train, verbose=0)
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    
    print("\nTraining Metrics:")
    print(f"Loss: {train_metrics[0]:.4f}")
    print(f"MAE: {train_metrics[1]:.4f}")
    print(f"MSE: {train_metrics[2]:.4f}")
    
    print("\nValidation Metrics:")
    print(f"Loss: {val_metrics[0]:.4f}")
    print(f"MAE: {val_metrics[1]:.4f}")
    print(f"MSE: {val_metrics[2]:.4f}")
    
    # Plot training history
    plot_training_history(history)
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    # save_model_and_scaler(model)
    
    # Prepare test data
    test_data, test_df = prepare_test_data()
    
    # Make predictions and save submission
    predict_and_save(model, test_data, test_df)
    
    print("\nDone!")

if __name__ == "__main__":
    main()

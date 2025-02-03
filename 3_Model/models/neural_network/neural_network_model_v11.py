"""
Neural Network Model for Bakery Sales Prediction - Version 11

Updates from v10:
- Added all features from feature selection analysis
- Adjusted architecture for increased feature set
- Added batch normalization layers
- Optimized dropout rates
- Maintained L1/L2 regularization
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
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

print("Keras version:", keras.__version__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'neural_network_model_v11'

# Create output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'neural_network', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'visualizations', 'neural_network', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def create_fourier_features(dates, n_components=4):
    """Create Fourier features for temporal patterns"""
    # Convert to days since epoch for continuous time
    days = (dates - pd.Timestamp("2013-01-01")).dt.total_seconds() / (24 * 60 * 60)
    
    # Weekly patterns (7-day cycle)
    weekly_features = {}
    for i in range(1, n_components + 1):
        weekly_features[f'weekly_sin_{i}'] = np.sin(2 * np.pi * i * days / 7)
        weekly_features[f'weekly_cos_{i}'] = np.cos(2 * np.pi * i * days / 7)
    
    # Yearly patterns (365.25-day cycle)
    yearly_features = {}
    for i in range(1, n_components + 1):
        yearly_features[f'yearly_sin_{i}'] = np.sin(2 * np.pi * i * days / 365.25)
        yearly_features[f'yearly_cos_{i}'] = np.cos(2 * np.pi * i * days / 365.25)
    
    return pd.DataFrame({**weekly_features, **yearly_features})

def build_model(input_dim):
    """Build the neural network model"""
    model = Sequential([
        # Input layer with batch normalization
        Input(shape=(input_dim,)),
        BatchNormalization(),
        
        # First hidden layer
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second hidden layer
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Third hidden layer
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1)
    ])
    
    # Compile model with custom learning rate
    initial_learning_rate = 0.001
    optimizer = Adam(learning_rate=initial_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean squared error for regression
        metrics=['mae', 'mape']  # Track Mean Absolute Error and Mean Absolute Percentage Error
    )
    
    return model

def load_and_prepare_data():
    """Load and prepare data for the neural network model"""
    print("\nLoading and preparing data...")
    
    # Load all data files
    train_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    weather_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    weather_codes_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'wettercode.csv')
    kiwo_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    
    # Load school holidays for all states
    school_holidays_dir = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'school_holidays')
    school_holidays_files = glob.glob(os.path.join(school_holidays_dir, '*.csv'))
    
    # Load main data
    df = pd.read_csv(train_path)
    df['date'] = pd.to_datetime(df['Datum'])
    
    # Load and process weather data
    weather_data = pd.read_csv(weather_path)
    weather_codes = pd.read_csv(weather_codes_path, sep=';', header=None, names=['code', 'description'])
    weather_data['date'] = pd.to_datetime(weather_data['Datum'])
    
    # Merge weather data with codes
    weather_data = pd.merge(weather_data, weather_codes[['code', 'description']], 
                           left_on='Wettercode', right_on='code', how='left')
    
    # Create weather features
    weather_dummies = pd.get_dummies(weather_data['description'], prefix='weather')
    weather_features = pd.concat([
        weather_data[['date', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']],
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
    
    # Load and merge other data
    kiwo_df = pd.read_csv(kiwo_path)
    windjammer_df = pd.read_csv(windjammer_path)
    holidays_df = pd.read_csv(holidays_path, sep=';')
    
    # Convert dates
    kiwo_df['date'] = pd.to_datetime(kiwo_df['Datum'])
    windjammer_df['date'] = pd.to_datetime(windjammer_df['Datum'])
    holidays_df['date'] = pd.to_datetime(holidays_df['Datum'], format='%d.%m.%Y')
    
    # Create temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_silvester'] = ((df['month'] == 12) & (df['day'] == 31)).astype(int)
    
    # Add season features
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
    
    # Add week of month
    df['week_of_month'] = df['date'].dt.day.apply(lambda x: (x - 1) // 7 + 1)
    
    # Add is_payday
    df['is_payday'] = ((df['day'] >= 25) | (df['day'] <= 3)).astype(int)
    
    # Add is_bridge_day
    holidays_dates = set(holidays_df['date'].dt.date)
    def is_bridge_day(row):
        date = row['date'].date()
        day_of_week = row['day_of_week']
        if day_of_week in [5, 6]:  # Weekend
            return 0
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)
        is_prev_holiday = prev_day in holidays_dates
        is_next_holiday = next_day in holidays_dates
        is_near_weekend = day_of_week in [0, 4]  # Monday or Friday
        return 1 if (is_prev_holiday or is_next_holiday) and is_near_weekend else 0
    
    df['is_bridge_day'] = df.apply(is_bridge_day, axis=1)
    
    # Merge all features
    df = pd.merge(df, weather_features, on='date', how='left')
    df = pd.merge(df, kiwo_df[['date', 'KielerWoche']], on='date', how='left')
    df = pd.merge(df, windjammer_df[['date', 'Windjammerparade']], on='date', how='left')
    df = pd.merge(df, holidays_df[['date', 'Feiertag']].rename(columns={'Feiertag': 'is_public_holiday'}),
                 on='date', how='left')
    
    # Load and merge school holidays for all states
    for file in school_holidays_files:
        state = os.path.basename(file).replace('school_holidays_', '').replace('.csv', '')
        state_df = pd.read_csv(file)
        state_df['date'] = pd.to_datetime(state_df['date'])
        df = pd.merge(df, 
                     state_df[['date', 'is_school_holiday']].rename(
                         columns={'is_school_holiday': f'is_school_holiday_{state}'}
                     ),
                     on='date', how='left')
    
    # Fill missing values
    df = df.fillna(0)
    
    # Create Fourier features
    fourier_features = create_fourier_features(df['date'])
    fourier_features.columns = fourier_features.columns.astype(str)  # Convert feature names to strings
    df = pd.concat([df, fourier_features], axis=1)
    
    # Select features for the model (excluding date and target)
    feature_cols = [col for col in df.columns if col not in ['date', 'Datum', 'id', 'Umsatz']]
    X = df[feature_cols]
    y = df['Umsatz']
    
    # Create train/validation split based on dates
    train_mask = (df['date'] >= '2013-07-01') & (df['date'] <= '2017-07-31')
    val_mask = (df['date'] >= '2017-08-01') & (df['date'] <= '2018-07-31')
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert back to DataFrame to keep feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Validation set shape: {X_val_scaled.shape}")
    print("\nFeatures used in the model:")
    print(X_train_scaled.columns.tolist())
    
    # Save feature names for later use
    feature_names = X_train_scaled.columns.tolist()
    feature_names = [str(name) for name in feature_names]  # Convert all feature names to strings
    np.save(os.path.join(OUTPUT_DIR, 'feature_names.npy'), feature_names)
    
    return X_train_scaled, X_val_scaled, y_train, y_val

def prepare_test_data():
    """Prepare test data for predictions"""
    print("\nPreparing test data...")
    
    # Load feature names used during training
    feature_names = np.load(os.path.join(OUTPUT_DIR, 'feature_names.npy'), allow_pickle=True)
    feature_names = [str(name) for name in feature_names]  # Convert all feature names to strings
    
    # Load test data template and additional data
    sample_submission_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv')
    kiwo_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
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
    
    # Fill missing values
    test_df['KielerWoche'] = test_df['KielerWoche'].fillna(0)
    test_df['Windjammerparade'] = test_df['Windjammerparade'].fillna(0)
    test_df['Feiertag'] = test_df['Feiertag'].fillna(0)
    test_df['Ferientag'] = test_df['Ferientag'].fillna(0)
    
    # Create temporal features
    test_df['weekday'] = test_df['Datum'].dt.dayofweek
    test_df['month'] = test_df['Datum'].dt.month
    test_df['day_of_month'] = test_df['Datum'].dt.day
    
    # Add Silvester event feature
    test_df['is_silvester'] = ((test_df['month'] == 12) & (test_df['day_of_month'] == 31)).astype(int)
    
    # Create Fourier features
    fourier_features = create_fourier_features(test_df['Datum'])
    fourier_features.columns = fourier_features.columns.astype(str)  # Convert feature names to strings
    
    # Create dummy variables
    weekday_dummies = pd.get_dummies(test_df['weekday'], prefix='weekday')
    product_dummies = pd.get_dummies(test_df['Warengruppe'], prefix='product')
    
    # Convert all column names to strings
    weekday_dummies.columns = weekday_dummies.columns.astype(str)
    product_dummies.columns = product_dummies.columns.astype(str)
    
    # Ensure all weekday columns exist (0-6)
    for i in range(7):
        col = f'weekday_{i}'
        if col not in weekday_dummies.columns:
            weekday_dummies[col] = 0
    
    # Ensure all product columns exist (1-6)
    for i in range(1, 7):
        col = f'product_{i}'
        if col not in product_dummies.columns:
            product_dummies[col] = 0
    
    # Sort columns to ensure consistent order
    weekday_dummies = weekday_dummies.reindex(sorted(weekday_dummies.columns), axis=1)
    product_dummies = product_dummies.reindex(sorted(product_dummies.columns), axis=1)
    
    # Combine features
    feature_columns = [
        # Event features with proven impact
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag', 'is_silvester'
    ]
    
    X_test = pd.concat([
        test_df[feature_columns],
        fourier_features,  # Add Fourier features
        weekday_dummies,
        product_dummies
    ], axis=1)
    
    # Convert all column names to strings
    X_test.columns = X_test.columns.astype(str)
    
    # Ensure all feature columns from training are present
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Keep only the columns used during training
    X_test = X_test[feature_names]
    
    return X_test, test_df

def train_model(X_train, X_val, y_train, y_val):
    """Train the neural network model"""
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    
    print("\nBuilding model...")
    model = build_model(X_train.shape[1])
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint to save best model
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # Reduce learning rate when training plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled,
        y_train.values.reshape(-1, 1),  # Ensure y is 2D
        validation_data=(X_val_scaled, y_val.values.reshape(-1, 1)),  # Ensure validation y is 2D
        epochs=200,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def plot_training_history(history):
    """Plot training history"""
    print("\nPlotting training history...")
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    # Plot MAPE
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mape'], label='Training MAPE')
    plt.plot(history.history['val_mape'], label='Validation MAPE')
    plt.title('Model MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history.png'))
    plt.close()

def main():
    """Main function to run the neural network model"""
    print("\nStarting neural network model training (v11 with Fourier features)...")
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    # Train the model
    model, history = train_model(X_train, X_val, y_train, y_val)
    
    # Prepare test data
    X_test, test_df = prepare_test_data()
    
    # Load scaler
    scaler = joblib.load(os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test_scaled)
    
    # Create submission file
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Umsatz': predictions.flatten()
    })
    
    # Ensure predictions are non-negative
    submission['Umsatz'] = submission['Umsatz'].clip(lower=0)
    
    # Save submission file
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission file saved to: {submission_path}")
    
    # Print validation metrics
    val_predictions = model.predict(X_val)
    val_mae = np.mean(np.abs(val_predictions.flatten() - y_val))
    val_mape = np.mean(np.abs((val_predictions.flatten() - y_val) / (y_val + 1e-7))) * 100
    print(f"\nValidation MAE: {val_mae:.2f}")
    print(f"Validation MAPE: {val_mape:.2f}%")
    
    # Create validation predictions plot
    plt.figure(figsize=(15, 6))
    plt.scatter(y_val, val_predictions, alpha=0.5)
    plt.plot([0, y_val.max()], [0, y_val.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Validation Set: Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'validation_predictions.png'))
    plt.close()
    
    # Create error distribution plot
    errors = val_predictions.flatten() - y_val
    plt.figure(figsize=(15, 6))
    sns.histplot(data=errors, stat='count', binwidth=(errors.max() - errors.min()) / 50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors on Validation Set')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'error_distribution.png'))
    plt.close()

if __name__ == "__main__":
    main()
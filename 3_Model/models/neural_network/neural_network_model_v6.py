"""
Neural Network Model for Bakery Sales Prediction - Version 6

This version combines the successful features of v4 with refined payment cycle features:
- Smooth payment cycle indicators (distance from start/middle of month)
- Cyclical encoding for annual patterns
- Weather and event data from v4
- Simplified feature set to reduce noise

The model uses a date-based train/validation split:
- Training: July 2013 to July 2017
- Validation: August 2017 to July 2018
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Keras version:", keras.__version__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'neural_network_model_v6'

# Create output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(SCRIPT_DIR, 'visualizations', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def create_cyclical_features(df, col, period):
    """Create sine and cosine features for cyclical data"""
    angle = 2 * np.pi * df[col] / period
    return np.sin(angle), np.cos(angle)

def add_temporal_features(df):
    """Add enhanced temporal features to the dataframe"""
    # Basic date components
    df['weekday'] = df['Datum'].dt.dayofweek
    df['month'] = df['Datum'].dt.month
    df['year'] = df['Datum'].dt.year
    df['day_of_month'] = df['Datum'].dt.day
    
    # Refined payment cycle features
    days_in_month = df['Datum'].dt.days_in_month
    mid_month = days_in_month / 2
    
    # Distance from start of month (0 = start, 1 = middle/end)
    df['dist_from_start'] = (df['day_of_month'] - 1) / mid_month
    df['dist_from_start'] = df['dist_from_start'].clip(0, 1)
    
    # Distance from middle of month (0 = middle, 1 = start/end)
    df['dist_from_middle'] = abs(df['day_of_month'] - mid_month) / mid_month
    df['dist_from_middle'] = df['dist_from_middle'].clip(0, 1)
    
    # Cyclical features for annual patterns
    df['month_sin'], df['month_cos'] = create_cyclical_features(df, 'month', 12)
    
    # Normalized year for long-term trends
    df['normalized_year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    return df

def create_neural_network(input_shape):
    """Create and compile the neural network model"""
    print(f"\nCreating neural network with input shape: {input_shape}")
    model = keras.Sequential([
        layers.InputLayer(input_shape=(input_shape,)),
        layers.BatchNormalization(),
        # Increased capacity for additional features
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    return model

def load_and_prepare_data():
    """Load and prepare data for the neural network model"""
    print("\nLoading and preparing data...")
    
    # Load all data files
    train_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    weather_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
    print(f"Reading training data from: {train_data_path}")
    df = pd.read_csv(train_data_path)
    
    print(f"Reading weather data from: {weather_data_path}")
    weather_df = pd.read_csv(weather_data_path)
    
    print(f"Reading Kieler Woche data from: {kiwo_data_path}")
    kiwo_df = pd.read_csv(kiwo_data_path)
    
    print(f"Reading Windjammer data from: {windjammer_path}")
    windjammer_df = pd.read_csv(windjammer_path)
    
    print(f"Reading public holidays data from: {holidays_path}")
    holidays_df = pd.read_csv(holidays_path, sep=';')
    
    print(f"Reading school holidays data from: {school_holidays_path}")
    school_holidays_df = pd.read_csv(school_holidays_path, sep=';')
    
    # Convert date columns
    df['Datum'] = pd.to_datetime(df['Datum'])
    weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
    kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    holidays_df['Datum'] = pd.to_datetime(holidays_df['Datum'], format='%d.%m.%Y')
    school_holidays_df['Datum'] = pd.to_datetime(school_holidays_df['Datum'], format='%d.%m.%Y')
    
    # Merge all data
    df = pd.merge(df, weather_df, on='Datum', how='left')
    df = pd.merge(df, kiwo_df, on='Datum', how='left')
    df = pd.merge(df, windjammer_df, on='Datum', how='left')
    df = pd.merge(df, holidays_df[['Datum', 'Feiertag']], on='Datum', how='left')
    df = pd.merge(df, school_holidays_df[['Datum', 'Ferientag']], on='Datum', how='left')
    
    # Fill missing values
    df['Bewoelkung'] = df['Bewoelkung'].fillna(df['Bewoelkung'].mean())
    df['Temperatur'] = df['Temperatur'].fillna(df['Temperatur'].mean())
    df['Windgeschwindigkeit'] = df['Windgeschwindigkeit'].fillna(df['Windgeschwindigkeit'].mean())
    df['Wettercode'] = df['Wettercode'].fillna(-1)
    df['KielerWoche'] = df['KielerWoche'].fillna(0)
    df['Windjammerparade'] = df['Windjammerparade'].fillna(0)
    df['Feiertag'] = df['Feiertag'].fillna(0)
    df['Ferientag'] = df['Ferientag'].fillna(0)
    
    # Add enhanced temporal features
    df = add_temporal_features(df)
    
    # Create dummy variables for weekday and Warengruppe
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday')
    product_dummies = pd.get_dummies(df['Warengruppe'], prefix='product')
    
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
    
    # Combine all features
    feature_columns = [
        # Basic temporal features
        'month', 'day_of_month',
        # Refined payment cycle features
        'dist_from_start', 'dist_from_middle',
        # Annual patterns
        'month_sin', 'month_cos',
        # Long-term trend
        'normalized_year',
        # Weather features
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
        # Event features
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag'
    ]
    
    X = pd.concat([
        df[feature_columns],
        weekday_dummies,
        product_dummies
    ], axis=1)
    
    y = df['Umsatz']
    
    # Create train/validation split based on dates
    train_mask = (df['Datum'] >= '2013-07-01') & (df['Datum'] <= '2017-07-31')
    val_mask = (df['Datum'] >= '2017-08-01') & (df['Datum'] <= '2018-07-31')
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val

def train_model(X_train, X_val, y_train, y_val):
    """Train the neural network model"""
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("\nTraining model...")
    model = create_neural_network(X_train.shape[1])
    
    # Add early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
    
    # Convert to numpy arrays and then to float32
    print("\nPreparing data for training...")
    X_train_scaled = np.array(X_train_scaled).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    X_val_scaled = np.array(X_val_scaled).astype(np.float32)
    y_val = np.array(y_val).astype(np.float32)
    
    print(f"\nTraining data shape: {X_train_scaled.shape}")
    print(f"Validation data shape: {X_val_scaled.shape}")
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose='auto'
    )
    
    return model, scaler, history

def plot_training_history(history):
    """Plot training history"""
    print("\nPlotting training history...")
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
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

def generate_predictions(trained_model, fitted_scaler):
    """Generate predictions for the test period"""
    print("\nGenerating predictions...")
    # Load test data template and additional data
    sample_submission_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv')
    weather_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
    submission_template = pd.read_csv(sample_submission_path)
    weather_df = pd.read_csv(weather_data_path)
    kiwo_df = pd.read_csv(kiwo_data_path)
    windjammer_df = pd.read_csv(windjammer_path)
    holidays_df = pd.read_csv(holidays_path, sep=';')
    school_holidays_df = pd.read_csv(school_holidays_path, sep=';')
    
    # Convert date columns
    weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
    kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    holidays_df['Datum'] = pd.to_datetime(holidays_df['Datum'], format='%d.%m.%Y')
    school_holidays_df['Datum'] = pd.to_datetime(school_holidays_df['Datum'], format='%d.%m.%Y')
    
    # Create test dates and merge with additional data
    test_dates = pd.to_datetime('20' + submission_template['id'].astype(str).str[:6], format='%Y%m%d')
    test_products = submission_template['id'].astype(str).str[6:].astype(int)
    
    # Create test features DataFrame
    test_df = pd.DataFrame({
        'Datum': test_dates,
        'Warengruppe': test_products
    })
    
    # Add enhanced temporal features
    test_df = add_temporal_features(test_df)
    
    # Merge with all data sources
    test_df = pd.merge(test_df, weather_df, on='Datum', how='left')
    test_df = pd.merge(test_df, kiwo_df, on='Datum', how='left')
    test_df = pd.merge(test_df, windjammer_df, on='Datum', how='left')
    test_df = pd.merge(test_df, holidays_df[['Datum', 'Feiertag']], on='Datum', how='left')
    test_df = pd.merge(test_df, school_holidays_df[['Datum', 'Ferientag']], on='Datum', how='left')
    
    # Fill missing values
    test_df['Bewoelkung'] = test_df['Bewoelkung'].fillna(test_df['Bewoelkung'].mean())
    test_df['Temperatur'] = test_df['Temperatur'].fillna(test_df['Temperatur'].mean())
    test_df['Windgeschwindigkeit'] = test_df['Windgeschwindigkeit'].fillna(test_df['Windgeschwindigkeit'].mean())
    test_df['Wettercode'] = test_df['Wettercode'].fillna(-1)
    test_df['KielerWoche'] = test_df['KielerWoche'].fillna(0)
    test_df['Windjammerparade'] = test_df['Windjammerparade'].fillna(0)
    test_df['Feiertag'] = test_df['Feiertag'].fillna(0)
    test_df['Ferientag'] = test_df['Ferientag'].fillna(0)
    
    # Create dummy variables
    weekday_dummies = pd.get_dummies(test_df['weekday'], prefix='weekday')
    product_dummies = pd.get_dummies(test_df['Warengruppe'], prefix='product')
    
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
        # Basic temporal features
        'month', 'day_of_month',
        # Refined payment cycle features
        'dist_from_start', 'dist_from_middle',
        # Annual patterns
        'month_sin', 'month_cos',
        # Long-term trend
        'normalized_year',
        # Weather features
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
        # Event features
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag'
    ]
    
    X_test = pd.concat([
        test_df[feature_columns],
        weekday_dummies,
        product_dummies
    ], axis=1)
    
    print("Generating predictions...")
    # Scale features and generate predictions
    X_test_scaled = fitted_scaler.transform(X_test)
    predictions = trained_model.predict(X_test_scaled, verbose='auto')
    
    # Create submission file
    submission_template['Umsatz'] = np.maximum(0, predictions.flatten())  # Ensure non-negative predictions
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_template.to_csv(submission_path, index=False)
    print(f"\nPredictions saved to: {submission_path}")

def main():
    """Main function to run the neural network model"""
    print("\nStarting neural network model training (v6 with refined features)...")
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    print("\nTraining neural network model...")
    model, scaler, history = train_model(X_train, X_val, y_train, y_val)
    
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nGenerating predictions...")
    generate_predictions(model, scaler)
    
    print(f"\nModel training and predictions complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main() 
"""
Neural Network Model V8
Hybrid approach combining neural networks with traditional models

This version enhances the model with additional temporal features:
- Week of year
- Quarter
- Is weekend flag
- Payment cycle indicators (start of month, end of month, mid month)
- Normalized year (relative to training start)
- Cyclical encoding of month and day features

Previous features retained:
- Weather data
- Event data (Kieler Woche, Windjammer)
- Holidays (Public and School)

The model uses a date-based train/validation split:
- Training: July 2013 to July 2017
- Validation: August 2017 to July 2018
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

print("Keras version:", keras.__version__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = 'neural_network_model_v8'

# Create output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'neural_network', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'visualizations', 'neural_network', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def create_neural_network(input_shape):
    """Create and compile the neural network model with enhanced regularization"""
    print(f"\nCreating neural network with input shape: {input_shape}")
    
    # Define custom MAPE metric
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
    
    # L2 regularization factor
    l2_factor = 0.001
    
    model = keras.Sequential([
        # Input and normalization
        layers.InputLayer(input_shape=(input_shape,)),
        layers.BatchNormalization(),
        
        # First dense block with strong regularization
        layers.Dense(128, 
                    kernel_regularizer=regularizers.l2(l2_factor),
                    activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second dense block with moderate regularization
        layers.Dense(64,
                    kernel_regularizer=regularizers.l2(l2_factor/2),
                    activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third dense block with light regularization
        layers.Dense(32,
                    kernel_regularizer=regularizers.l2(l2_factor/4),
                    activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(1)
    ])
    
    # Create optimizer with explicit learning rate
    optimizer = 'adam'  # Use string identifier instead of optimizer instance
    
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', mape]
    )
    
    model.summary()
    return model

def load_and_prepare_data():
    """Load and prepare data for the neural network model"""
    print("\nLoading and preparing data...")
    
    # Load all data files
    train_data_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    weather_data_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_data_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
    # Check if all required files exist
    required_files = [
        (train_data_path, "Training data"),
        (weather_data_path, "Weather data"),
        (kiwo_data_path, "Kieler Woche data"),
        (windjammer_path, "Windjammer data"),
        (holidays_path, "Public holidays data"),
        (school_holidays_path, "School holidays data")
    ]
    
    for file_path, file_desc in required_files:
        if not os.path.exists(file_path):
            sys.exit(f"Error: {file_desc} file not found at {file_path}")
    
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
    
    if df.empty:
        sys.exit("Error: Training dataframe is empty after merges.")
    
    # Fill missing values
    df['Bewoelkung'] = df['Bewoelkung'].fillna(df['Bewoelkung'].mean())
    df['Temperatur'] = df['Temperatur'].fillna(df['Temperatur'].mean())
    df['Windgeschwindigkeit'] = df['Windgeschwindigkeit'].fillna(df['Windgeschwindigkeit'].mean())
    df['Wettercode'] = df['Wettercode'].fillna(-1)
    df['KielerWoche'] = df['KielerWoche'].fillna(0)
    df['Windjammerparade'] = df['Windjammerparade'].fillna(0)
    df['Feiertag'] = df['Feiertag'].fillna(0)
    df['Ferientag'] = df['Ferientag'].fillna(0)
    
    # Create basic temporal features
    df['weekday'] = df['Datum'].dt.dayofweek
    df['month'] = df['Datum'].dt.month
    df['year'] = df['Datum'].dt.year
    df['day_of_month'] = df['Datum'].dt.day
    df['week_of_year'] = df['Datum'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['Datum'].dt.quarter
    
    # Enhanced temporal features
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Payment cycle indicators
    df['is_start_of_month'] = (df['day_of_month'] <= 5).astype(int)  # First 5 days
    df['is_end_of_month'] = (df['day_of_month'] >= 25).astype(int)   # Last ~5-6 days
    df['is_mid_month'] = ((df['day_of_month'] > 5) & (df['day_of_month'] < 25)).astype(int)
    
    # Normalized year for long-term trends
    df['normalized_year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
    
    # Cyclical encoding for temporal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
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
        'month', 'day_of_month', 'week_of_year', 'quarter',
        # Enhanced temporal features
        'is_weekend', 'normalized_year',
        # Payment cycle features
        'is_start_of_month', 'is_end_of_month', 'is_mid_month',
        # Cyclical features
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
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
    
    # Print feature names for debugging
    print("\nFeatures used in the model:")
    print(X.columns.tolist())
    
    return X_train, X_val, y_train, y_val

def train_model(X_train, X_val, y_train, y_val):
    """Train the neural network model"""
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print("\nTraining model...")
    model = create_neural_network(X_train.shape[1])
    
    # Add model checkpoint to save best model
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'best_model.keras'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Add early stopping with increased patience
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction with increased patience
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,  # Increased patience
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
        epochs=200,  # Increased epochs due to better regularization
        batch_size=64,  # Increased batch size for stability
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose='auto'
    )
    
    # Save the final model and scaler
    print("\nSaving model and scaler...")
    model.save(os.path.join(OUTPUT_DIR, 'final_model.keras'))
    import joblib
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    
    return model, scaler, history

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

def generate_predictions(trained_model, fitted_scaler):
    """Generate predictions for the test period"""
    print("\nGenerating predictions...")
    # Load test data template and additional data
    sample_submission_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv')
    weather_data_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_data_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
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
        'Warengruppe': test_products,
        'weekday': test_dates.dt.dayofweek,
        'month': test_dates.dt.month,
        'year': test_dates.dt.year,
        'day_of_month': test_dates.dt.day,
        'week_of_year': test_dates.dt.isocalendar().week.astype(int),
        'quarter': test_dates.dt.quarter
    })
    
    # Enhanced temporal features
    test_df['is_weekend'] = test_df['weekday'].isin([5, 6]).astype(int)
    test_df['normalized_year'] = (test_df['year'] - test_df['year'].min()) / (test_df['year'].max() - test_df['year'].min())
    
    # Payment cycle indicators
    test_df['is_start_of_month'] = (test_df['day_of_month'] <= 5).astype(int)
    test_df['is_end_of_month'] = (test_df['day_of_month'] >= 25).astype(int)
    test_df['is_mid_month'] = ((test_df['day_of_month'] > 5) & (test_df['day_of_month'] < 25)).astype(int)
    
    # Cyclical encoding for temporal features
    test_df['month_sin'] = np.sin(2 * np.pi * test_df['month'] / 12)
    test_df['month_cos'] = np.cos(2 * np.pi * test_df['month'] / 12)
    test_df['day_sin'] = np.sin(2 * np.pi * test_df['day_of_month'] / 31)
    test_df['day_cos'] = np.cos(2 * np.pi * test_df['day_of_month'] / 31)
    test_df['week_sin'] = np.sin(2 * np.pi * test_df['week_of_year'] / 52)
    test_df['week_cos'] = np.cos(2 * np.pi * test_df['week_of_year'] / 52)
    
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
        'month', 'day_of_month', 'week_of_year', 'quarter',
        # Enhanced temporal features
        'is_weekend', 'normalized_year',
        # Payment cycle features
        'is_start_of_month', 'is_end_of_month', 'is_mid_month',
        # Cyclical features
        'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_sin', 'week_cos',
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
    predictions = trained_model.predict(X_test_scaled, verbose=1)
    
    # Create submission file
    submission_template['Umsatz'] = np.maximum(0, predictions.flatten())  # Ensure non-negative predictions
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_template.to_csv(submission_path, index=False)
    print(f"\nPredictions saved to: {submission_path}")

def main():
    """Main function to run the neural network model"""
    print("\nStarting neural network model training (v7 with enhanced regularization)...")
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
"""
Neural Network Model V9
Final version with all optimizations and improvements
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers
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
MODEL_NAME = 'neural_network_model_v9'

# Create output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'neural_network', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'visualizations', 'neural_network', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def create_neural_network(input_shape):
    """Create and compile the neural network model"""
    print(f"\nCreating neural network with input shape: {input_shape}")
    
    # Define custom MAPE metric
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
    
    model = keras.Sequential([
        # Input and normalization
        layers.InputLayer(input_shape=(input_shape,)),
        layers.BatchNormalization(),
        
        # First dense block
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        # Second dense block
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Third dense block
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        # Fourth dense block (additional layer from v4)
        layers.Dense(16, activation='relu'),
        
        # Output layer
        layers.Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', mape]
    )
    
    model.summary()
    return model

def load_and_prepare_data():
    """Load and prepare data for the neural network model"""
    print("\nLoading and preparing data...")
    
    # Load all data files
    train_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    weather_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
    # Check if all required files exist
    required_files = [
        (train_path, "Training data"),
        (weather_path, "Weather data"),
        (kiwo_path, "Kieler Woche data"),
        (windjammer_path, "Windjammer data"),
        (holidays_path, "Public holidays data"),
        (school_holidays_path, "School holidays data")
    ]
    
    for file_path, file_desc in required_files:
        if not os.path.exists(file_path):
            sys.exit(f"Error: {file_desc} file not found at {file_path}")
    
    # Load all data
    df = pd.read_csv(train_path)
    weather_df = pd.read_csv(weather_path)
    kiwo_df = pd.read_csv(kiwo_path)
    windjammer_df = pd.read_csv(windjammer_path)
    holidays_df = pd.read_csv(holidays_path, sep=';')
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
    df['day_of_month'] = df['Datum'].dt.day
    
    # Add Silvester event feature
    df['is_silvester'] = ((df['month'] == 12) & (df['day_of_month'] == 31)).astype(int)
    
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
        # Weather features
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
        # Event features
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag', 'is_silvester'
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
    
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Convert to numpy arrays
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
        epochs=200,
        batch_size=64,
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
    weather_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
    submission_template = pd.read_csv(sample_submission_path)
    weather_df = pd.read_csv(weather_path)
    kiwo_df = pd.read_csv(kiwo_path)
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
        'day_of_month': test_dates.dt.day
    })
    
    # Add Silvester event feature
    test_df['is_silvester'] = ((test_df['month'] == 12) & (test_df['day_of_month'] == 31)).astype(int)
    
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
        # Weather features
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
        # Event features
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag', 'is_silvester'
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
    print("\nStarting neural network model training (v9 with Silvester event)...")
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
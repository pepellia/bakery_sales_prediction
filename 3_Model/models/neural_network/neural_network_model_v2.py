"""
Neural Network Model V2
Enhanced version with additional features and improved architecture

This version enhances the base neural network model by incorporating weather data
as additional features. The weather features include:
- Cloud cover (Bewoelkung): Scale from 0-8, where 0 is clear sky and 8 is completely overcast
- Temperature (Temperatur): Average daily temperature in Celsius
- Wind speed (Windgeschwindigkeit): Wind speed in km/h
- Weather code (Wettercode): Numeric code representing weather conditions

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
MODEL_NAME = 'neural_network_model_v2'

# Create output directories
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(SCRIPT_DIR, 'visualizations', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def create_neural_network(input_shape):
    """Create and compile the neural network model"""
    print(f"\nCreating neural network with input shape: {input_shape}")
    model = keras.Sequential()
    
    # Add layers
    model.add(layers.InputLayer(shape=(input_shape,)))
    model.add(layers.BatchNormalization())
    # Increased capacity to handle additional weather features
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1))
    
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
    
    # Load training data
    train_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    weather_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    
    print(f"Reading training data from: {train_data_path}")
    df = pd.read_csv(train_data_path)
    
    print(f"Reading weather data from: {weather_data_path}")
    weather_df = pd.read_csv(weather_data_path)
    
    # Convert date columns
    df['Datum'] = pd.to_datetime(df['Datum'])
    weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
    
    # Merge sales data with weather data
    df = pd.merge(df, weather_df, on='Datum', how='left')
    
    # Handle missing weather values
    df['Bewoelkung'].fillna(df['Bewoelkung'].mean(), inplace=True)
    df['Temperatur'].fillna(df['Temperatur'].mean(), inplace=True)
    df['Windgeschwindigkeit'].fillna(df['Windgeschwindigkeit'].mean(), inplace=True)
    df['Wettercode'].fillna(-1, inplace=True)  # Use -1 to indicate missing weather code
    
    # Create date-based features
    df['weekday'] = df['Datum'].dt.dayofweek
    df['month'] = df['Datum'].dt.month
    df['year'] = df['Datum'].dt.year
    df['day_of_month'] = df['Datum'].dt.day
    
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
        'month', 'day_of_month',
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode'
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
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,  # Increased epochs with early stopping
        batch_size=32,
        callbacks=[early_stopping],
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
    # Load test data template and weather data
    sample_submission_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv')
    weather_data_path = os.path.join(SCRIPT_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    
    submission_template = pd.read_csv(sample_submission_path)
    weather_df = pd.read_csv(weather_data_path)
    weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
    
    # Create test dates and merge with weather data
    test_dates = pd.to_datetime('20' + submission_template['id'].astype(str).str[:6], format='%Y%m%d')
    test_products = submission_template['id'].astype(str).str[6:].astype(int)
    
    # Create test features DataFrame
    test_df = pd.DataFrame({
        'Datum': test_dates,
        'weekday': test_dates.dt.dayofweek,
        'month': test_dates.dt.month,
        'day_of_month': test_dates.dt.day,
        'Warengruppe': test_products
    })
    
    # Merge with weather data
    test_df = pd.merge(test_df, weather_df, on='Datum', how='left')
    
    # Handle missing weather values
    test_df['Bewoelkung'].fillna(test_df['Bewoelkung'].mean(), inplace=True)
    test_df['Temperatur'].fillna(test_df['Temperatur'].mean(), inplace=True)
    test_df['Windgeschwindigkeit'].fillna(test_df['Windgeschwindigkeit'].mean(), inplace=True)
    test_df['Wettercode'].fillna(-1, inplace=True)
    
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
        'month', 'day_of_month',
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode'
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
    print("\nStarting neural network model training (v2 with weather data)...")
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
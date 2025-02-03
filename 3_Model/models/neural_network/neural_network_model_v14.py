"""
Neural Network Model for Bakery Sales Prediction - Version 14 based on v4

This version enhances the model by incorporating additional event data:
- Windjammer Parade: Annual sailing event
- Public Holidays: Official holidays in Schleswig-Holstein
- School Holidays: School vacation periods
- Weather data (from v2)
- Kieler Woche festival (from v3)

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
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MODEL_NAME = 'neural_network_model_v14'

# Create output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '3_Model', 'output', 'neural_network', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, '3_Model', 'visualizations', 'neural_network', MODEL_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Product group dictionary for readable names
WARENGRUPPEN = {
    1: "Brot",
    2: "Brötchen",
    3: "Croissant",
    4: "Konditorei",
    5: "Kuchen",
    6: "Saisonbrot"
}

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

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    return mape

def evaluate_model_performance(model, X_val_scaled, y_val, val_data):
    """Evaluate model performance with MAPE for overall and per product group"""
    print("\nEvaluating model performance...")
    
    # Get predictions
    val_predictions = model.predict(X_val_scaled, verbose='auto')
    
    # Calculate overall MAPE
    overall_mape = calculate_mape(y_val, val_predictions)
    print("\n=== MAPE Results ===")
    print(f"Overall MAPE: {overall_mape:.2f}%")
    print("\nMAPE by Product Group:")
    print("-" * 40)
    
    # Calculate MAPE per product group
    mape_by_group = {}
    val_data['predictions'] = val_predictions
    
    # Create detailed results for CSV
    detailed_results = []
    
    for group in sorted(val_data['Warengruppe'].unique()):
        group_mask = val_data['Warengruppe'] == group
        group_true = val_data.loc[group_mask, 'Umsatz']
        group_pred = val_data.loc[group_mask, 'predictions']
        group_mape = calculate_mape(group_true, group_pred)
        group_name = WARENGRUPPEN.get(group, f"Group {group}")
        mape_by_group[group_name] = group_mape
        
        # Calculate additional metrics
        group_mae = np.mean(np.abs(group_true - group_pred))
        group_rmse = np.sqrt(np.mean((group_true - group_pred) ** 2))
        
        print(f"{group_name:12} MAPE: {group_mape:6.2f}%")
        
        detailed_results.append({
            'Product_Group': group_name,
            'MAPE': group_mape,
            'MAE': group_mae,
            'RMSE': group_rmse,
            'Sample_Size': len(group_true)
        })
    
    print("-" * 40)
    print(f"{'Overall':12} MAPE: {overall_mape:6.2f}%")
    
    # Add overall metrics to detailed results
    detailed_results.append({
        'Product_Group': 'Overall',
        'MAPE': overall_mape,
        'MAE': np.mean(np.abs(y_val - val_predictions)),
        'RMSE': np.sqrt(np.mean((y_val - val_predictions) ** 2)),
        'Sample_Size': len(y_val)
    })
    
    # Save detailed results to CSV
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'mape_results.csv'), index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    
    # Bar plot
    bars = plt.bar(range(len(mape_by_group)), list(mape_by_group.values()))
    plt.axhline(y=overall_mape, color='r', linestyle='--', label='Overall MAPE')
    
    # Customize the plot
    plt.title('MAPE by Product Group', pad=20)
    plt.xlabel('Product Group')
    plt.ylabel('MAPE (%)')
    plt.xticks(range(len(mape_by_group)), list(mape_by_group.keys()), rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save visualization
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'mape_by_product_group.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return overall_mape, mape_by_group

def load_and_prepare_data():
    """Load and prepare data for the neural network model"""
    print("\nLoading and preparing data...")
    
    # Load all data files
    train_data_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    weather_data_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_data_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
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
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag'  # Added new event features
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
    
    val_data = df[val_mask].copy()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val, val_data

def train_model(X_train, X_val, y_train, y_val, val_data):
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
        patience=20,  # Increased patience
        restore_best_weights=True,
        verbose=1
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,  # Increased patience
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
        epochs=150,  # Increased epochs
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose='auto'
    )
    
    # Evaluate model performance with MAPE
    overall_mape, mape_by_group = evaluate_model_performance(model, X_val_scaled, y_val, val_data)
    
    return model, scaler, history, overall_mape, mape_by_group

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
    sample_submission_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv')
    weather_data_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'wetter.csv')
    kiwo_data_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
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
        'weekday': test_dates.dt.dayofweek,
        'month': test_dates.dt.month,
        'day_of_month': test_dates.dt.day,
        'Warengruppe': test_products
    })
    
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
        'month', 'day_of_month',
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
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
    print("\nStarting neural network model training (v4 with additional event data)...")
    X_train, X_val, y_train, y_val, val_data = load_and_prepare_data()
    
    print("\nTraining neural network model...")
    model, scaler, history, overall_mape, mape_by_group = train_model(X_train, X_val, y_train, y_val, val_data)
    
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nGenerating predictions...")
    generate_predictions(model, scaler)
    
    print(f"\nModel training and predictions complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main() 
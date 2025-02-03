"""
Neural Network Model for Bakery Sales Prediction - Version 16 based on v14

This version enhances the model with:
1. Improved seasonal product handling
2. Advanced feature engineering
3. Enhanced model architecture
4. Robust data preprocessing
5. Optimized training strategy

Features include:
- Product availability patterns
- Rolling statistics and lag features
- Cyclical time encoding
- Weather and event interaction features
- Outlier handling
- Enhanced architecture with residual connections
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

print("Keras version:", keras.__version__)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MODEL_NAME = 'neural_network_model_v16'

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
    
    # Input layer
    inputs = layers.Input(shape=(input_shape,))
    x = layers.BatchNormalization()(inputs)
    
    # First block
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Second block
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Third block
    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)
    
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
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
    val_predictions = val_predictions.flatten()  # Ensure 1D array
    
    # Calculate overall MAPE
    overall_mape = calculate_mape(y_val, val_predictions)
    print("\n=== MAPE Results ===")
    print(f"Overall MAPE: {overall_mape:.2f}%")
    print("\nMAPE by Product Group:")
    print("-" * 40)
    
    # Calculate MAPE per product group
    mape_by_group = {}
    val_data = val_data.copy()
    val_data['predictions'] = val_predictions
    
    # Create detailed results for CSV
    detailed_results = []
    
    for group in sorted(val_data['Warengruppe'].unique()):
        group_mask = val_data['Warengruppe'] == group
        group_true = val_data.loc[group_mask, 'Umsatz'].values  # Convert to numpy array
        group_pred = val_data.loc[group_mask, 'predictions'].values  # Convert to numpy array
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
    y_val_np = np.array(y_val)  # Convert to numpy array
    val_predictions_np = np.array(val_predictions)  # Convert to numpy array
    
    detailed_results.append({
        'Product_Group': 'Overall',
        'MAPE': overall_mape,
        'MAE': np.mean(np.abs(y_val_np - val_predictions_np)),
        'RMSE': np.sqrt(np.mean((y_val_np - val_predictions_np) ** 2)),
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
    
    # Add cyclical features
    df = add_cyclical_features(df)
    
    # Add rolling statistics features
    df = add_rolling_features(df)
    
    # Add availability patterns
    df = add_availability_patterns(df)
    
    # Create dummy variables for weekday and Warengruppe
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday')
    product_dummies = pd.get_dummies(df['Warengruppe'], prefix='product')
    
    # Split into training and validation sets based on date
    split_date = pd.Timestamp('2017-08-01')
    train_mask = df['Datum'] < split_date
    val_mask = df['Datum'] >= split_date
    
    # Create feature matrix X and target variable y
    feature_columns = [
        'month', 'day_of_month',
        'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag',
        'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
        'sales_rolling_mean_7d',
        'availability_ratio'
    ]
    
    X = pd.concat([
        df[feature_columns],
        weekday_dummies,
        product_dummies
    ], axis=1)
    
    y = df['Umsatz']
    
    # Split the data
    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]
    val_data = df[val_mask]
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    return X_train, X_val, y_train, y_val, val_data

def add_cyclical_features(df):
    """Add cyclical encoding for temporal features"""
    # Encode month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Encode day of week
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/7)
    
    return df

def add_rolling_features(df):
    """Add rolling statistics features"""
    # Sort by date and product group
    df = df.sort_values(['Datum', 'Warengruppe'])
    
    # Group by product
    for group in df['Warengruppe'].unique():
        mask = df['Warengruppe'] == group
        group_data = df[mask].copy()
        
        # Calculate rolling mean only
        rolling_7d = group_data['Umsatz'].rolling(window=7, min_periods=1)
        df.loc[mask, 'sales_rolling_mean_7d'] = rolling_7d.mean()
    
    return df

def add_availability_patterns(df):
    """Add product availability patterns based on historical data"""
    # Calculate monthly availability ratio
    availability = df.groupby(['month', 'Warengruppe'])['Umsatz'].agg(
        days_with_sales=('count')
    ).reset_index()
    
    monthly_days = df.groupby('month')['Datum'].nunique().reset_index()
    availability = availability.merge(monthly_days, on='month')
    availability['availability_ratio'] = availability['days_with_sales'] / availability['Datum']
    
    # Merge back to original dataframe
    df = df.merge(
        availability[['month', 'Warengruppe', 'availability_ratio']], 
        on=['month', 'Warengruppe'],
        how='left'
    )
    
    return df

def train_model(X_train, X_val, y_train, y_val, val_data):
    """Train the neural network model with improved training strategy"""
    print("Scaling features...")
    
    # Use RobustScaler for better outlier handling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create model
    model = create_neural_network(X_train.shape[1])
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    evaluate_model_performance(model, X_val_scaled, y_val, val_data)
    
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
    
    # Add cyclical features
    test_df = add_cyclical_features(test_df)
    
    # Add rolling statistics features
    test_df = add_rolling_features(test_df)
    
    # Add availability patterns
    test_df = add_availability_patterns(test_df)
    
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
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag',
        'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
        'sales_rolling_mean_7d',
        'availability_ratio'
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
    print("\nStarting neural network model training (v16 with additional event data)...")
    X_train, X_val, y_train, y_val, val_data = load_and_prepare_data()
    
    print("\nTraining neural network model...")
    model, scaler, history = train_model(X_train, X_val, y_train, y_val, val_data)
    
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nGenerating predictions...")
    generate_predictions(model, scaler)
    
    print(f"\nModel training and predictions complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main()
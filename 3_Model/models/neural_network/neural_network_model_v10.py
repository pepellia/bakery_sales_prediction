"""
Neural Network Model for Bakery Sales Prediction - Version 10

- Simplified architecture (32-16-1) as simpler models performed better
- Added Fourier features for better temporal pattern capture
- Removed weather features (shown to have minimal impact)
- Kept important event features with L1/L2 regularization
- Balanced feature engineering over model complexity

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
MODEL_NAME = 'neural_network_model_v10'

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

def create_neural_network(input_shape):
    """Create and compile the neural network model"""
    print(f"\nCreating neural network with input shape: {input_shape}")
    
    # Define custom MAPE metric
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + 1e-7))) * 100
    
    # Define regularization
    reg = regularizers.l1_l2(l1=1e-5, l2=1e-4)
    
    model = keras.Sequential([
        # Input and normalization
        layers.InputLayer(input_shape=(input_shape,)),
        layers.BatchNormalization(),
        
        # First dense block - simplified architecture
        layers.Dense(32, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.2),
        
        # Second dense block
        layers.Dense(16, activation='relu', kernel_regularizer=reg),
        layers.Dropout(0.1),
        
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
    kiwo_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv')
    windjammer_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv')
    school_holidays_path = os.path.join(SCRIPT_DIR, '..', '..', '..', '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv')
    
    # Check if all required files exist
    required_files = [
        (train_path, "Training data"),
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
    kiwo_df = pd.read_csv(kiwo_path)
    windjammer_df = pd.read_csv(windjammer_path)
    holidays_df = pd.read_csv(holidays_path, sep=';')
    school_holidays_df = pd.read_csv(school_holidays_path, sep=';')
    
    # Convert date columns
    df['Datum'] = pd.to_datetime(df['Datum'])
    kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    holidays_df['Datum'] = pd.to_datetime(holidays_df['Datum'], format='%d.%m.%Y')
    school_holidays_df['Datum'] = pd.to_datetime(school_holidays_df['Datum'], format='%d.%m.%Y')
    
    # Merge all data
    df = pd.merge(df, kiwo_df, on='Datum', how='left')
    df = pd.merge(df, windjammer_df, on='Datum', how='left')
    df = pd.merge(df, holidays_df[['Datum', 'Feiertag']], on='Datum', how='left')
    df = pd.merge(df, school_holidays_df[['Datum', 'Ferientag']], on='Datum', how='left')
    
    if df.empty:
        sys.exit("Error: Training dataframe is empty after merges.")
    
    # Fill missing values for event features
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
    
    # Create Fourier features
    fourier_features = create_fourier_features(df['Datum'])
    
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
        # Event features with proven impact
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag', 'is_silvester'
    ]
    
    X = pd.concat([
        df[feature_columns],
        fourier_features,  # Add Fourier features
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
    
    # Plot training history
    plot_training_history(history)
    
    return model  # Return only the model

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

def prepare_test_data():
    """Prepare test data for predictions"""
    print("\nPreparing test data...")
    
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
        # Event features with proven impact
        'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag', 'is_silvester'
    ]
    
    X_test = pd.concat([
        test_df[feature_columns],
        fourier_features,  # Add Fourier features
        weekday_dummies,
        product_dummies
    ], axis=1)
    
    return X_test, test_df

def main():
    """Main function to run the neural network model"""
    print("\nStarting neural network model training (v10 with Fourier features)...")
    X_train, X_val, y_train, y_val = load_and_prepare_data()
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train the model
    model = train_model(X_train_scaled, X_val_scaled, y_train, y_val)
    
    # Prepare test data
    X_test, test_df = prepare_test_data()
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
    val_predictions = model.predict(X_val_scaled)
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
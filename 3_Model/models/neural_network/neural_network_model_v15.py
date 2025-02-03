"""
Neural Network Model for Bakery Sales Prediction - Version 15 based on v14

This version implements a multi-output approach:
- Separate output neurons for each product group
- Shared feature extraction layers
- Specialized branches for each product group
- Enhanced model architecture for better per-group predictions
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
MODEL_NAME = 'neural_network_model_v15'

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
    """Create and compile the neural network model with multiple outputs"""
    print(f"\nCreating neural network with input shape: {input_shape}")
    
    # Input layer
    inputs = layers.Input(shape=(input_shape,))
    
    # Shared layers for feature extraction
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Separate branches for each product group
    outputs = []
    for i in range(6):
        branch = layers.Dense(32, activation='relu', name=f'dense_1_group_{i+1}')(x)
        branch = layers.Dropout(0.2, name=f'dropout_1_group_{i+1}')(branch)
        branch = layers.Dense(16, activation='relu', name=f'dense_2_group_{i+1}')(branch)
        branch = layers.Dense(1, name=f'output_group_{i+1}')(branch)
        outputs.append(branch)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Define metrics for each output
    metrics_dict = {}
    for i in range(6):
        metrics_dict[f'output_group_{i+1}'] = ['mae']
    
    # Compile model with metrics for each output
    model.compile(
        optimizer='adam',
        loss=['mse'] * 6,  # MSE loss for each output
        loss_weights=[1.0] * 6,  # Equal weights for all outputs
        metrics=metrics_dict  # MAE metric for each output
    )
    
    model.summary()
    return model

def prepare_multi_output_data(y, warengruppen_data, X):
    """Prepare target data for multi-output model by replicating features for each group"""
    X_multi = []  # List to store replicated features for each group
    y_multi = []  # List to store targets for each group
    
    for group in range(1, 7):
        # Create mask for current product group
        group_mask = warengruppen_data == group
        
        # Store targets for this group
        y_group = y[group_mask].values
        y_multi.append(y_group)
        
        # Store features for this group
        X_group = X[group_mask]
        X_multi.append(X_group)
    
    # Find minimum length across all groups
    min_length = min(len(y_group) for y_group in y_multi)
    
    # Trim all groups to minimum length
    X_multi = [X_group[:min_length] for X_group in X_multi]
    y_multi = [y_group[:min_length] for y_group in y_multi]
    
    # Stack all feature matrices
    X_stacked = np.vstack(X_multi)
    
    # Create output arrays with same length as X_stacked
    y_stacked = []
    samples_per_group = X_stacked.shape[0] // 6
    
    for group_idx in range(6):
        # Create array of zeros
        y_out = np.zeros(X_stacked.shape[0])
        # Fill in the values for this group at the correct positions
        y_out[group_idx * samples_per_group:(group_idx + 1) * samples_per_group] = y_multi[group_idx]
        y_stacked.append(y_out)
    
    return X_stacked, y_stacked

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
    
    # Create dummy variables for weekday
    weekday_dummies = pd.get_dummies(df['weekday'], prefix='weekday')
    
    # Ensure all weekday columns exist (0-6)
    for i in range(7):
        col = f'weekday_{i}'
        if col not in weekday_dummies.columns:
            weekday_dummies[col] = 0
    
    # Sort weekday columns to ensure consistent order
    weekday_dummies = weekday_dummies.reindex(sorted(weekday_dummies.columns), axis=1)
    
    # Split data into training and validation sets
    train_mask = df['Datum'] < pd.Timestamp('2017-08-01')
    
    # Combine all features
    feature_columns = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode',
                      'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag',
                      'month', 'year', 'day_of_month']
    
    # Create feature matrices
    X = pd.concat([df[feature_columns], weekday_dummies], axis=1)
    
    # Split into train and validation sets
    X_train = X[train_mask]
    X_val = X[~train_mask]
    y_train = df.loc[train_mask, 'Umsatz']
    y_val = df.loc[~train_mask, 'Umsatz']
    
    # Store Warengruppe information
    train_groups = df.loc[train_mask, 'Warengruppe']
    val_groups = df.loc[~train_mask, 'Warengruppe']
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val, train_groups, val_groups

def train_model(X_train, X_val, y_train, y_val, train_groups, val_groups):
    """Train the neural network model"""
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Prepare multi-output targets using Warengruppe information
    X_train_stacked, y_train_multi = prepare_multi_output_data(y_train, train_groups, X_train_scaled)
    X_val_stacked, y_val_multi = prepare_multi_output_data(y_val, val_groups, X_val_scaled)
    
    print("Training model...")
    model = create_neural_network(X_train.shape[1])
    
    # Add early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    # Add learning rate reduction on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=8,
        min_lr=1e-6,
        verbose=0
    )
    
    history = model.fit(
        X_train_stacked,
        y_train_multi,
        validation_data=(X_val_stacked, y_val_multi),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=2  # Show one line per epoch
    )
    
    # Evaluate model performance with MAPE
    overall_mape, mape_by_group = evaluate_model_performance(model, X_val_scaled, y_val, val_groups)
    
    return model, scaler, history, overall_mape, mape_by_group

def evaluate_model_performance(model, X_val_scaled, y_val, val_groups):
    """Evaluate model performance with MAPE for overall and per product group"""
    print("\nEvaluating model performance...")
    
    # Get predictions group by group
    val_predictions = np.zeros_like(y_val)
    
    for group in range(1, 7):
        # Get data for this group
        group_mask = val_groups == group
        X_group = X_val_scaled[group_mask]
        
        # Get predictions for this group
        predictions = model.predict(X_group, verbose=0)
        val_predictions[group_mask] = predictions[group - 1].flatten()
    
    # Calculate overall MAPE
    overall_mape = calculate_mape(y_val, val_predictions)
    print(f"\nOverall MAPE: {overall_mape:.2f}%")
    
    # Calculate MAPE per product group
    mape_by_group = {}
    
    for group in range(1, 7):
        group_mask = val_groups == group
        group_true = y_val[group_mask]
        group_pred = val_predictions[group_mask]
        group_mape = calculate_mape(group_true, group_pred)
        group_name = WARENGRUPPEN.get(group, f"Group {group}")
        mape_by_group[group_name] = group_mape
        print(f"MAPE for {group_name}: {group_mape:.2f}%")
    
    # Plot MAPE by product group
    plt.figure(figsize=(10, 6))
    groups = list(mape_by_group.keys())
    mapes = list(mape_by_group.values())
    
    plt.bar(groups, mapes)
    plt.axhline(y=overall_mape, color='r', linestyle='--', label='Overall MAPE')
    plt.title('MAPE by Product Group')
    plt.xlabel('Product Group')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'mape_by_product_group.png'))
    plt.close()
    
    # Save MAPE results to file
    results_df = pd.DataFrame({
        'Product_Group': groups,
        'MAPE': mapes
    })
    results_df.loc[len(results_df)] = ['Overall', overall_mape]
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'mape_results.csv'), index=False)
    
    return overall_mape, mape_by_group

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    return mape

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
    
    # Plot MAE for each output
    plt.subplot(1, 2, 2)
    for i in range(1, 7):
        plt.plot(history.history[f'output_group_{i}_mae'], 
                label=f'Group {i} MAE')
    plt.title('Model MAE by Group')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'training_history.png'))
    plt.close()

def generate_predictions(trained_model, fitted_scaler):
    """Generate predictions for test data"""
    print("\nGenerating predictions...")
    
    # Load test data
    test_data = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'test.csv'))
    submission_template = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'sample_submission.csv'))
    
    # Extract date components from ID
    test_data['month'] = pd.to_numeric(test_data['id'].astype(str).str[4:6], errors='coerce').fillna(1).astype(int)
    test_data['day_of_month'] = pd.to_numeric(test_data['id'].astype(str).str[6:8], errors='coerce').fillna(1).astype(int)
    test_data['Warengruppe'] = pd.to_numeric(test_data['id'].astype(str).str[8:], errors='coerce').fillna(1).astype(int)
    
    # Prepare features
    X_test = pd.DataFrame({
        'month': test_data['month'],
        'day_of_month': test_data['day_of_month'],
        'Warengruppe': test_data['Warengruppe']
    })
    
    # Merge with weather data
    weather_df = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'wetter.csv'))
    weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
    weather_df['month'] = weather_df['Datum'].dt.month
    weather_df['day_of_month'] = weather_df['Datum'].dt.day
    X_test = pd.merge(X_test, weather_df[['month', 'day_of_month', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode']], 
                     on=['month', 'day_of_month'], how='left')
    
    # Merge with Kieler Woche data
    kiwo_df = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data', 'kiwo.csv'))
    kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
    kiwo_df['month'] = kiwo_df['Datum'].dt.month
    kiwo_df['day_of_month'] = kiwo_df['Datum'].dt.day
    X_test = pd.merge(X_test, kiwo_df[['month', 'day_of_month', 'KielerWoche']], 
                     on=['month', 'day_of_month'], how='left')
    
    # Merge with Windjammer data
    windjammer_df = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv'))
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    windjammer_df['month'] = windjammer_df['Datum'].dt.month
    windjammer_df['day_of_month'] = windjammer_df['Datum'].dt.day
    X_test = pd.merge(X_test, windjammer_df[['month', 'day_of_month', 'Windjammerparade']], 
                     on=['month', 'day_of_month'], how='left')
    
    # Merge with holidays data
    holidays_df = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'Feiertage-SH.csv'), sep=';')
    holidays_df['Datum'] = pd.to_datetime(holidays_df['Datum'], format='%d.%m.%Y')
    holidays_df['month'] = holidays_df['Datum'].dt.month
    holidays_df['day_of_month'] = holidays_df['Datum'].dt.day
    X_test = pd.merge(X_test, holidays_df[['month', 'day_of_month', 'Feiertag']], 
                     on=['month', 'day_of_month'], how='left')
    
    # Merge with school holidays data
    school_holidays_df = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'Schulferientage-SH.csv'), sep=';')
    school_holidays_df['Datum'] = pd.to_datetime(school_holidays_df['Datum'], format='%d.%m.%Y')
    school_holidays_df['month'] = school_holidays_df['Datum'].dt.month
    school_holidays_df['day_of_month'] = school_holidays_df['Datum'].dt.day
    X_test = pd.merge(X_test, school_holidays_df[['month', 'day_of_month', 'Ferientag']], 
                     on=['month', 'day_of_month'], how='left')
    
    # Fill missing values in test data
    numeric_cols = ['Bewoelkung', 'Temperatur', 'Windgeschwindigkeit']
    for col in numeric_cols:
        mean_val = X_test[col].mean()
        X_test = X_test.assign(**{col: X_test[col].fillna(mean_val)})
    
    categorical_cols = ['Wettercode', 'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag']
    for col in categorical_cols:
        X_test = X_test.assign(**{col: X_test[col].fillna(0)})
    
    # Reindex to ensure all columns are present
    feature_columns = fitted_scaler.feature_names_in_
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)
    
    # Create dummy variables for product groups
    product_dummies = pd.get_dummies(X_test['Warengruppe'], prefix='product')
    
    # Ensure all product columns exist (1-6)
    for i in range(1, 7):
        col = f'product_{i}'
        if col not in product_dummies.columns:
            product_dummies[col] = 0
    
    # Combine features
    X_test = pd.concat([
        X_test[['month', 'day_of_month', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'Wettercode', 
                'KielerWoche', 'Windjammerparade', 'Feiertag', 'Ferientag']],
        product_dummies
    ], axis=1)
    
    # Scale features and generate predictions
    X_test_scaled = fitted_scaler.transform(X_test)
    predictions_multi = trained_model.predict(X_test_scaled, verbose=0)
    
    print(f"Shape of test data: {X_test.shape}")
    print(f"Shape of predictions_multi: {[p.shape for p in predictions_multi]}")
    print(f"Shape of submission template: {submission_template.shape}")
    
    # Initialize predictions array with the same length as submission template
    predictions = np.zeros(len(submission_template))
    
    # Map predictions back to the correct rows based on Warengruppe
    for group in range(1, 7):
        group_mask = (test_data['Warengruppe'] == group).values  # Convert to numpy array
        group_indices = np.where(group_mask)[0]  # Get indices where mask is True
        if len(group_indices) > 0:
            predictions[group_indices] = predictions_multi[group - 1][group_indices]
    
    # Create submission file
    submission_template['Umsatz'] = np.maximum(0, predictions)  # Ensure non-negative predictions
    submission_file = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission_template.to_csv(submission_file, index=False)
    print(f"Predictions saved to {submission_file}")

def main():
    """Main function to run the neural network model"""
    print("\nStarting neural network model training (v15 with multi-output approach)...")
    X_train, X_val, y_train, y_val, train_groups, val_groups = load_and_prepare_data()
    
    print("\nTraining neural network model...")
    model, scaler, history, overall_mape, mape_by_group = train_model(
        X_train, X_val, y_train, y_val, train_groups, val_groups
    )
    
    print("\nPlotting training history...")
    plot_training_history(history)
    
    print("\nGenerating predictions...")
    generate_predictions(model, scaler)
    
    print(f"\nModel training and predictions complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main()
# Required libraries
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Tuple, Any
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import signal
from itertools import combinations
from datetime import datetime
import glob

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Set up signal handling
def signal_handler(signum, frame):
    print("\nReceived interrupt signal. Cleaning up...")
    plt.close('all')  # Close all plot windows
    sys.exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_and_preprocess_data() -> pd.DataFrame:
    """
    Load and preprocess all available data sources into a single DataFrame.
    """
    print("Loading main training data...")
    # Load main training data
    train_data = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation/input/competition_data/train.csv'))
    train_data['date'] = pd.to_datetime(train_data['Datum'])
    
    print("Loading weather codes and data...")
    # Load weather codes
    weather_codes = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation/input/compiled_data/wettercode.csv'), 
                              sep=';', header=None, names=['code', 'description'])
    weather_codes['code'] = weather_codes['code'].astype(int)
    
    # Load weather data
    weather_data = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation/input/competition_data/wetter.csv'))
    weather_data['date'] = pd.to_datetime(weather_data['Datum'])
    
    print("Loading events data...")
    # Load Kieler Woche data
    kiwo_data = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation/input/competition_data/kiwo.csv'))
    kiwo_data['date'] = pd.to_datetime(kiwo_data['Datum'])
    
    # Load Windjammer data
    windjammer_data = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation/input/compiled_data/windjammer.csv'))
    windjammer_data['date'] = pd.to_datetime(windjammer_data['Datum'])
    
    # Load public holidays data
    print("Loading public holidays data...")
    holidays_data = pd.read_csv(os.path.join(PROJECT_ROOT, '0_DataPreparation/input/compiled_data/Feiertage-SH.csv'),
                              sep=';')
    holidays_data['date'] = pd.to_datetime(holidays_data['Datum'], format='%d.%m.%Y')
    holidays_data = holidays_data.rename(columns={'Feiertag': 'is_public_holiday'})
    
    print("Loading school holiday data...")
    # Load school holiday data for all states
    holiday_files = glob.glob(os.path.join(PROJECT_ROOT, '0_DataPreparation/input/compiled_data/school_holidays/*.csv'))
    holiday_dfs = {}
    for file in holiday_files:
        state_name = os.path.basename(file).replace('school_holidays_', '').replace('.csv', '')
        df = pd.read_csv(file)
        df['date'] = pd.to_datetime(df['date'])
        holiday_dfs[state_name] = df
    
    print("Creating temporal features...")
    # Create temporal features
    temporal_features = pd.DataFrame({
        'date': train_data['date'].unique()
    })
    temporal_features['date'] = pd.to_datetime(temporal_features['date'])
    temporal_features['year'] = temporal_features['date'].dt.year
    temporal_features['month'] = temporal_features['date'].dt.month
    temporal_features['day'] = temporal_features['date'].dt.day
    temporal_features['day_of_week'] = temporal_features['date'].dt.dayofweek
    temporal_features['week_of_year'] = temporal_features['date'].dt.isocalendar().week
    temporal_features['is_weekend'] = temporal_features['day_of_week'].isin([5, 6]).astype(int)
    temporal_features['is_month_start'] = temporal_features['date'].dt.is_month_start.astype(int)
    temporal_features['is_month_end'] = temporal_features['date'].dt.is_month_end.astype(int)
    temporal_features['quarter'] = temporal_features['date'].dt.quarter
    
    # Add Silvester flag
    temporal_features['is_silvester'] = ((temporal_features['month'] == 12) & 
                                       (temporal_features['day'] == 31)).astype(int)
    
    # Add season features using one-hot encoding
    temporal_features['is_winter'] = temporal_features['month'].isin([12, 1, 2]).astype(int)
    temporal_features['is_spring'] = temporal_features['month'].isin([3, 4, 5]).astype(int)
    temporal_features['is_summer'] = temporal_features['month'].isin([6, 7, 8]).astype(int)
    temporal_features['is_fall'] = temporal_features['month'].isin([9, 10, 11]).astype(int)
    
    # Add week of month
    temporal_features['week_of_month'] = temporal_features['date'].dt.day.apply(
        lambda x: (x - 1) // 7 + 1
    )
    
    # Add is_payday (assuming typical German payday around 25th-last day of month)
    temporal_features['is_payday'] = ((temporal_features['day'] >= 25) | 
                                    (temporal_features['day'] <= 3)).astype(int)
    
    # Add is_bridge_day (day between holiday and weekend)
    holidays_dates = set(holidays_data['date'].dt.date)
    def is_bridge_day(row):
        date = row['date'].date()
        day_of_week = row['day_of_week']
        # Check if it's a workday (Monday-Friday)
        if day_of_week in [5, 6]:  # Weekend
            return 0
        # Check adjacent days
        prev_day = date - pd.Timedelta(days=1)
        next_day = date + pd.Timedelta(days=1)
        # If day is between a holiday and weekend
        is_prev_holiday = prev_day in holidays_dates
        is_next_holiday = next_day in holidays_dates
        is_near_weekend = day_of_week in [0, 4]  # Monday or Friday
        return 1 if (is_prev_holiday or is_next_holiday) and is_near_weekend else 0
    
    temporal_features['is_bridge_day'] = temporal_features.apply(is_bridge_day, axis=1)
    
    print("Processing weather features...")
    # Merge weather data with codes to get descriptions
    weather_data = pd.merge(weather_data, weather_codes[['code', 'description']], 
                           left_on='Wettercode', right_on='code', how='left')
    
    # Create weather features
    weather_features = weather_data.copy()
    
    # Calculate feels_like_temperature using temperature, wind, and humidity
    # Using a simplified version of the heat index formula
    weather_features['feels_like_temperature'] = weather_features.apply(
        lambda row: row['Temperatur'] - (0.2 * row['Windgeschwindigkeit']), axis=1
    )
    
    # Create is_good_weather feature
    weather_features['is_good_weather'] = weather_features.apply(
        lambda row: 1 if (
            row['Temperatur'] >= 15 and  # Comfortable temperature
            row['Bewoelkung'] <= 5 and   # Less than 50% cloud cover
            'Regen' not in str(row['description']) and
            'Schnee' not in str(row['description'])
        ) else 0, 
        axis=1
    )
    
    # One-hot encode weather descriptions (these are more meaningful than codes)
    description_dummies = pd.get_dummies(weather_features['description'], prefix='weather')
    weather_features = pd.concat([
        weather_features[['date', 'Bewoelkung', 'Temperatur', 'Windgeschwindigkeit', 'feels_like_temperature', 'is_good_weather']], 
        description_dummies
    ], axis=1)
    
    print("Merging all features...")
    # Merge all features
    final_df = train_data.copy()
    
    # Add temporal features
    final_df = pd.merge(final_df, temporal_features, on='date', how='left')
    
    # Add weather features
    final_df = pd.merge(final_df, weather_features, on='date', how='left')
    
    # Add Kieler Woche features
    final_df = pd.merge(final_df, kiwo_data.drop('Datum', axis=1), 
                       left_on='date', right_on='date', how='left')
    
    # Add Windjammer features
    final_df = pd.merge(final_df, windjammer_data.drop('Datum', axis=1), 
                       left_on='date', right_on='date', how='left')
    
    # Add public holidays
    final_df = pd.merge(final_df, 
                       holidays_data[['date', 'is_public_holiday']],
                       on='date', how='left')
    
    # Add holiday features from all states
    for state, holiday_df in holiday_dfs.items():
        print(f"  - Adding holiday data for {state}")
        final_df = pd.merge(final_df, 
                           holiday_df[['date', 'is_school_holiday']].rename(
                               columns={'is_school_holiday': f'is_school_holiday_{state}'}
                           ),
                           on='date', how='left')
    
    print("Handling missing values...")
    # Fill missing values
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    final_df[numeric_cols] = final_df[numeric_cols].fillna(0)
    
    # Fill missing categorical values
    categorical_cols = final_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['date', 'Datum']:  # Skip date columns
            final_df[col] = final_df[col].fillna(0)
    
    print(f"\nFinal dataset shape: {final_df.shape}")
    print("\nFeature columns:")
    feature_cols = [col for col in sorted(final_df.columns) if col not in ['date', 'Datum', 'id', 'Umsatz']]
    print(f"Total features: {len(feature_cols)}")
    for col in feature_cols:
        print(f"- {col}")
    
    return final_df

def select_features(df: pd.DataFrame, target_col: str = 'Umsatz', 
                   n_features: int = 20) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """
    Select the most important features using f_regression.
    First analyze each product group separately, then combine results.
    
    Args:
        df: Input DataFrame with features
        target_col: Name of the target column
        n_features: Number of top features to select
        
    Returns:
        Tuple containing:
        - DataFrame with selected features
        - List of selected feature names
        - DataFrame with feature scores
    """
    try:
        print(f"Starting feature selection process for top {n_features} features...")
        
        # Identify columns to exclude (date columns, IDs, and target)
        exclude_cols = ['date', 'Datum', 'id', target_col]
        
        # Get all feature columns (including dummy variables)
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and col != 'Warengruppe']  # Handle Warengruppe separately
        
        print(f"Total number of features: {len(feature_cols)}")
        print("Features to be analyzed:", feature_cols)
        
        # Initialize feature scores dictionary
        feature_scores_dict = {col: 0.0 for col in feature_cols}
        feature_scores_dict['Warengruppe'] = 0.0  # Add Warengruppe
        
        # Analyze each product group separately
        product_groups = df['Warengruppe'].unique()
        print(f"\nAnalyzing {len(product_groups)} product groups...")
        
        for group in product_groups:
            print(f"  Processing product group {group}")
            group_df = df[df['Warengruppe'] == group]
            
            # Ensure we're working with pandas objects
            X: pd.DataFrame = pd.DataFrame(group_df[feature_cols])
            y: pd.Series = pd.Series(group_df[target_col])
            
            # Handle any NaN values before scaling
            X = X.fillna(0)
            y = y.fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate f_regression scores for this group
            f_scores, _ = f_regression(X_scaled, y)
            
            # Update overall scores (use maximum score across groups)
            for col, score in zip(feature_cols, f_scores):
                feature_scores_dict[col] = max(feature_scores_dict[col], score)
        
        # Calculate Warengruppe importance using overall sales variation
        group_means = df.groupby('Warengruppe')[target_col].mean()
        group_std = float(group_means.std())
        feature_scores_dict['Warengruppe'] = group_std * 10  # Scale up to make it comparable, but not too dominant
        
        # Convert to DataFrame for sorting
        feature_scores_df = pd.DataFrame({
            'Feature': list(feature_scores_dict.keys()),
            'Score': list(feature_scores_dict.values())
        })
        
        # Sort and select top features
        feature_scores_df = feature_scores_df.sort_values('Score', ascending=False)
        
        # Explicitly convert to List[str]
        top_features = feature_scores_df['Feature'].head(n_features)
        selected_features_list = [str(feature) for feature in top_features]
        
        # Create selected features DataFrame
        X_selected = pd.DataFrame(df[selected_features_list])
        
        return X_selected, selected_features_list, feature_scores_df
        
    except KeyboardInterrupt:
        print("\nReceived interrupt during feature selection. Cleaning up...")
        plt.close('all')
        sys.exit(1)

def main() -> None:
    try:
        print("=== Starting Feature Selection Analysis ===")
        print("\nStep 1: Data Loading and Preprocessing")
        print("-" * 40)
        df = load_and_preprocess_data()
        
        print("\nStep 2: Feature Selection and Importance Analysis")
        print("-" * 40)
        result: Tuple[pd.DataFrame, List[str], pd.DataFrame] = select_features(df)
        X_selected, selected_features, feature_scores = result
        
        print("\nTop selected features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
        
        print("\nStep 3: Visualization")
        print("-" * 40)
        # Create bar plot using the already calculated feature scores
        print("Creating feature importance plot...")
        plt.figure(figsize=(12, 8))
        top_n_features = feature_scores.head(20)  # Get top 20 features
        sns.barplot(data=top_n_features, x='Score', y='Feature')
        plt.title('Top 20 Feature Importance Scores')
        plt.tight_layout()
        
        print("\nStep 4: Saving Results")
        print("-" * 40)
        # Create output directory
        output_dir = os.path.join(SCRIPT_DIR, 'output', 'feature_selection')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

        # Save selected features to file
        features_path = os.path.join(output_dir, 'selected_features_v3.txt')
        with open(features_path, 'w') as f:
            f.write('\n'.join(selected_features))
        print(f"Saved selected features to: {features_path}")

        # Save all features with scores to file
        all_features_path = os.path.join(output_dir, 'all_features_importance_v3.txt')
        with open(all_features_path, 'w') as f:
            f.write("Feature,Score\n")  # Header
            for _, row in feature_scores.iterrows():
                f.write(f"{row['Feature']},{row['Score']:.6f}\n")
        print(f"Saved all features importance to: {all_features_path}")

        # Save plot
        plot_path = os.path.join(output_dir, 'feature_importance_v3.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved feature importance plot to: {plot_path}")

        print("\n=== Analysis Complete ===")
        print("Results saved to:")
        print(f"- Feature importance plot: {plot_path}")
        print(f"- Selected features list: {features_path}")
        print(f"- All features importance: {all_features_path}")
        
    except KeyboardInterrupt:
        print("\nReceived interrupt. Cleaning up...")
        plt.close('all')
        sys.exit(1)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        plt.close('all')
        sys.exit(1)

if __name__ == '__main__':
    main()

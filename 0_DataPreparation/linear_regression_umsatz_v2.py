# Required libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, '0_DataPreparation'))
from config import (TRAIN_PATH, VIZ_DIR, WARENGRUPPEN, get_warengruppe_name,
                   WEATHER_PATH, KIWO_PATH, WINDJAMMER_PATH)

def load_and_merge_data():
    """Load and merge all relevant datasets"""
    print("Loading and merging data...")
    
    # Load main sales data
    df = pd.read_csv(TRAIN_PATH)
    df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Load weather data
    weather_df = pd.read_csv(WEATHER_PATH)
    weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
    
    # Load event data
    kiwo_df = pd.read_csv(KIWO_PATH)
    kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
    kiwo_df['is_kiwo'] = 1
    
    windjammer_df = pd.read_csv(WINDJAMMER_PATH)
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    windjammer_df['is_windjammer'] = 1
    
    # Merge all datasets
    df = pd.merge(df, weather_df, on='Datum', how='left')
    df = pd.merge(df, kiwo_df[['Datum', 'is_kiwo']], on='Datum', how='left')
    df = pd.merge(df, windjammer_df[['Datum', 'is_windjammer']], on='Datum', how='left')
    
    # Fill missing values
    df['is_kiwo'] = df['is_kiwo'].fillna(0)
    df['is_windjammer'] = df['is_windjammer'].fillna(0)
    
    # Fill missing weather data with median values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df

def prepare_features(df):
    """Prepare enhanced feature set for the model"""
    print("Preparing features...")
    
    # Basic temporal features
    df['Jahr'] = df['Datum'].dt.year
    df['Monat'] = df['Datum'].dt.month
    df['Wochentag'] = df['Datum'].dt.dayofweek
    df['Tag_im_Monat'] = df['Datum'].dt.day
    df['Woche_im_Jahr'] = df['Datum'].dt.isocalendar().week
    df['is_weekend'] = df['Wochentag'].isin([5, 6]).astype(int)
    
    # Month position features
    df['month_position'] = pd.cut(df['Tag_im_Monat'], 
                                bins=[0, 10, 20, 31], 
                                labels=['start', 'middle', 'end'])
    month_pos_dummies = pd.get_dummies(df['month_position'], prefix='month_pos')
    df = pd.concat([df, month_pos_dummies], axis=1)
    
    # Season features
    df['season'] = pd.cut(df['Monat'], 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['winter', 'spring', 'summer', 'fall'])
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)
    
    # Product group features
    product_dummies = pd.get_dummies(df['Warengruppe'], prefix='Warengruppe')
    df = pd.concat([df, product_dummies], axis=1)
    
    # Weather features (assuming columns from weather data)
    if 'Temperatur' in df.columns:
        df['temp_category'] = pd.cut(df['Temperatur'], 
                                   bins=[-np.inf, 5, 15, 25, np.inf],
                                   labels=['cold', 'mild', 'warm', 'hot'])
        temp_dummies = pd.get_dummies(df['temp_category'], prefix='temp')
        df = pd.concat([df, temp_dummies], axis=1)
    
    return df

def train_model(X_train, y_train, X_test, y_test):
    """Train and evaluate the enhanced linear regression model"""
    print("\nTraining model...")
    
    # Handle any remaining missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_imputed, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_imputed)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return model, y_pred, rmse, r2

def create_visualizations(df, y_test, y_pred, model, feature_columns, viz_dir):
    """Create enhanced set of visualizations"""
    print("\nCreating visualizations...")
    
    # 1. Prediction vs Actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs Actual Values')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'prediction_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sales by Weekday
    plt.figure(figsize=(12, 6))
    weekday_avg = df.groupby('Wochentag')['Umsatz'].agg(['mean', 'std']).round(2)
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_avg.index = weekday_names
    weekday_avg['mean'].plot(kind='bar', yerr=weekday_avg['std'], capsize=5)
    plt.title('Average Sales by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sales_by_weekday.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Sales by Product Group
    plt.figure(figsize=(12, 6))
    product_avg = df.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std']).round(2)
    product_avg.index = [get_warengruppe_name(idx) for idx in product_avg.index]
    product_avg['mean'].plot(kind='bar', yerr=product_avg['std'], capsize=5)
    plt.title('Average Sales by Product Group')
    plt.xlabel('Product Group')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sales_by_product.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Sales Trend
    plt.figure(figsize=(15, 6))
    monthly_sales = df.groupby(df['Datum'].dt.to_period('M'))['Umsatz'].mean()
    monthly_sales.plot(kind='line', marker='o')
    plt.title('Average Monthly Sales')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'sales_trend.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Coefficient': model.coef_
    })
    feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance.head(15), x='Coefficient', y='Feature')
    plt.title('Top 15 Feature Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return feature_importance

def main():
    """Main function to run the enhanced linear regression analysis"""
    # Create output directories
    viz_dir = os.path.join(VIZ_DIR, 'linear_regression_v2')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load and merge data
    df = load_and_merge_data()
    
    # Prepare enhanced feature set
    df = prepare_features(df)
    
    # Select features for the model
    feature_columns = [col for col in df.columns if col not in 
                      ['Datum', 'Umsatz', 'Warengruppe', 'month_position', 
                       'season', 'temp_category', 'Temperatur']]
    
    X = df[feature_columns]
    y = df['Umsatz']
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model, y_pred, rmse, r2 = train_model(X_train, y_train, X_test, y_test)
    
    # Print model performance
    print("\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Create visualizations
    feature_importance = create_visualizations(df, y_test, y_pred, model, feature_columns, viz_dir)
    
    # Print feature importance summary
    print("\nTop 15 Most Important Features:")
    print(feature_importance[['Feature', 'Coefficient']].head(15))
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(viz_dir, f'model_results_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        f.write("Enhanced Linear Regression Model Results\n")
        f.write("=====================================\n\n")
        f.write(f"Model Performance:\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"R²: {r2:.4f}\n\n")
        f.write("Top 15 Most Important Features:\n")
        f.write(feature_importance[['Feature', 'Coefficient']].head(15).to_string())
    
    print(f"\nDetailed results saved to: {results_file}")
    print("\nVisualizations saved to:")
    print(f"1. {os.path.join(viz_dir, 'prediction_vs_actual.png')} - Prediction vs Actual Values")
    print(f"2. {os.path.join(viz_dir, 'sales_by_weekday.png')} - Sales by Weekday")
    print(f"3. {os.path.join(viz_dir, 'sales_by_product.png')} - Sales by Product Group")
    print(f"4. {os.path.join(viz_dir, 'sales_trend.png')} - Sales Trend")
    print(f"5. {os.path.join(viz_dir, 'feature_importance.png')} - Feature Importance")

if __name__ == "__main__":
    main()

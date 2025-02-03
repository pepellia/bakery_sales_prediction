import os
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better visualization
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'competition_data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '3_Model', 'analysis', 'visualizations', 'data_quality')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load all relevant datasets"""
    print("Loading datasets...")
    
    # Load main sales data
    sales_path = os.path.join(DATA_DIR, 'train.csv')
    sales_df = pd.read_csv(sales_path)
    sales_df['Datum'] = pd.to_datetime(sales_df['Datum'])
    
    # Load weather data
    weather_path = os.path.join(DATA_DIR, 'wetter.csv')
    weather_df = pd.read_csv(weather_path)
    weather_df['Datum'] = pd.to_datetime(weather_df['Datum'])
    
    # Convert numeric columns in weather data
    numeric_cols = ['Temperatur', 'Windgeschwindigkeit', 'Bewoelkung']
    for col in numeric_cols:
        weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')
    
    # Load weather codes
    weather_codes_path = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input', 'compiled_data', 'wettercode.csv')
    weather_codes = pd.read_csv(weather_codes_path, sep=';', header=None, names=['code', 'description'])
    
    return sales_df, weather_df, weather_codes

def analyze_missing_patterns(df, name, output_dir):
    """Analyze and visualize missing data patterns"""
    print(f"\nAnalyzing missing data patterns for {name}...")
    
    # Create missing value matrix
    plt.figure(figsize=(12, 6))
    msno.matrix(df)
    plt.title(f'Missing Value Matrix - {name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'missing_matrix_{name}.png'))
    plt.close()
    
    # Print missing value statistics
    print(f"\nMissing value statistics for {name}:")
    missing_stats = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Percentage Missing': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Percentage Missing', ascending=False)
    
    stats_with_missing = missing_stats[missing_stats['Missing Values'] > 0]
    if not stats_with_missing.empty:
        print(stats_with_missing)
    else:
        print("No missing values found!")
    
    # Create nullity correlation heatmap (only for numeric columns)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        correlation = numeric_df.isnull().corr()
        sns.heatmap(correlation, 
                   mask=np.zeros_like(correlation, dtype=bool),
                   cmap=plt.cm.RdYlBu,
                   annot=True)
        plt.title(f'Nullity Correlation Heatmap - {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'nullity_correlation_{name}.png'))
        plt.close()

def analyze_merged_data(sales_df, weather_df):
    """Analyze missing patterns in merged dataset"""
    print("\nAnalyzing merged dataset...")
    
    # Merge sales and weather data
    merged_df = pd.merge(sales_df, weather_df, on='Datum', how='left')
    
    # Analyze patterns in merged data
    analyze_missing_patterns(merged_df, 'merged_data', OUTPUT_DIR)
    
    # Analyze temporal patterns
    plt.figure(figsize=(15, 6))
    msno.bar(merged_df.set_index('Datum').sort_index())
    plt.title('Missing Values Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'missing_temporal_patterns.png'))
    plt.close()
    
    # Additional temporal analysis
    print("\nTemporal missing value patterns:")
    merged_df['Year'] = merged_df['Datum'].dt.year
    merged_df['Month'] = merged_df['Datum'].dt.month
    
    # Missing values by year
    yearly_missing = merged_df.groupby('Year').apply(lambda x: x.isnull().sum())
    print("\nMissing values by year:")
    print(yearly_missing[yearly_missing > 0].to_string())
    
    # Missing values by month
    monthly_missing = merged_df.groupby('Month').apply(lambda x: x.isnull().sum())
    print("\nMissing values by month:")
    print(monthly_missing[monthly_missing > 0].to_string())

def main():
    """Main function to run the analysis"""
    print("Starting missing data analysis...")
    
    # Load all datasets
    sales_df, weather_df, weather_codes = load_data()
    
    # Analyze individual datasets
    analyze_missing_patterns(sales_df, 'sales', OUTPUT_DIR)
    analyze_missing_patterns(weather_df, 'weather', OUTPUT_DIR)
    
    # Analyze merged dataset
    analyze_merged_data(sales_df, weather_df)
    
    print(f"\nAnalysis complete. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

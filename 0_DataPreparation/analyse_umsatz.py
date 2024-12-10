import pandas as pd
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
from config import (TRAIN_PATH, WEATHER_PATH, KIWO_PATH, 
                   VIZ_DIR, WARENGRUPPEN)

def save_analysis_results(stats_dict, correlations_dict, output_dir):
    """Save analysis results to files"""
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save statistics
    stats_file = os.path.join(output_dir, f'sales_statistics_{timestamp}.txt')
    with open(stats_file, 'w') as f:
        f.write("Sales Analysis Statistics\n")
        f.write("=======================\n\n")
        for title, stats in stats_dict.items():
            f.write(f"{title}\n")
            f.write("-" * len(title) + "\n")
            f.write(f"{stats}\n\n")
    
    # Save correlations
    corr_file = os.path.join(output_dir, f'sales_correlations_{timestamp}.txt')
    with open(corr_file, 'w') as f:
        f.write("Sales Correlations Analysis\n")
        f.write("=========================\n\n")
        for title, corr in correlations_dict.items():
            f.write(f"{title}\n")
            f.write("-" * len(title) + "\n")
            f.write(f"{corr}\n\n")
    
    return stats_file, corr_file

def analyze_sales():
    """Analyze sales data and their relationships with various factors"""
    print("Analyzing sales data...")
    
    # Create output directories
    output_dir = os.path.join(SCRIPT_DIR, 'output', 'sales_analysis')
    viz_output_dir = os.path.join(VIZ_DIR, 'sales_analysis')
    for directory in [output_dir, viz_output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Read data
    train_df = pd.read_csv(TRAIN_PATH)
    kiwo_df = pd.read_csv(KIWO_PATH)
    wetter_df = pd.read_csv(WEATHER_PATH)
    
    # Convert dates to datetime
    for df in [train_df, kiwo_df, wetter_df]:
        df['Datum'] = pd.to_datetime(df['Datum'])
    
    # Merge all dataframes
    merged_df = train_df.merge(kiwo_df, on='Datum', how='left')
    final_df = merged_df.merge(wetter_df, on='Datum', how='left')
    
    # Ensure KielerWoche is coded as 0 and 1
    final_df['KielerWoche'] = final_df['KielerWoche'].fillna(0).astype(int)
    
    # Prepare statistics dictionary
    stats_dict = {
        'Overall Statistics': final_df.describe(),
        'Statistics by Product Group': final_df.groupby('Warengruppe')['Umsatz'].describe(),
        'Statistics by Kieler Woche': final_df.groupby('KielerWoche')['Umsatz'].describe()
    }
    
    # Print descriptive statistics
    print("\nDescriptive Statistics:")
    print(stats_dict['Overall Statistics'])
    
    # Create visualizations
    # 1. Sales distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=final_df, x='Umsatz', bins=30)
    plt.title('Sales Distribution')
    plt.xlabel('Sales (€)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(viz_output_dir, 'sales_distribution.png'))
    plt.close()
    
    # 2. Temperature distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=final_df, x='Temperatur', bins=30)
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(viz_output_dir, 'temperature_distribution.png'))
    plt.close()
    
    # 3. Sales vs. Temperature
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=final_df, x='Temperatur', y='Umsatz')
    plt.title('Sales vs. Temperature')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Sales (€)')
    plt.savefig(os.path.join(viz_output_dir, 'sales_vs_temperature.png'))
    plt.close()
    
    # 4. Sales during Kieler Woche
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=final_df, x='KielerWoche', y='Umsatz')
    plt.title('Sales Comparison: During vs. Outside Kieler Woche')
    plt.xlabel('Kieler Woche (0 = Outside, 1 = During)')
    plt.ylabel('Sales (€)')
    plt.savefig(os.path.join(viz_output_dir, 'sales_kiwo_comparison.png'))
    plt.close()
    
    # 5. Sales distribution by product group
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=final_df, x='Warengruppe', y='Umsatz')
    plt.title('Sales Distribution by Product Group')
    plt.xlabel('Product Group')
    plt.ylabel('Sales (€)')
    plt.xticks(range(len(WARENGRUPPEN)), 
               [name for name in WARENGRUPPEN.values()],
               rotation=45)
    plt.savefig(os.path.join(viz_output_dir, 'sales_by_product.png'))
    plt.close()
    
    # Calculate correlations
    numeric_cols = ['Umsatz', 'KielerWoche', 'Temperatur', 'Bewoelkung', 'Windgeschwindigkeit']
    correlations = final_df[numeric_cols].corr()
    sales_corr = correlations.loc['Umsatz']
    
    # Sort correlations by absolute value
    sales_corr_sorted = pd.concat([
        pd.Series({'Umsatz': 1.0}),
        sales_corr[1:].abs().sort_values(ascending=False).map(
            lambda x: sales_corr[sales_corr.abs() == x].iloc[0]
        )
    ])
    
    # Store correlations
    correlations_dict = {
        'Sales Correlations (sorted by strength)': sales_corr_sorted,
        'Full Correlation Matrix': correlations
    }
    
    # Print correlations
    print("\nCorrelations with Sales (sorted by strength):")
    print(sales_corr_sorted)
    
    # 6. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, 
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.3f',
                annot_kws={'size': 10})
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(viz_output_dir, 'correlation_matrix.png'), 
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    # Save analysis results
    stats_file, corr_file = save_analysis_results(stats_dict, correlations_dict, output_dir)
    
    print(f"\nAnalysis results saved to:")
    print(f"- Statistics: {stats_file}")
    print(f"- Correlations: {corr_file}")
    print(f"- Visualizations: {viz_output_dir}")

if __name__ == "__main__":
    analyze_sales()

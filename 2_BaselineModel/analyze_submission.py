import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Import configuration
from bakery_sales_prediction.config import (TRAIN_PATH, TEST_PATH, 
                                          SAMPLE_SUBMISSION_PATH,
                                          WARENGRUPPEN, get_warengruppe_name)

# Define paths
MODEL_TO_ANALYZE = 'simple_weekday_model'  # Change this to analyze different models
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output', 'submission_analysis', MODEL_TO_ANALYZE)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input files
submission_path = os.path.join(SCRIPT_DIR, 'output', MODEL_TO_ANALYZE, 'submission.csv')

def load_submission_data():
    """Load and prepare submission data"""
    print(f"Reading submission from: {submission_path}")
    df = pd.read_csv(submission_path)
    
    # Convert ID to datetime
    df['date'] = pd.to_datetime(df['id'].astype(str).str[:6], format='%y%m%d')
    df['weekday'] = df['date'].dt.day_name()
    
    # Add product group names
    df['product_group'] = df['id'].astype(str).str[-1].astype(int)
    df['product_name'] = df['product_group'].map(get_warengruppe_name)
    
    return df

def plot_daily_predictions():
    """Plot daily prediction patterns"""
    df = load_submission_data()
    
    # Calculate daily totals
    daily_totals = df.groupby('date')['Umsatz'].sum().reset_index()
    
    # Create the plot
    plt.figure(figsize=(15, 6))
    plt.plot(daily_totals['date'], daily_totals['Umsatz'], marker='o')
    plt.title('Daily Total Sales Predictions')
    plt.xlabel('Date')
    plt.ylabel('Predicted Sales (€)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'daily_predictions.png'))
    plt.close()

def plot_weekday_patterns():
    """Plot weekday patterns in predictions"""
    df = load_submission_data()
    
    # Calculate average sales by weekday
    weekday_avg = df.groupby(['weekday', 'product_name'])['Umsatz'].agg(['mean', 'std']).reset_index()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_avg['weekday'] = pd.Categorical(weekday_avg['weekday'], categories=weekday_order, ordered=True)
    
    # Overall weekday pattern
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='weekday', y='Umsatz', order=weekday_order)
    plt.title('Average Sales by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Predicted Sales (€)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'weekday_patterns.png'))
    plt.close()
    
    # Weekday pattern by product group
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df, x='weekday', y='Umsatz', hue='product_name', order=weekday_order)
    plt.title('Sales Distribution by Weekday and Product Group')
    plt.xlabel('Weekday')
    plt.ylabel('Predicted Sales (€)')
    plt.legend(title='Product Group', bbox_to_anchor=(1.05, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'weekday_product_patterns.png'))
    plt.close()

def analyze_distribution():
    """Analyze the distribution of predictions"""
    df = load_submission_data()
    
    # Overall distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Umsatz', bins='auto')
    plt.title('Distribution of Sales Predictions')
    plt.xlabel('Predicted Sales (€)')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sales_distribution.png'))
    plt.close()
    
    # Distribution by product group
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df, x='product_name', y='Umsatz')
    plt.title('Sales Distribution by Product Group')
    plt.xlabel('Product Group')
    plt.ylabel('Predicted Sales (€)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sales_distribution_by_product.png'))
    plt.close()
    
    # Calculate and save statistics
    stats = df.groupby('product_name')['Umsatz'].describe()
    stats.to_csv(os.path.join(OUTPUT_DIR, 'prediction_statistics.csv'))
    
    # Print summary statistics
    print("\nSummary Statistics by Product Group:")
    print(stats.to_string())

def main():
    """Main analysis function"""
    print("Analyzing submission predictions...")
    
    # Create visualizations
    plot_daily_predictions()
    plot_weekday_patterns()
    analyze_distribution()
    
    print(f"\nAnalysis complete. Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 
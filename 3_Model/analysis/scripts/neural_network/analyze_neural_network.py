"""
Analyzes neural network model predictions for bakery sales forecasting.
Generates visualizations and statistics including:
- Weekly sales patterns per product
- Weekday sales trends across periods
- Timeline comparisons of daily sales
- Product-specific and overall statistics
- Weekday-product sales breakdowns

Usage: python analyze_neural_network.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# German product names mapping
PRODUCT_NAMES = {
    1: 'Brot',
    2: 'Brötchen',
    3: 'Croissant',
    4: 'Konditorei',
    5: 'Kuchen',
    6: 'Saisonbrot'
}

# German weekday names
WEEKDAY_NAMES = {
    0: 'Montag',
    1: 'Dienstag',
    2: 'Mittwoch',
    3: 'Donnerstag',
    4: 'Freitag',
    5: 'Samstag',
    6: 'Sonntag'
}

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..')
MODEL_NAME = 'neural_network_model_v14'
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'neural_network', MODEL_NAME)
OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis', 'results', 'neural_network', MODEL_NAME)
VISUALIZATION_DIR = os.path.join(BASE_DIR, 'analysis', 'visualizations', 'neural_network', MODEL_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def load_data():
    """Load all necessary data for analysis"""
    # Load predictions
    pred_path = os.path.join(MODEL_DIR, 'submission.csv')
    if not os.path.exists(pred_path):
        print(f"Prediction file not found at: {pred_path}")
        return None, None, None
    
    # Load training and validation data
    train_path = os.path.join(BASE_DIR, '..', '0_DataPreparation', 'input', 'competition_data', 'train.csv')
    if not os.path.exists(train_path):
        print(f"Training data file not found at: {train_path}")
        return None, None, None
    
    train_df = pd.read_csv(train_path)
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    train_df['weekday'] = train_df['Datum'].dt.dayofweek
    
    # Split into training and validation
    val_mask = (train_df['Datum'] >= '2017-08-01') & (train_df['Datum'] <= '2018-07-31')
    val_df = train_df[val_mask].copy()
    train_df = train_df[~val_mask].copy()
    
    # Load and prepare predictions
    pred_df = pd.read_csv(pred_path)
    pred_df['date'] = pd.to_datetime('20' + pred_df['id'].astype(str).str[:6], format='%Y%m%d')
    pred_df['product'] = pred_df['id'].astype(str).str[-1].astype(int)
    pred_df['weekday'] = pred_df['date'].dt.dayofweek
    
    return train_df, val_df, pred_df

def plot_weekly_patterns(train_df, val_df, pred_df):
    """Plot weekly sales patterns for each product"""
    print("\nCreating weekly pattern plots...")
    
    for product in range(1, 7):
        plt.figure(figsize=(12, 7))
        
        # Training data
        train_product = train_df[train_df['Warengruppe'] == product]
        train_avg = train_product.groupby('weekday')['Umsatz'].agg(['mean', 'std']).reset_index()
        plt.plot(train_avg['weekday'], train_avg['mean'], 'b-', label='Training', marker='o')
        plt.fill_between(train_avg['weekday'], 
                        train_avg['mean'] - train_avg['std'],
                        train_avg['mean'] + train_avg['std'],
                        alpha=0.2, color='blue')
        
        # Validation data
        val_product = val_df[val_df['Warengruppe'] == product]
        val_avg = val_product.groupby('weekday')['Umsatz'].agg(['mean', 'std']).reset_index()
        plt.plot(val_avg['weekday'], val_avg['mean'], 'g-', label='Validation', marker='s')
        plt.fill_between(val_avg['weekday'],
                        val_avg['mean'] - val_avg['std'],
                        val_avg['mean'] + val_avg['std'],
                        alpha=0.2, color='green')
        
        # Predictions
        pred_product = pred_df[pred_df['product'] == product]
        pred_avg = pred_product.groupby('weekday')['Umsatz'].mean()
        plt.plot(pred_avg.index, pred_avg.values, 'r--', label='Predicted', marker='^')
        
        plt.title(f'{PRODUCT_NAMES[product]} - Weekly Sales Pattern', fontsize=14, pad=20)
        plt.xlabel('Weekday', fontsize=12)
        plt.ylabel('Average Sales', fontsize=12)
        plt.xticks(range(7), [WEEKDAY_NAMES[i] for i in range(7)], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # Add statistics
        stats_text = f"Statistics:\n"
        stats_text += f"Training Mean: {train_product['Umsatz'].mean():.1f}\n"
        stats_text += f"Training Std: {train_product['Umsatz'].std():.1f}\n"
        stats_text += f"Validation Mean: {val_product['Umsatz'].mean():.1f}\n"
        stats_text += f"Validation Std: {val_product['Umsatz'].std():.1f}\n"
        stats_text += f"Predicted Mean: {pred_product['Umsatz'].mean():.1f}"
        
        plt.text(1.15, 0.5, stats_text,
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=10,
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, f'weekly_pattern_product_{product}.png'))
        plt.close()

def plot_weekday_patterns(train_df, val_df, pred_df):
    """Plot average sales by weekday across all periods"""
    print("\nCreating weekday pattern plots...")
    
    plt.figure(figsize=(12, 7))
    
    # Calculate average sales by weekday for each period
    train_avg = train_df.groupby('weekday')['Umsatz'].mean()
    val_avg = val_df.groupby('weekday')['Umsatz'].mean()
    pred_avg = pred_df.groupby('weekday')['Umsatz'].mean()
    
    # Plot with error bars
    plt.errorbar(train_avg.index, train_avg.values,
                yerr=train_df.groupby('weekday')['Umsatz'].std(),
                fmt='o-', label='Training', capsize=5)
    plt.errorbar(val_avg.index, val_avg.values,
                yerr=val_df.groupby('weekday')['Umsatz'].std(),
                fmt='s-', label='Validation', capsize=5)
    plt.plot(pred_avg.index, pred_avg.values, '^--', label='Predicted')
    
    plt.title('Average Daily Sales by Weekday', fontsize=14, pad=20)
    plt.xlabel('Weekday', fontsize=12)
    plt.ylabel('Average Sales', fontsize=12)
    plt.xticks(range(7), [WEEKDAY_NAMES[i] for i in range(7)], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add overall statistics
    stats_text = f"Overall Statistics:\n"
    stats_text += f"Training Mean: {train_df['Umsatz'].mean():.1f}\n"
    stats_text += f"Training Std: {train_df['Umsatz'].std():.1f}\n"
    stats_text += f"Validation Mean: {val_df['Umsatz'].mean():.1f}\n"
    stats_text += f"Validation Std: {val_df['Umsatz'].std():.1f}\n"
    stats_text += f"Predicted Mean: {pred_df['Umsatz'].mean():.1f}"
    
    plt.text(1.15, 0.5, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=10,
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'weekday_patterns.png'))
    plt.close()

def plot_timeline_sales(train_df, val_df, pred_df):
    """Plot timeline of total daily sales for all periods"""
    print("\nCreating timeline plots...")
    
    # Calculate daily totals
    train_daily = train_df.groupby('Datum')['Umsatz'].sum().reset_index()
    val_daily = val_df.groupby('Datum')['Umsatz'].sum().reset_index()
    pred_daily = pred_df.groupby('date')['Umsatz'].sum().reset_index()
    
    plt.figure(figsize=(15, 7))
    
    # Plot each period
    plt.plot(train_daily['Datum'], train_daily['Umsatz'], 'b-', label='Training', alpha=0.6)
    plt.plot(val_daily['Datum'], val_daily['Umsatz'], 'g-', label='Validation', alpha=0.6)
    plt.plot(pred_daily['date'], pred_daily['Umsatz'], 'r--', label='Predicted', alpha=0.8)
    
    plt.title('Daily Total Sales Timeline', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Daily Sales', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add period labels
    plt.axvline(x=pd.to_datetime('2017-08-01'), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=pd.to_datetime('2018-08-01'), color='gray', linestyle='--', alpha=0.5)
    
    plt.text(pd.to_datetime('2016-01-01'), plt.ylim()[1], 'Training Period',
             horizontalalignment='center', verticalalignment='bottom')
    plt.text(pd.to_datetime('2018-01-01'), plt.ylim()[1], 'Validation Period',
             horizontalalignment='center', verticalalignment='bottom')
    plt.text(pd.to_datetime('2019-01-01'), plt.ylim()[1], 'Prediction Period',
             horizontalalignment='center', verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'timeline_sales.png'))
    plt.close()

def create_product_heatmap(data, title, filename):
    """Create a heatmap of average sales by product and weekday"""
    print(f"\nCreating {title} heatmap...")
    
    pivot_data = data.pivot_table(
        values='Umsatz',
        index='product' if 'product' in data.columns else 'Warengruppe',
        columns='weekday',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[WEEKDAY_NAMES[i] for i in range(7)],
                yticklabels=[PRODUCT_NAMES[i] for i in range(1, 7)])
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Weekday', fontsize=12)
    plt.ylabel('Product', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, filename))
    plt.close()

def plot_sales_distribution(train_df, val_df, pred_df):
    """Plot distribution of sales across different sets"""
    print("\nCreating sales distribution plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot training data distribution
    sns.kdeplot(data=train_df['Umsatz'], label='Training', alpha=0.7)
    
    # Plot validation data distribution
    sns.kdeplot(data=val_df['Umsatz'], label='Validation', alpha=0.7)
    
    # Plot test predictions distribution
    sns.kdeplot(data=pred_df['Umsatz'], label='Test Predictions', alpha=0.7)
    
    plt.title('Distribution of Sales Across Different Sets')
    plt.xlabel('Sales (€)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'sales_distribution.png'))
    plt.close()

def analyze_special_events(train_df, val_df, pred_df):
    """Analyze predictions for special events like Easter Saturday"""
    print("\nAnalyzing special events...")
    
    # Function to identify Easter Saturday
    def is_easter_saturday(date):
        easter_saturdays = {
            2013: '2013-03-30',
            2014: '2014-04-19',
            2015: '2015-04-04',
            2016: '2016-03-26',
            2017: '2017-04-15',
            2018: '2018-03-31',
            2019: '2019-04-20'
        }
        for year, date_str in easter_saturdays.items():
            if pd.to_datetime(date_str) == date:
                return True
        return False
    
    # Analyze Easter Saturday sales
    train_easter = train_df[train_df['Datum'].apply(is_easter_saturday)]
    val_easter = val_df[val_df['Datum'].apply(is_easter_saturday)]
    pred_easter = pred_df[pred_df['date'].apply(is_easter_saturday)]
    
    plt.figure(figsize=(12, 6))
    
    # Plot Easter Saturday sales by product
    for period, data, label, color in [
        (train_easter, 'Warengruppe', 'Training', 'blue'),
        (val_easter, 'Warengruppe', 'Validation', 'green'),
        (pred_easter, 'product', 'Predicted', 'red')
    ]:
        if not period.empty:
            sales_by_product = period.groupby(data)['Umsatz'].mean()
            plt.plot(range(1, 7), [sales_by_product.get(i, 0) for i in range(1, 7)],
                    marker='o', label=label, color=color)
    
    plt.title('Average Easter Saturday Sales by Product')
    plt.xlabel('Product Group')
    plt.ylabel('Sales (€)')
    plt.xticks(range(1, 7), [PRODUCT_NAMES[i] for i in range(1, 7)], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'easter_saturday_sales.png'))
    plt.close()
    
    # Save Easter Saturday analysis
    with open(os.path.join(OUTPUT_DIR, 'easter_saturday_analysis.txt'), 'w') as f:
        f.write("Easter Saturday Sales Analysis\n")
        f.write("=" * 30 + "\n\n")
        
        for period, data in [
            (train_easter, "Training"),
            (val_easter, "Validation"),
            (pred_easter, "Predicted")
        ]:
            if not period.empty:
                f.write(f"{data} Period:\n")
                f.write(f"Average Sales: {period['Umsatz'].mean():.2f}\n")
                f.write(f"Max Sales: {period['Umsatz'].max():.2f}\n")
                f.write(f"Min Sales: {period['Umsatz'].min():.2f}\n\n")

def plot_separated_validation_test_comparison(val_df, pred_df):
    """Create separate plots for validation and test period sales"""
    print("\nCreating separated validation-test comparison plots...")
    
    plt.figure(figsize=(15, 10))
    
    # Plot validation data
    plt.subplot(2, 1, 1)
    for product in range(1, 7):
        product_data = val_df[val_df['Warengruppe'] == product]
        plt.plot(product_data['Datum'], product_data['Umsatz'],
                label=PRODUCT_NAMES[product], alpha=0.7)
    
    plt.title('Validation Period Sales by Product')
    plt.xlabel('Date')
    plt.ylabel('Sales (€)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot test predictions
    plt.subplot(2, 1, 2)
    for product in range(1, 7):
        product_data = pred_df[pred_df['product'] == product]
        plt.plot(product_data['date'], product_data['Umsatz'],
                label=PRODUCT_NAMES[product], alpha=0.7)
    
    plt.title('Test Period Predictions by Product')
    plt.xlabel('Date')
    plt.ylabel('Sales (€)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'separated_validation_test_comparison.png'),
                bbox_inches='tight')
    plt.close()

def plot_yearly_stacked_timeline(train_df, val_df, pred_df):
    """Create stacked yearly plots for all periods"""
    print("\nCreating yearly stacked timeline...")
    
    # Calculate daily totals for each period
    train_daily = train_df.groupby(['Datum', 'Warengruppe'])['Umsatz'].sum().reset_index()
    val_daily = val_df.groupby(['Datum', 'Warengruppe'])['Umsatz'].sum().reset_index()
    pred_daily = pred_df.groupby(['date', 'product'])['Umsatz'].sum().reset_index()
    
    # Create figure with subplots for each year
    years = range(2013, 2020)
    fig, axes = plt.subplots(len(years), 1, figsize=(15, 4*len(years)))
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        
        # Filter data for current year
        train_year = train_daily[train_daily['Datum'].dt.year == year]
        val_year = val_daily[val_daily['Datum'].dt.year == year]
        pred_year = pred_daily[pred_daily['date'].dt.year == year]
        
        # Plot data
        if not train_year.empty:
            for product in range(1, 7):
                product_data = train_year[train_year['Warengruppe'] == product]
                ax.plot(product_data['Datum'], product_data['Umsatz'],
                       label=f"{PRODUCT_NAMES[product]} (Train)" if idx == 0 else "",
                       alpha=0.7)
        
        if not val_year.empty:
            for product in range(1, 7):
                product_data = val_year[val_year['Warengruppe'] == product]
                ax.plot(product_data['Datum'], product_data['Umsatz'],
                       label=f"{PRODUCT_NAMES[product]} (Val)" if idx == 0 else "",
                       alpha=0.7)
        
        if not pred_year.empty:
            for product in range(1, 7):
                product_data = pred_year[pred_year['product'] == product]
                ax.plot(product_data['date'], product_data['Umsatz'],
                       label=f"{PRODUCT_NAMES[product]} (Pred)" if idx == 0 else "",
                       alpha=0.7)
        
        ax.set_title(f'Sales in {year}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales (€)')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'yearly_stacked_timeline.png'),
                bbox_inches='tight')
    plt.close()

def plot_product_comparison(train_df, val_df, pred_df):
    """Create product comparison plots"""
    print("\nCreating product comparison plots...")
    
    # Calculate average sales by product for each period
    train_avg = train_df.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std']).reset_index()
    val_avg = val_df.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std']).reset_index()
    pred_avg = pred_df.groupby('product')['Umsatz'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(PRODUCT_NAMES))
    width = 0.25
    
    # Plot bars with error bars
    plt.bar(x - width, train_avg['mean'], width, yerr=train_avg['std'],
            label='Training', alpha=0.7, capsize=5)
    plt.bar(x, val_avg['mean'], width, yerr=val_avg['std'],
            label='Validation', alpha=0.7, capsize=5)
    plt.bar(x + width, pred_avg['mean'], width, yerr=pred_avg['std'],
            label='Predicted', alpha=0.7, capsize=5)
    
    plt.xlabel('Product Group')
    plt.ylabel('Average Sales (€)')
    plt.title('Average Sales by Product Group - Comparison Across Periods')
    plt.xticks(x, [PRODUCT_NAMES[i] for i in range(1, 7)], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'product_comparison.png'))
    plt.close()

def analyze_predictions():
    """Analyze predictions from the model"""
    print(f"\nAnalyzing {MODEL_NAME}...")
    
    # Load data
    train_df, val_df, pred_df = load_data()
    if train_df is None or val_df is None or pred_df is None:
        print("Error loading data. Exiting...")
        return
    
    # Create visualizations
    plot_weekly_patterns(train_df, val_df, pred_df)
    plot_weekday_patterns(train_df, val_df, pred_df)
    plot_timeline_sales(train_df, val_df, pred_df)
    plot_sales_distribution(train_df, val_df, pred_df)
    plot_separated_validation_test_comparison(val_df, pred_df)
    plot_yearly_stacked_timeline(train_df, val_df, pred_df)
    plot_product_comparison(train_df, val_df, pred_df)
    
    # Create heatmaps
    create_product_heatmap(train_df, 'Training Data - Average Sales by Product and Weekday', 'heatmap_training.png')
    create_product_heatmap(val_df, 'Validation Data - Average Sales by Product and Weekday', 'heatmap_validation.png')
    create_product_heatmap(pred_df, 'Predicted Data - Average Sales by Product and Weekday', 'heatmap_predicted.png')
    
    # Analyze special events
    analyze_special_events(train_df, val_df, pred_df)
    
    # Generate summary statistics
    summary_stats = {
        'Training': {
            'mean': train_df['Umsatz'].mean(),
            'std': train_df['Umsatz'].std(),
            'min': train_df['Umsatz'].min(),
            'max': train_df['Umsatz'].max(),
            'total_days': train_df['Datum'].nunique()
        },
        'Validation': {
            'mean': val_df['Umsatz'].mean(),
            'std': val_df['Umsatz'].std(),
            'min': val_df['Umsatz'].min(),
            'max': val_df['Umsatz'].max(),
            'total_days': val_df['Datum'].nunique()
        },
        'Predicted': {
            'mean': pred_df['Umsatz'].mean(),
            'std': pred_df['Umsatz'].std(),
            'min': pred_df['Umsatz'].min(),
            'max': pred_df['Umsatz'].max(),
            'total_days': pred_df['date'].nunique()
        }
    }
    
    # Save summary statistics
    with open(os.path.join(OUTPUT_DIR, 'summary_statistics.txt'), 'w') as f:
        f.write(f"Summary Statistics for {MODEL_NAME}\n")
        f.write("=" * 50 + "\n\n")
        
        for period, stats in summary_stats.items():
            f.write(f"{period} Period:\n")
            f.write(f"  Mean Sales: {stats['mean']:.2f}\n")
            f.write(f"  Std Dev: {stats['std']:.2f}\n")
            f.write(f"  Min Sales: {stats['min']:.2f}\n")
            f.write(f"  Max Sales: {stats['max']:.2f}\n")
            f.write(f"  Total Days: {stats['total_days']}\n\n")
    
    print("\nAnalysis completed! Results saved to:")
    print(f"- Visualizations: {VISUALIZATION_DIR}")
    print(f"- Statistics: {OUTPUT_DIR}")

if __name__ == "__main__":
    analyze_predictions()
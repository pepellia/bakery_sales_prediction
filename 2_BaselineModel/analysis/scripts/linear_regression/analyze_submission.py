"""
Analysis of Linear Regression Model Predictions

This script analyzes the predictions from the linear regression model, including:
- Training performance
- Validation performance
- Prediction patterns
- Product group analysis
- Interactive timeline visualization
- Special events analysis (Easter Saturday and Windjammer Parade)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))  # 2_BaselineModel directory

# Data paths
DATA_DIR = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'competition_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')

# Define paths
MODEL_NAME = 'simple_weekday_model'
MODEL_OUTPUT = os.path.join(MODEL_ROOT, 'output', MODEL_NAME)
ANALYSIS_OUTPUT = os.path.join(MODEL_ROOT, 'analysis', 'results', 'linear_regression', MODEL_NAME)
ANALYSIS_VIZ = os.path.join(MODEL_ROOT, 'analysis', 'visualizations', 'linear_regression', MODEL_NAME)

# Create output directories
for directory in [ANALYSIS_OUTPUT, ANALYSIS_VIZ]:
    os.makedirs(directory, exist_ok=True)

# Product group configurations
WARENGRUPPEN = {
    1: 'Brot',
    2: 'Brötchen',
    3: 'Croissant',
    4: 'Konditorei',
    5: 'Kuchen',
    6: 'Saisonbrot'
}

def get_warengruppe_name(code):
    """Get the name of a product group by its code"""
    return WARENGRUPPEN.get(code, f'Unknown ({code})')

def load_data():
    """Load model predictions and actual data"""
    # Load training data
    train_data = pd.read_csv(TRAIN_PATH)
    train_data['Datum'] = pd.to_datetime(train_data['Datum'])
    
    # Create train/validation split based on dates
    train_mask = (train_data['Datum'] >= '2013-08-01') & (train_data['Datum'] <= '2017-07-31')
    val_mask = (train_data['Datum'] >= '2017-08-01') & (train_data['Datum'] <= '2018-07-31')
    
    train_df = train_data[train_mask].copy()
    val_df = train_data[val_mask].copy()
    
    # Load test predictions
    submission_df = pd.read_csv(os.path.join(MODEL_OUTPUT, 'submission.csv'))
    
    # Create test DataFrame with proper date format
    test_dates = []
    test_products = []
    for id_str in submission_df['id'].astype(str):
        date_str = '20' + id_str[:6]
        product = int(id_str[-1])
        date = pd.to_datetime(date_str, format='%Y%m%d')
        test_dates.append(date)
        test_products.append(product)
    
    test_df = pd.DataFrame({
        'Datum': test_dates,
        'Warengruppe': test_products,
        'Umsatz': submission_df['Umsatz']
    })
    
    # Load model predictions
    train_pred = pd.read_csv(os.path.join(MODEL_OUTPUT, 'train_predictions.csv'))
    val_pred = pd.read_csv(os.path.join(MODEL_OUTPUT, 'val_predictions.csv'))
    
    # Calculate metrics
    metrics = {}
    
    # Training metrics
    train_rmse = np.sqrt(np.mean((train_df['Umsatz'] - train_pred['predicted_sales']) ** 2))
    train_mae = np.mean(np.abs(train_df['Umsatz'] - train_pred['predicted_sales']))
    train_r2 = 1 - np.sum((train_df['Umsatz'] - train_pred['predicted_sales']) ** 2) / np.sum((train_df['Umsatz'] - train_df['Umsatz'].mean()) ** 2)
    
    metrics['training'] = {
        'RMSE': train_rmse,
        'MAE': train_mae,
        'R2': train_r2
    }
    
    # Validation metrics
    val_rmse = np.sqrt(np.mean((val_df['Umsatz'] - val_pred['predicted_sales']) ** 2))
    val_mae = np.mean(np.abs(val_df['Umsatz'] - val_pred['predicted_sales']))
    val_r2 = 1 - np.sum((val_df['Umsatz'] - val_pred['predicted_sales']) ** 2) / np.sum((val_df['Umsatz'] - val_df['Umsatz'].mean()) ** 2)
    
    metrics['validation'] = {
        'RMSE': val_rmse,
        'MAE': val_mae,
        'R2': val_r2
    }
    
    # Add product group names
    for df in [train_df, val_df, test_df]:
        df['Warengruppe_Name'] = df['Warengruppe'].map(get_warengruppe_name)
        df['Wochentag'] = df['Datum'].dt.dayofweek
    
    return train_df, val_df, test_df, train_pred, val_pred, metrics

def analyze_metrics(metrics):
    """Analyze and compare training and validation metrics"""
    print("\nModel Performance Metrics:")
    
    print("\nTraining Metrics:")
    for metric, value in metrics['training'].items():
        print(f"{metric}: {value:.3f}" if value is not None else f"{metric}: N/A")
    
    print("\nValidation Metrics:")
    for metric, value in metrics['validation'].items():
        print(f"{metric}: {value:.3f}" if value is not None else f"{metric}: N/A")
    
    # Save metrics summary
    metrics_summary = pd.DataFrame({
        'Training': metrics['training'],
        'Validation': metrics['validation']
    }).round(3)
    
    metrics_summary.to_csv(os.path.join(ANALYSIS_OUTPUT, 'metrics_summary.csv'))

def plot_sales_distribution(train_set, val_set, test_df):
    """Plot distribution of sales across different sets"""
    plt.figure(figsize=(12, 6))
    
    # Plot training data distribution
    sns.kdeplot(data=train_set['Umsatz'], label='Training', alpha=0.7)
    
    # Plot validation data distribution
    sns.kdeplot(data=val_set['Umsatz'], label='Validation', alpha=0.7)
    
    # Plot test predictions distribution
    sns.kdeplot(data=test_df['Umsatz'], label='Test Predictions', alpha=0.7)
    
    plt.title('Distribution of Sales Across Different Sets')
    plt.xlabel('Sales (€)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'sales_distribution.png'))
    plt.close()

def analyze_by_product(train_df, val_df, test_df):
    """Analyze performance by product group"""
    # Calculate statistics by product group for each period
    train_stats = train_df.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std', 'count']).round(2)
    val_stats = val_df.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std', 'count']).round(2)
    test_stats = test_df.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std', 'count']).round(2)
    
    # Combine statistics
    stats_comparison = pd.DataFrame({
        'Train_Mean': train_stats['mean'],
        'Train_Std': train_stats['std'],
        'Train_Count': train_stats['count'],
        'Val_Mean': val_stats['mean'],
        'Val_Std': val_stats['std'],
        'Val_Count': val_stats['count'],
        'Test_Mean': test_stats['mean'],
        'Test_Std': test_stats['std'],
        'Test_Count': test_stats['count']
    })
    
    # Save product analysis
    stats_comparison.to_csv(os.path.join(ANALYSIS_OUTPUT, 'product_analysis.csv'))
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(train_stats))
    width = 0.25
    
    plt.bar(x - width, train_stats['mean'], width, label='Training', alpha=0.7)
    plt.bar(x, val_stats['mean'], width, label='Validation', alpha=0.7)
    plt.bar(x + width, test_stats['mean'], width, label='Test', alpha=0.7)
    
    plt.xlabel('Product Group')
    plt.ylabel('Average Sales (€)')
    plt.title('Average Sales by Product Group - Train vs Validation vs Test')
    plt.xticks(x, [WARENGRUPPEN[i] for i in range(1, len(train_stats) + 1)])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'product_comparison.png'))
    plt.close()

def analyze_temporal_patterns(train_df, val_df, test_df):
    """Analyze temporal patterns in the data"""
    # Add weekday names
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    train_df['Weekday'] = pd.Categorical(train_df['Datum'].dt.dayofweek.map(dict(enumerate(weekday_names))), categories=weekday_names)
    val_df['Weekday'] = pd.Categorical(val_df['Datum'].dt.dayofweek.map(dict(enumerate(weekday_names))), categories=weekday_names)
    test_df['Weekday'] = pd.Categorical(test_df['Datum'].dt.dayofweek.map(dict(enumerate(weekday_names))), categories=weekday_names)
    
    # Calculate average sales by weekday for each period
    train_weekday = train_df.groupby('Weekday', observed=True)['Umsatz'].mean()
    val_weekday = val_df.groupby('Weekday', observed=True)['Umsatz'].mean()
    test_weekday = test_df.groupby('Weekday', observed=True)['Umsatz'].mean()
    
    # Plot weekday patterns
    plt.figure(figsize=(12, 6))
    plt.plot(weekday_names, train_weekday, marker='o', label='Training', alpha=0.7)
    plt.plot(weekday_names, val_weekday, marker='s', label='Validation', alpha=0.7)
    plt.plot(weekday_names, test_weekday, marker='^', label='Test', alpha=0.7)
    
    plt.title('Average Sales by Weekday - All Periods')
    plt.xlabel('Weekday')
    plt.ylabel('Average Sales (€)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'weekday_patterns.png'))
    plt.close()
    
    # Save temporal analysis
    temporal_stats = pd.DataFrame({
        'Train_Mean': train_weekday,
        'Val_Mean': val_weekday,
        'Test_Mean': test_weekday
    }).round(2)
    temporal_stats.to_csv(os.path.join(ANALYSIS_OUTPUT, 'temporal_analysis.csv'))

def plot_timeline_sales(train_df, val_df, test_df):
    """Plot daily total sales for training, validation, and test periods"""
    # Calculate daily totals
    train_daily = train_df.groupby('Datum')['Umsatz'].sum().reset_index()
    val_daily = val_df.groupby('Datum')['Umsatz'].sum().reset_index()
    test_daily = test_df.groupby('Datum')['Umsatz'].sum().reset_index()
    
    # Find global min and max for y-axis scaling
    y_min = min(
        train_daily['Umsatz'].min(),
        val_daily['Umsatz'].min(),
        test_daily['Umsatz'].min()
    )
    y_max = max(
        train_daily['Umsatz'].max(),
        val_daily['Umsatz'].max(),
        test_daily['Umsatz'].max()
    )
    
    # Add some padding to y-axis limits
    y_padding = (y_max - y_min) * 0.1
    y_min = max(0, y_min - y_padding)  # Don't go below 0
    y_max = y_max + y_padding
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Training period plot
    ax1.plot(train_daily['Datum'], train_daily['Umsatz'], 'b-', linewidth=1)
    ax1.set_title('Training Period: Total Daily Sales (2013-08-01 to 2017-07-31)')
    ax1.set_xlabel('')
    ax1.set_ylabel('Total Sales (€)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(y_min, y_max)
    
    # Validation period plot
    ax2.plot(val_daily['Datum'], val_daily['Umsatz'], 'g-', linewidth=1)
    ax2.set_title('Validation Period: Total Daily Sales (2017-08-01 to 2018-07-31)')
    ax2.set_xlabel('')
    ax2.set_ylabel('Total Sales (€)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(y_min, y_max)
    
    # Test period plot
    ax3.plot(test_daily['Datum'], test_daily['Umsatz'], 'r-', linewidth=1)
    ax3.set_title('Test Period: Predicted Total Daily Sales')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Total Sales (€)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(y_min, y_max)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'timeline_sales.png'))
    plt.close()
    
    # Save the data
    timeline_data = {
        'training_period': train_daily.to_dict(orient='records'),
        'validation_period': val_daily.to_dict(orient='records'),
        'test_period': test_daily.to_dict(orient='records')
    }
    
    with open(os.path.join(ANALYSIS_OUTPUT, 'timeline_data.json'), 'w') as f:
        json.dump(timeline_data, f, indent=2, default=str)

def plot_weekday_patterns(train_df, val_df, test_df):
    """Plot weekly patterns for each product group"""
    # Create a figure with subplots for each product group
    n_products = len(WARENGRUPPEN)
    fig, axes = plt.subplots(n_products, 1, figsize=(12, 4*n_products))
    
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
    
    for (product_group, product_name), ax in zip(WARENGRUPPEN.items(), axes):
        # Get data for this product
        train_product = train_df[train_df['Warengruppe'] == product_group]
        val_product = val_df[val_df['Warengruppe'] == product_group]
        test_product = test_df[test_df['Warengruppe'] == product_group]
        
        # Calculate average sales by weekday
        train_weekday_avg = train_product.groupby('Wochentag')['Umsatz'].mean()
        val_weekday_avg = val_product.groupby('Wochentag')['Umsatz'].mean()
        test_weekday_avg = test_product.groupby('Wochentag')['Umsatz'].mean()
        
        # Plot weekday patterns
        ax.plot(weekday_names, train_weekday_avg, 
                marker='o', label='Actual (Train)', color='blue', linewidth=2)
        ax.plot(weekday_names, val_weekday_avg, 
                marker='s', label='Actual (Validation)', color='green', linewidth=2)
        ax.plot(weekday_names, test_weekday_avg, 
                marker='^', label='Predicted (Test)', color='red', linestyle=':', linewidth=2)
        
        ax.set_title(f'{product_name} - Weekly Sales Pattern')
        ax.set_xlabel('Weekday')
        ax.set_ylabel('Average Sales (€)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'weekly_patterns.png'))
    plt.close()

def plot_yearly_stacked_timeline(train_df, val_df, test_df):
    """Create stacked yearly plots for training period, followed by validation and test periods"""
    # Create figure with subplots
    fig, axes = plt.subplots(6, 1, figsize=(15, 20))
    
    # Calculate global min and max for y-axis scaling
    all_sales = []
    
    # Training period plots (2013-2017)
    years = [(2013, '2013-08-01', '2014-07-31'),
             (2014, '2014-08-01', '2015-07-31'),
             (2015, '2015-08-01', '2016-07-31'),
             (2016, '2016-08-01', '2017-07-31')]
    
    # Collect all sales data for y-axis scaling
    for year, start_date, end_date in years:
        mask = (train_df['Datum'] >= start_date) & (train_df['Datum'] <= end_date)
        year_data = train_df[mask].copy()
        daily_totals = year_data.groupby('Datum')['Umsatz'].sum()
        all_sales.extend(daily_totals.values)
    
    # Add validation and test data
    val_daily = val_df.groupby('Datum')['Umsatz'].sum()
    test_daily = test_df.groupby('Datum')['Umsatz'].sum()
    all_sales.extend(val_daily.values)
    all_sales.extend(test_daily.values)
    
    # Calculate global y-axis limits
    y_min = 0  # Set minimum to 0 for better visualization
    y_max = max(all_sales) * 1.05  # Add 5% padding
    
    # Plot training periods
    for i, (year, start_date, end_date) in enumerate(years):
        # Filter data for the year
        mask = (train_df['Datum'] >= start_date) & (train_df['Datum'] <= end_date)
        year_data = train_df[mask].copy()
        
        # Calculate daily totals
        daily_totals = year_data.groupby('Datum')['Umsatz'].sum().reset_index()
        
        # Plot
        axes[i].plot(daily_totals['Datum'], daily_totals['Umsatz'], 'b-', linewidth=1)
        axes[i].set_title(f'Training Period {year}-{year+1} (Aug-Jul)')
        axes[i].set_ylabel('Total Sales (€)')
        axes[i].grid(True, alpha=0.3)
        
        # Format x-axis to show months
        axes[i].xaxis.set_major_locator(mdates.MonthLocator())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        
        # Set axis limits
        axes[i].set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
        axes[i].set_ylim(y_min, y_max)
    
    # Validation period plot (2017-08-01 to 2018-07-31)
    val_daily = val_df.groupby('Datum')['Umsatz'].sum().reset_index()
    axes[4].plot(val_daily['Datum'], val_daily['Umsatz'], 'g-', linewidth=1)
    axes[4].set_title('Validation Period (2017-08-01 to 2018-07-31)')
    axes[4].set_ylabel('Total Sales (€)')
    axes[4].grid(True, alpha=0.3)
    axes[4].xaxis.set_major_locator(mdates.MonthLocator())
    axes[4].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[4].set_ylim(y_min, y_max)
    
    # Test period plot
    test_daily = test_df.groupby('Datum')['Umsatz'].sum().reset_index()
    axes[5].plot(test_daily['Datum'], test_daily['Umsatz'], 'r-', linewidth=1)
    axes[5].set_title('Test Period: Predicted Total Daily Sales')
    axes[5].set_ylabel('Total Sales (€)')
    axes[5].grid(True, alpha=0.3)
    axes[5].xaxis.set_major_locator(mdates.MonthLocator())
    axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[5].set_ylim(y_min, y_max)
    
    # Add year to the bottom x-axis
    axes[5].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'yearly_stacked_timeline.png'))
    plt.close()

def plot_validation_test_comparison(val_df, val_pred, test_df):
    """Create a single plot comparing validation, validation predictions, and test period sales"""
    print("\nPlotting validation and test comparison...")
    
    # Calculate daily totals
    val_daily = val_df.groupby('Datum')['Umsatz'].sum().reset_index()
    val_pred_daily = val_pred.groupby('Datum')['predicted_sales'].sum().reset_index()
    test_daily = test_df.groupby('Datum')['Umsatz'].sum().reset_index()
    
    # Ensure dates are datetime
    val_daily['Datum'] = pd.to_datetime(val_daily['Datum'])
    val_pred_daily['Datum'] = pd.to_datetime(val_pred_daily['Datum'])
    test_daily['Datum'] = pd.to_datetime(test_daily['Datum'])
    
    # Calculate days since start of validation period
    val_start = val_daily['Datum'].min()
    
    # Calculate days since start
    val_daily['days'] = (val_daily['Datum'] - val_start).dt.days
    val_pred_daily['days'] = (val_pred_daily['Datum'] - val_start).dt.days
    test_daily['days'] = (test_daily['Datum'] - val_start).dt.days
    
    # Create plot
    plt.figure(figsize=(15, 6))
    
    plt.plot(val_daily['days'], val_daily['Umsatz'], 
            label='Validation (Actual)', alpha=0.5, color='blue')
    plt.plot(val_pred_daily['days'], val_pred_daily['predicted_sales'], 
            label='Validation (Predicted)', alpha=0.5, color='green')
    plt.plot(test_daily['days'], test_daily['Umsatz'], 
            label='Test (Predicted)', alpha=0.5, color='red')
    
    plt.title('Comparison of Validation and Test Period Sales')
    plt.xlabel('Days since start of validation period')
    plt.ylabel('Total Daily Sales (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'validation_test_comparison.png'))
    plt.close()

def plot_separated_validation_test_comparison(val_df, val_pred, test_df):
    """Create separate plots for validation, validation predictions, and test period sales"""
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    # Calculate daily totals
    val_daily = val_df.groupby('Datum')['Umsatz'].sum().reset_index()
    val_pred_daily = val_pred.groupby('Datum')['predicted_sales'].sum().reset_index()
    test_daily = test_df.groupby('Datum')['Umsatz'].sum().reset_index()
    
    # Ensure dates are datetime
    val_daily['Datum'] = pd.to_datetime(val_daily['Datum'])
    val_pred_daily['Datum'] = pd.to_datetime(val_pred_daily['Datum'])
    test_daily['Datum'] = pd.to_datetime(test_daily['Datum'])
    
    # Calculate days since start of validation period for validation plots
    val_start = val_daily['Datum'].min()
    val_daily['days'] = (val_daily['Datum'] - val_start).dt.days
    val_pred_daily['days'] = (val_pred_daily['Datum'] - val_start).dt.days
    
    # Calculate days since start of test period for test plot
    test_start = test_daily['Datum'].min()
    test_daily['days'] = (test_daily['Datum'] - test_start).dt.days
    
    # Find global min and max for y-axis scaling
    y_min = min(
        val_daily['Umsatz'].min(),
        val_pred_daily['predicted_sales'].min(),
        test_daily['Umsatz'].min()
    )
    y_max = max(
        val_daily['Umsatz'].max(),
        val_pred_daily['predicted_sales'].max(),
        test_daily['Umsatz'].max()
    )
    
    # Add some padding to y-axis limits
    y_padding = (y_max - y_min) * 0.1
    y_min = max(0, y_min - y_padding)  # Don't go below 0
    y_max = y_max + y_padding
    
    # First subplot: Validation period actual sales
    ax1.plot(val_daily['days'], val_daily['Umsatz'], 
            color='green', alpha=0.8)
    ax1.set_title('Validation Period Sales')
    ax1.set_ylabel('Total Daily Sales (€)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(y_min, y_max)
    
    # Second subplot: Validation period predictions
    ax2.plot(val_pred_daily['days'], val_pred_daily['predicted_sales'], 
            color='blue', alpha=0.8)
    ax2.set_title('Validation Predictions')
    ax2.set_ylabel('Total Daily Sales (€)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(y_min, y_max)
    
    # Third subplot: Test period
    ax3.plot(test_daily['days'], test_daily['Umsatz'], 
            color='red', alpha=0.8)
    ax3.set_title('Test Period Sales')
    ax3.set_xlabel('Days since period start')
    ax3.set_ylabel('Total Sales (€)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_VIZ, 'separated_validation_test_comparison.png'))
    plt.close()

def create_interactive_timeline(train_df, val_df, test_df):
    """Create an interactive timeline visualization using plotly"""
    print("\nCreating interactive timeline visualization...")
    
    # Calculate daily totals
    train_daily = train_df.groupby('Datum')['Umsatz'].sum().reset_index()
    val_daily = val_df.groupby('Datum')['Umsatz'].sum().reset_index()
    test_daily = test_df.groupby('Datum')['Umsatz'].sum().reset_index()
    
    # Ensure dates are datetime
    train_daily['Datum'] = pd.to_datetime(train_daily['Datum'])
    val_daily['Datum'] = pd.to_datetime(val_daily['Datum'])
    test_daily['Datum'] = pd.to_datetime(test_daily['Datum'])
    
    # Convert timestamps to strings for JSON serialization
    timeline_data = {
        'training_period': [
            {'Datum': d.strftime('%Y-%m-%d'), 'Umsatz': u} 
            for d, u in zip(train_daily['Datum'], train_daily['Umsatz'])
        ],
        'validation_period': [
            {'Datum': d.strftime('%Y-%m-%d'), 'Umsatz': u} 
            for d, u in zip(val_daily['Datum'], val_daily['Umsatz'])
        ],
        'test_period': [
            {'Datum': d.strftime('%Y-%m-%d'), 'Umsatz': u} 
            for d, u in zip(test_daily['Datum'], test_daily['Umsatz'])
        ]
    }
    
    # Save timeline data
    with open(os.path.join(ANALYSIS_OUTPUT, 'timeline_data.json'), 'w') as f:
        json.dump(timeline_data, f)
    
    # Create figure with subplots
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=('Training Period: Total Daily Sales (2013-08-01 to 2017-07-31)',
                                     'Validation Period: Total Daily Sales (2017-08-01 to 2018-07-31)',
                                     'Test Period: Predicted Total Daily Sales'))
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=train_daily['Datum'], y=train_daily['Umsatz'],
                  mode='lines', name='Training', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=val_daily['Datum'], y=val_daily['Umsatz'],
                  mode='lines', name='Validation', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=test_daily['Datum'], y=test_daily['Umsatz'],
                  mode='lines', name='Test', line=dict(color='red')),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        title_text="Interactive Timeline of Sales",
        hovermode='x unified'
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Total Sales (€)")
    
    # Update x-axes labels
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    # Generate HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Sales Timeline</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ margin: 0; padding: 20px; }}
            #timeline {{ width: 100%; height: 900px; }}
        </style>
    </head>
    <body>
        <div id="timeline">
            {fig.to_html(full_html=False, include_plotlyjs=False)}
        </div>
    </body>
    </html>
    """
    
    # Save the HTML file
    html_path = os.path.join(ANALYSIS_VIZ, 'interactive_timeline.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"Interactive timeline saved to: {html_path}")

def analyze_special_events(train_df, val_df, test_df):
    """Analyze predictions for Easter Saturday and Windjammer Parade"""
    # Load special event dates
    easter_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'easter_saturday.csv')
    windjammer_path = os.path.join(MODEL_ROOT, '..', '0_DataPreparation', 'input', 'compiled_data', 'windjammer.csv')
    
    # Load Easter dates (handle the comma in the CSV)
    easter_df = pd.read_csv(easter_path, names=['Datum', 'Flag'], header=0)
    easter_df['Datum'] = pd.to_datetime(easter_df['Datum'])
    
    # Load Windjammer dates
    windjammer_df = pd.read_csv(windjammer_path)
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    
    # Filter for test period dates (2018-2019)
    test_easter_dates = easter_df[easter_df['Datum'].dt.year.isin([2018, 2019])]['Datum']
    test_windjammer_dates = windjammer_df[windjammer_df['Datum'].dt.year.isin([2018, 2019])]['Datum']
    
    print("\nTest Easter Dates:")
    print(test_easter_dates)
    
    # Convert test_df dates to datetime if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(test_df['Datum']):
        test_df['Datum'] = pd.to_datetime(test_df['Datum'])
    
    print("\nTest Data Dates Sample:")
    print(test_df['Datum'].head())
    print("\nTest Data Date Range:")
    print(f"Start: {test_df['Datum'].min()}")
    print(f"End: {test_df['Datum'].max()}")
    
    # Analyze Easter Saturday predictions
    easter_predictions = test_df[test_df['Datum'].isin(test_easter_dates)]
    print("\nEaster Saturday Predictions:")
    print(easter_predictions[['Datum', 'Warengruppe_Name', 'Umsatz']].sort_values(['Datum', 'Warengruppe_Name']))
    
    # Analyze Windjammer Parade predictions
    windjammer_predictions = test_df[test_df['Datum'].isin(test_windjammer_dates)]
    print("\nWindjammer Parade Predictions:")
    print(windjammer_predictions[['Datum', 'Warengruppe_Name', 'Umsatz']].sort_values(['Datum', 'Warengruppe_Name']))
    
    # Analyze historical Easter Saturday sales
    historical_easter = train_df[train_df['Datum'].isin(easter_df['Datum'])]
    print("\nHistorical Easter Saturday Sales:")
    easter_summary = historical_easter.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std', 'min', 'max']).round(2)
    easter_summary.index = easter_summary.index.map(get_warengruppe_name)
    print(easter_summary)
    
    # Analyze historical Windjammer Parade sales
    historical_windjammer = train_df[train_df['Datum'].isin(windjammer_df['Datum'])]
    print("\nHistorical Windjammer Parade Sales:")
    windjammer_summary = historical_windjammer.groupby('Warengruppe')['Umsatz'].agg(['mean', 'std', 'min', 'max']).round(2)
    windjammer_summary.index = windjammer_summary.index.map(get_warengruppe_name)
    print(windjammer_summary)
    
    # Save results to file
    with open(os.path.join(ANALYSIS_OUTPUT, 'special_events_analysis.txt'), 'w') as f:
        f.write("Easter Saturday Predictions:\n")
        f.write(easter_predictions[['Datum', 'Warengruppe_Name', 'Umsatz']].sort_values(['Datum', 'Warengruppe_Name']).to_string())
        f.write("\n\nWindjammer Parade Predictions:\n")
        f.write(windjammer_predictions[['Datum', 'Warengruppe_Name', 'Umsatz']].sort_values(['Datum', 'Warengruppe_Name']).to_string())
        f.write("\n\nHistorical Easter Saturday Sales:\n")
        f.write(easter_summary.to_string())
        f.write("\n\nHistorical Windjammer Parade Sales:\n")
        f.write(windjammer_summary.to_string())

def main():
    """Main analysis function"""
    print(f"\nAnalyzing {MODEL_NAME} predictions...")
    
    # Load data
    train_df, val_df, test_df, train_pred, val_pred, metrics = load_data()
    
    # Analyze metrics
    analyze_metrics(metrics)
    
    # Plot sales distribution
    plot_sales_distribution(train_df, val_df, test_df)
    
    # Analyze by product
    analyze_by_product(train_df, val_df, test_df)
    
    # Analyze temporal patterns
    analyze_temporal_patterns(train_df, val_df, test_df)
    
    # Plot timeline sales
    plot_timeline_sales(train_df, val_df, test_df)
    
    # Plot weekday patterns
    plot_weekday_patterns(train_df, val_df, test_df)
    
    # Plot yearly stacked timeline
    plot_yearly_stacked_timeline(train_df, val_df, test_df)
    
    # Plot validation and test comparisons
    plot_validation_test_comparison(val_df, val_pred, test_df)
    plot_separated_validation_test_comparison(val_df, val_pred, test_df)
    
    # Analyze special events
    analyze_special_events(train_df, val_df, test_df)
    
    # Create interactive timeline
    create_interactive_timeline(train_df, val_df, test_df)
    
    print("\nAnalysis complete! Check the analysis/results and analysis/visualizations directories for output files.")

if __name__ == "__main__":
    main()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Set up paths
SCRIPT_PATH = os.path.abspath(__file__)
PROJECT_ROOT = "/Users/admin/Dropbox/@PARA/Projects/opencampus/bakery_sales_prediction"
TRAIN_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "competition_data", "train.csv")
EASTER_SATURDAY_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "easter_saturday.csv")
WINDJAMMER_PATH = os.path.join(PROJECT_ROOT, "0_DataPreparation", "input", "compiled_data", "windjammer.csv")

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "1_DatasetCharacteristics", "output", "special_events")
print(f"Creating output directory: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Product group dictionary
WARENGRUPPEN = {
    1: "Brot",
    2: "Brötchen",
    3: "Croissant",
    4: "Konditorei",
    5: "Kuchen",
    6: "Saisonbrot"
}

# Function to calculate mean and confidence interval
def mean_confidence_interval(data, confidence=0.95):
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = sem * stats.t.ppf((1 + confidence) / 2, len(data)-1)
    return mean, ci

# Function to create bar plot
def create_bar_plot(results_df, title, output_file, min_samples=2):
    # Filter out events with insufficient samples
    results_df = results_df[results_df['Samples'] >= min_samples].copy()
    
    if len(results_df) == 0:
        print(f"Skipping plot '{title}' - insufficient data")
        return
        
    plt.figure(figsize=(10, 6))
    bar_plot = plt.bar(results_df['Event'], results_df['Mean'], yerr=results_df['CI'], 
                       capsize=5, color='skyblue', alpha=0.7)

    # Add warning to title if any event has small sample size
    if any((results_df['Samples'] < 5) & (results_df['Samples'] >= min_samples)):
        title += "\n(Warning: Some events have less than 5 samples)"

    plt.title(title, fontsize=12, pad=15)
    plt.ylabel('Umsatz', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    # Add value labels on top of each bar
    for i, bar in enumerate(bar_plot):
        height = bar.get_height()
        ci = results_df.iloc[i]['CI']
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}\n(±{ci:.1f})',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

# Load and prepare data
print(f"Loading training data from: {TRAIN_PATH}")
train_df = pd.read_csv(TRAIN_PATH)
print(f"Loaded {len(train_df)} rows from training data")
train_df["Datum"] = pd.to_datetime(train_df["Datum"])

# Print training data date range
print("\nTraining data date range:")
print(f"Start: {train_df['Datum'].min()}")
print(f"End: {train_df['Datum'].max()}")

# Easter Saturday data
print(f"\nLoading Easter Saturday data from: {EASTER_SATURDAY_PATH}")
easter_df = pd.read_csv(EASTER_SATURDAY_PATH, skiprows=1, names=["Datum", "is_easter_saturday"])
easter_df["Datum"] = pd.to_datetime(easter_df["Datum"], format="%Y-%m-%d")
# Filter Easter dates to training data range
easter_df = easter_df[
    (easter_df["Datum"] >= train_df["Datum"].min()) & 
    (easter_df["Datum"] <= train_df["Datum"].max())
]

# Windjammer data
print(f"\nLoading Windjammer data from: {WINDJAMMER_PATH}")
windjammer_df = pd.read_csv(WINDJAMMER_PATH)
windjammer_df["Datum"] = pd.to_datetime(windjammer_df["Datum"])
# Filter Windjammer dates to training data range
windjammer_df = windjammer_df[
    (windjammer_df["Datum"] >= train_df["Datum"].min()) & 
    (windjammer_df["Datum"] <= train_df["Datum"].max())
]

# Print available dates for each event
print("\nAvailable dates for each event (within training data range):")
print("Silvester dates:", sorted(train_df[train_df['Datum'].dt.strftime('%m-%d') == '12-31']['Datum'].unique()))
print("Easter Saturday dates:", sorted(easter_df['Datum'].unique()))
print("Windjammer dates:", sorted(windjammer_df['Datum'].unique()))

# First: Create plot for total sales across all product groups
# Group by date and product group first, then sum up sales
daily_group_sales = train_df.groupby(['Datum', 'Warengruppe'])['Umsatz'].sum().reset_index()
daily_sales = daily_group_sales.groupby('Datum')['Umsatz'].sum().reset_index()

events_total = {
    'Silvester': daily_sales[daily_sales['Datum'].dt.strftime('%m-%d') == '12-31']['Umsatz'],
    'Oster-Samstag': daily_sales[daily_sales['Datum'].isin(easter_df['Datum'])]['Umsatz'],
    'Windjammerparade': daily_sales[daily_sales['Datum'].isin(windjammer_df['Datum'])]['Umsatz']
}

results_total = {
    'Event': [],
    'Mean': [],
    'CI': [],
    'Samples': []
}

for event_name, data in events_total.items():
    print(f"\nProcessing total sales for {event_name} with {len(data)} samples")
    mean, ci = mean_confidence_interval(data)
    results_total['Event'].append(event_name)
    results_total['Mean'].append(mean)
    results_total['CI'].append(ci)
    results_total['Samples'].append(len(data))

results_df_total = pd.DataFrame(results_total)
print("\nTotal Sales Results:")
print(results_df_total)

create_bar_plot(
    results_df_total,
    'Durchschnittlicher Gesamtumsatz an speziellen Events mit Konfidenzintervallen',
    os.path.join(OUTPUT_DIR, 'special_events_total_barplot.png')
)

# Second: Create separate plots for each product group
for warengruppe, name in WARENGRUPPEN.items():
    print(f"\nAnalyzing {name} (Warengruppe {warengruppe}):")
    group_sales = daily_group_sales[daily_group_sales['Warengruppe'] == warengruppe]
    
    # Print dates available for this product group
    print(f"Available dates for {name}:")
    print("Silvester:", sorted(group_sales[group_sales['Datum'].dt.strftime('%m-%d') == '12-31']['Datum'].unique()))
    print("Easter Saturday:", sorted(group_sales[group_sales['Datum'].isin(easter_df['Datum'])]['Datum'].unique()))
    print("Windjammer:", sorted(group_sales[group_sales['Datum'].isin(windjammer_df['Datum'])]['Datum'].unique()))
    
    events_group = {
        'Silvester': group_sales[group_sales['Datum'].dt.strftime('%m-%d') == '12-31']['Umsatz'],
        'Oster-Samstag': group_sales[group_sales['Datum'].isin(easter_df['Datum'])]['Umsatz'],
        'Windjammerparade': group_sales[group_sales['Datum'].isin(windjammer_df['Datum'])]['Umsatz']
    }
    
    results_group = {
        'Event': [],
        'Mean': [],
        'CI': [],
        'Samples': []
    }
    
    for event_name, data in events_group.items():
        print(f"Processing {name} sales for {event_name} with {len(data)} samples")
        mean, ci = mean_confidence_interval(data) if len(data) >= 2 else (np.nan, np.nan)
        results_group['Event'].append(event_name)
        results_group['Mean'].append(mean)
        results_group['CI'].append(ci)
        results_group['Samples'].append(len(data))
    
    results_df_group = pd.DataFrame(results_group)
    print(f"\n{name} Results:")
    print(results_df_group)
    
    create_bar_plot(
        results_df_group,
        f'Durchschnittlicher Umsatz {name} an speziellen Events mit Konfidenzintervallen',
        os.path.join(OUTPUT_DIR, f'special_events_{name.lower()}_barplot.png')
    )

print("\nDone!")

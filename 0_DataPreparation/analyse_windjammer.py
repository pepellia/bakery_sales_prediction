import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, '0_DataPreparation'))
from config import (TRAIN_PATH, COMPILED_DATA_DIR, VIZ_DIR,
                   WARENGRUPPEN, get_warengruppe_name)

def save_analysis_results(stats_dict, summary_text, output_dir):
    """Save analysis results to files"""
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw statistics
    stats_file = os.path.join(output_dir, f'windjammer_statistics_{timestamp}.txt')
    with open(stats_file, 'w') as f:
        f.write("Windjammer Analysis Statistics\n")
        f.write("============================\n\n")
        for title, stats in stats_dict.items():
            f.write(f"{title}\n")
            f.write("-" * len(title) + "\n")
            f.write(f"{stats}\n\n")
    
    # Save markdown summary
    summary_file = os.path.join(output_dir, f'windjammer_summary_{timestamp}.md')
    with open(summary_file, 'w') as f:
        f.write("# Windjammer Analysis Summary\n\n")
        f.write(summary_text)
    
    return stats_file, summary_file

def analyze_windjammer_impact():
    """Analyze the impact of Windjammer parade on sales"""
    print("Analyzing Windjammer parade impact on sales...")
    
    # Create output directory
    output_dir = os.path.join(SCRIPT_DIR, 'output', 'windjammer_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    train_df = pd.read_csv(TRAIN_PATH)
    windjammer_df = pd.read_csv(os.path.join(COMPILED_DATA_DIR, 'windjammer.csv'))
    
    # Convert dates to datetime
    train_df['Datum'] = pd.to_datetime(train_df['Datum'])
    windjammer_df['Datum'] = pd.to_datetime(windjammer_df['Datum'])
    
    # Add weekday (0 = Monday, 6 = Sunday)
    train_df['Wochentag'] = train_df['Datum'].dt.dayofweek
    train_df['Wochentag_Name'] = train_df['Datum'].dt.day_name()
    
    # Add Windjammer parade flag
    train_df['Windjammerparade'] = train_df['Datum'].isin(windjammer_df['Datum']).astype(int)
    
    # Prepare statistics dictionary
    stats_dict = {}
    
    # 1. Overall sales comparison
    print("\n1. Sales comparison: General vs. Windjammer Parade")
    general_stats = train_df[train_df['Windjammerparade'] == 0]['Umsatz'].describe()
    parade_stats = train_df[train_df['Windjammerparade'] == 1]['Umsatz'].describe()
    
    stats_dict['General Sales'] = general_stats
    stats_dict['Parade Day Sales'] = parade_stats
    
    print("\nGeneral sales:")
    print(general_stats)
    print("\nSales on parade days:")
    print(parade_stats)
    
    # 2. Saturday comparison
    saturdays_df = train_df[train_df['Wochentag'] == 5]  # 5 = Saturday
    regular_sat_stats = saturdays_df[saturdays_df['Windjammerparade'] == 0]['Umsatz'].describe()
    parade_sat_stats = saturdays_df[saturdays_df['Windjammerparade'] == 1]['Umsatz'].describe()
    
    stats_dict['Regular Saturday Sales'] = regular_sat_stats
    stats_dict['Parade Saturday Sales'] = parade_sat_stats
    
    print("\n2. Sales comparison: Regular Saturdays vs. Parade Saturdays")
    print("\nSales on regular Saturdays:")
    print(regular_sat_stats)
    print("\nSales on parade Saturdays:")
    print(parade_sat_stats)
    
    # 3. Product group analysis
    product_impacts = []
    print("\n3. Impact by product group:")
    for group_id, group_name in WARENGRUPPEN.items():
        group_data = train_df[train_df['Warengruppe'] == group_id]
        normal_avg = group_data[group_data['Windjammerparade'] == 0]['Umsatz'].mean()
        parade_avg = group_data[group_data['Windjammerparade'] == 1]['Umsatz'].mean()
        if parade_avg > 0:  # Check if we have parade data for this group
            increase = ((parade_avg / normal_avg) - 1) * 100
            impact = f"{group_name}: {increase:.1f}% {'increase' if increase > 0 else 'decrease'}"
            product_impacts.append((increase, impact))
            print(impact)
    
    # Calculate overall impacts
    avg_normal = train_df[train_df['Windjammerparade'] == 0]['Umsatz'].mean()
    avg_parade = train_df[train_df['Windjammerparade'] == 1]['Umsatz'].mean()
    avg_normal_saturday = saturdays_df[saturdays_df['Windjammerparade'] == 0]['Umsatz'].mean()
    avg_parade_saturday = saturdays_df[saturdays_df['Windjammerparade'] == 1]['Umsatz'].mean()
    
    overall_increase = ((avg_parade / avg_normal) - 1) * 100
    saturday_increase = ((avg_parade_saturday / avg_normal_saturday) - 1) * 100
    
    print(f"\nPercentage differences:")
    print(f"Parade vs. normal days: {overall_increase:.1f}% increase")
    print(f"Parade vs. regular Saturdays: {saturday_increase:.1f}% increase")
    
    # Create summary text
    summary_text = f"""## Overall Impact
- {overall_increase:.1f}% increase in sales on parade days vs. normal days
- {saturday_increase:.1f}% increase vs. regular Saturdays

## Product-specific Impact
"""
    # Sort product impacts by percentage
    product_impacts.sort(reverse=True)
    for _, impact in product_impacts:
        summary_text += f"- {impact}\n"
    
    summary_text += f"""
## Key Statistics
- Average parade day sales: €{avg_parade:.2f}
- Average normal day sales: €{avg_normal:.2f}
- Average parade Saturday sales: €{avg_parade_saturday:.2f}
- Average regular Saturday sales: €{avg_normal_saturday:.2f}

## Visualizations
The following visualizations have been generated:
1. General sales comparison (box plot)
2. Saturday-specific comparison (box plot)
3. Product group comparison on parade days (box plot)
"""
    
    # Create visualizations
    viz_output_dir = os.path.join(VIZ_DIR, 'windjammer_analysis')
    os.makedirs(viz_output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Box-Plot: General comparison
    plt.subplot(1, 3, 1)
    sns.boxplot(data=train_df, x='Windjammerparade', y='Umsatz')
    plt.title('Sales: General vs. Windjammer Parade')
    plt.xlabel('Windjammer Parade')
    plt.xticks([0, 1], ['Other Days', 'Parade'])
    
    # Box-Plot: Saturday comparison
    plt.subplot(1, 3, 2)
    sns.boxplot(data=saturdays_df, x='Windjammerparade', y='Umsatz')
    plt.title('Sales: Regular vs. Parade Saturdays')
    plt.xlabel('Windjammer Parade')
    plt.xticks([0, 1], ['Regular', 'Parade'])
    
    # Box-Plot: Product group comparison
    plt.subplot(1, 3, 3)
    sns.boxplot(data=train_df[train_df['Windjammerparade'] == 1],
                x='Warengruppe', y='Umsatz')
    plt.title('Sales by Product (Parade Days)')
    plt.xlabel('Product Group')
    plt.xticks(range(len(WARENGRUPPEN)), 
               [name[:10] for name in WARENGRUPPEN.values()],
               rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_output_dir, 'windjammer_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save analysis results
    stats_file, summary_file = save_analysis_results(stats_dict, summary_text, output_dir)
    
    print(f"\nAnalysis results saved to:")
    print(f"- Statistics: {stats_file}")
    print(f"- Summary: {summary_file}")
    print(f"- Visualizations: {viz_output_dir}")

if __name__ == "__main__":
    analyze_windjammer_impact()

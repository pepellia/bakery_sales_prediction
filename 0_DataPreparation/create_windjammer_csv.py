import pandas as pd
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, '0_DataPreparation'))
from config import KIWO_PATH, COMPILED_DATA_DIR

def create_windjammer_csv():
    """Create Windjammer events dataset from Kieler Woche data"""
    print("Creating Windjammer events dataset...")
    
    # Read Kieler Woche data
    kiwo_df = pd.read_csv(KIWO_PATH)
    kiwo_df['Datum'] = pd.to_datetime(kiwo_df['Datum'])
    
    # Add weekday (0 = Monday, 6 = Sunday)
    kiwo_df['Wochentag'] = kiwo_df['Datum'].dt.dayofweek
    
    # Group by year to identify Kieler Woche periods
    kiwo_df['Jahr'] = kiwo_df['Datum'].dt.year
    kiwo_gruppen = kiwo_df.groupby('Jahr')
    
    # List for Windjammer parade days
    windjammer_tage = []
    
    # For each year
    for jahr, gruppe in kiwo_gruppen:
        # Find all Saturdays (weekday 5) during Kieler Woche
        samstage = gruppe[gruppe['Wochentag'] == 5].sort_values('Datum')
        
        # If there are at least two Saturdays, take the second one
        if len(samstage) >= 2:
            windjammer_tag = samstage.iloc[1]['Datum']
            windjammer_tage.append({
                'Datum': windjammer_tag,
                'Windjammerparade': 1
            })
    
    # Create DataFrame
    windjammer_df = pd.DataFrame(windjammer_tage)
    
    # Create output path
    output_path = os.path.join(COMPILED_DATA_DIR, 'windjammer.csv')
    
    # Save as CSV
    windjammer_df.to_csv(output_path, index=False)
    
    print("\nWindjammer parade days:")
    print(windjammer_df['Datum'].dt.strftime('%Y-%m-%d').to_string(index=False))
    print(f"\nDataset saved to: {output_path}")

if __name__ == "__main__":
    create_windjammer_csv()

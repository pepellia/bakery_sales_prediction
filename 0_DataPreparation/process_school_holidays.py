import pandas as pd
import re
from datetime import datetime, timedelta, date
import os

def convert_umlauts(text):
    """Convert German umlauts to their alternative spelling."""
    umlaut_map = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss',
        'Ä': 'Ae',
        'Ö': 'Oe',
        'Ü': 'Ue'
    }
    for umlaut, replacement in umlaut_map.items():
        text = text.replace(umlaut, replacement)
    return text

def parse_date(date_str):
    """Convert German date string to datetime object."""
    return datetime.strptime(date_str.strip(), '%d.%m.%Y').date()

def get_date_range(start_date, end_date):
    """Generate all dates between start and end date inclusive."""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime('%Y-%m-%d'))  # Store as string
        current_date += timedelta(days=1)
    return dates

def process_holidays(file_path):
    """Process the markdown file and extract holiday information."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content by year sections
    year_sections = content.split('\n# ')[1:]  # Skip the first empty section
    
    state_holidays = {}
    
    for year_section in year_sections:
        # Split section into state sections
        state_sections = re.split(r'\n##\s+', year_section)[1:]  # Skip the year header
        
        for section in state_sections:
            # Extract state name and holiday entries
            lines = section.strip().split('\n')
            state_name = lines[0].strip().rstrip(':')  # Remove any trailing colon
            holiday_entries = [line.strip() for line in lines[1:] if line.strip().startswith('-')]
            
            if state_name not in state_holidays:
                state_holidays[state_name] = set()  # Initialize set for new state
            
            print(f"\nProcessing {state_name}:")
            for entry in holiday_entries:
                # Extract date range using regex
                match = re.search(r'(\d{2}\.\d{2}\.\d{4})\s*-\s*(\d{2}\.\d{2}\.\d{4})', entry)
                if match:
                    start_date = parse_date(match.group(1))
                    end_date = parse_date(match.group(2))
                    print(f"Holiday period: {start_date} to {end_date}")
                    
                    # Get all dates in the range
                    holiday_dates = get_date_range(start_date, end_date)
                    state_holidays[state_name].update(holiday_dates)  # Add to set
            
            print(f"Total holiday dates for {state_name}: {len(state_holidays[state_name])}")
    
    # Convert sets to sorted lists
    return {state: sorted(dates) for state, dates in state_holidays.items()}

def create_holiday_csv(state_name, holiday_dates, output_dir):
    """Create a CSV file for a state with holiday flags."""
    # Create date range from 2012 to 2019
    start_date = date(2012, 1, 1)
    end_date = date(2019, 12, 31)
    
    dates = []
    holiday_flags = []
    
    current_date = start_date
    holiday_count = 0
    while current_date <= end_date:
        current_date_str = current_date.strftime('%Y-%m-%d')
        dates.append(current_date_str)
        is_holiday = 1 if current_date_str in holiday_dates else 0
        holiday_flags.append(is_holiday)
        holiday_count += is_holiday
        current_date += timedelta(days=1)
    
    print(f"\nSummary for {state_name}:")
    print(f"Total days: {len(dates)}")
    print(f"Total holidays: {holiday_count}")
    
    df = pd.DataFrame({
        'date': dates,
        'is_school_holiday': holiday_flags
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert state name to filename-friendly format
    filename = convert_umlauts(state_name.lower().replace(" ", "_"))
    output_file = os.path.join(output_dir, f'school_holidays_{filename}.csv')
    df.to_csv(output_file, index=False)
    print(f"Created {output_file}")

def main():
    input_file = '0_DataPreparation/input/compiled_data/school-holidays-2012-2019.md'
    output_dir = '0_DataPreparation/input/compiled_data/school_holidays'
    
    # Process the holiday data
    state_holidays = process_holidays(input_file)
    
    # Create CSV files for each state
    for state_name, holiday_dates in state_holidays.items():
        create_holiday_csv(state_name, holiday_dates, output_dir)

if __name__ == '__main__':
    main() 
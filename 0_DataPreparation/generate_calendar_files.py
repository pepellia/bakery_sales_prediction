import datetime
import csv
import os

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'input', 'compiled_data')

def easter_sunday(year):
    """
    Returns Easter Sunday as a datetime.date for the given year.
    Based on the Anonymous Gregorian Computus.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = 1 + (h + l - 7*m + 114) % 31
    return datetime.date(year, month, day)

def german_federal_holidays(year):
    """
    Return a set of datetime.date objects for all German
    *nationwide* public holidays in a given year.
    """
    # Fixed-date holidays
    holidays = set([
        datetime.date(year, 1, 1),   # Neujahr
        datetime.date(year, 5, 1),   # Tag der Arbeit
        datetime.date(year, 10, 3),  # Tag der Deutschen Einheit
        datetime.date(year, 12, 25), # Erster Weihnachtstag
        datetime.date(year, 12, 26), # Zweiter Weihnachtstag
    ])
    
    # Easter-based holidays
    easter = easter_sunday(year)
    good_friday = easter - datetime.timedelta(days=2)
    easter_monday = easter + datetime.timedelta(days=1)
    ascension = easter + datetime.timedelta(days=39)     # Christi Himmelfahrt
    whit_monday = easter + datetime.timedelta(days=50)   # Pfingstmontag
    
    holidays.update([good_friday, easter_monday, ascension, whit_monday])
    
    return holidays

# Generate all days from 2012-01-01 to 2019-12-31
start_date = datetime.date(2012, 1, 1)
end_date = datetime.date(2019, 12, 31)

all_days = []
current = start_date
while current <= end_date:
    all_days.append(current)
    current += datetime.timedelta(days=1)

# Precompute the holiday set for each year to speed up checks
holidays_by_year = {}
for y in range(2012, 2020):
    holidays_by_year[y] = german_federal_holidays(y)

# Helper functions
def is_public_holiday(d):
    return d in holidays_by_year[d.year]

def is_weekend(d):
    return d.weekday() >= 5  # Monday=0, Sunday=6

def is_workday(d):
    return (d.weekday() < 5) and (not is_public_holiday(d))

# Identify bridge days (Brückentage)
bridge_days = set()

for d in all_days:
    if d.weekday() < 5:  # 0=Mon,4=Fri
        yesterday = d - datetime.timedelta(days=1)
        tomorrow = d + datetime.timedelta(days=1)
        
        if is_public_holiday(yesterday) and is_weekend(tomorrow):
            bridge_days.add(d)
        elif is_weekend(yesterday) and is_public_holiday(tomorrow):
            bridge_days.add(d)

# Identify the day-before-holiday
day_before_holiday = set()

for d in all_days:
    tomorrow = d + datetime.timedelta(days=1)
    if tomorrow <= end_date:
        if is_public_holiday(tomorrow):
            day_before_holiday.add(d)

# Identify paydays
paydays = set()

def get_last_working_day(year, month):
    if month == 12:
        last_day = datetime.date(year, 12, 31)
    else:
        next_month = datetime.date(year, month+1, 1)
        last_day = next_month - datetime.timedelta(days=1)

    while is_weekend(last_day) or is_public_holiday(last_day):
        last_day -= datetime.timedelta(days=1)
    return last_day

def get_common_payday(year, month, day_of_month):
    try:
        d = datetime.date(year, month, day_of_month)
    except ValueError:
        return None
    while is_weekend(d) or is_public_holiday(d):
        d -= datetime.timedelta(days=1)
    return d

for y in range(2012, 2020):
    for m in range(1, 13):
        # Last working day of this month
        lwd = get_last_working_day(y, m)
        paydays.add(lwd)
        
        # 15th (or preceding working day if weekend/holiday)
        pd_15 = get_common_payday(y, m, 15)
        if pd_15:
            paydays.add(pd_15)
        
        # 25th (or preceding working day if weekend/holiday)
        pd_25 = get_common_payday(y, m, 25)
        if pd_25:
            paydays.add(pd_25)

# Write out the CSVs
with open(os.path.join(OUTPUT_DIR, "bridge_days.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Datum", "is_bridge_day"])
    for d in sorted(bridge_days):
        writer.writerow([d.isoformat(), 1])

with open(os.path.join(OUTPUT_DIR, "day_before_holiday.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Datum", "is_day_before_holiday"])
    for d in sorted(day_before_holiday):
        writer.writerow([d.isoformat(), 1])

with open(os.path.join(OUTPUT_DIR, "payday.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Datum", "is_payday"])
    for d in sorted(paydays):
        writer.writerow([d.isoformat(), 1])

print("CSV files generated in", OUTPUT_DIR)
print("  - bridge_days.csv")
print("  - day_before_holiday.csv")
print("  - payday.csv")

# Simple script to display finalrun.csv parts
# This follows your original notebook approach

import csv
import sys

# IMPORTANT: Increase field size limit for large CSV fields
csv.field_size_limit(sys.maxsize)

# Read the CSV file
with open('finalrun.csv', 'r') as f:
    reader = csv.reader(f)
    finalrun = list(reader)

print(f"Total rows in finalrun: {len(finalrun)}")
print(f"Total columns: {len(finalrun[0])}\n")

# Display header
print("="*80)
print("HEADER (Column Names)")
print("="*80)
header = finalrun[0]
for i, col in enumerate(header, 1):
    print(f"{i:2d}. {col}")

# Display each data row
print("\n" + "="*80)
print("DATA ROWS")
print("="*80)

for row_idx in range(1, len(finalrun)):
    print(f"\n{'='*80}")
    print(f"ROW {row_idx}: {finalrun[row_idx][1] if len(finalrun[row_idx]) > 1 else 'N/A'}")  # Label column
    print(f"{'='*80}")
    
    # Display each column value for this row
    for col_idx, (col_name, value) in enumerate(zip(header, finalrun[row_idx])):
        if col_idx < 5:  # Show first 5 columns always
            print(f"  {col_name}: {value}")
    
    print(f"\n  ... and {len(header) - 5} more columns")
    
    # Ask if user wants to see all columns for this row
    show_all = input(f"\n  Show all columns for Row {row_idx}? (y/n/q to quit): ").strip().lower()
    
    if show_all == 'q':
        break
    elif show_all == 'y':
        for col_idx, (col_name, value) in enumerate(zip(header, finalrun[row_idx])):
            print(f"    {col_idx+1}. {col_name}: {value}")

print("\n\nDone!")

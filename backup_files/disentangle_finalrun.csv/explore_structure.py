#!/usr/bin/env python3
"""
Minimal script to explore finalrun.csv structure safely
Shows structure without overwhelming output
"""
import csv
import sys

# CRITICAL: Increase field size limit for large CSV fields
csv.field_size_limit(sys.maxsize)

# Load the CSV
with open('finalrun.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)

header = data[0]
rows = data[1:]

print("="*80)
print("FINALRUN.CSV STRUCTURE")
print("="*80)
print(f"Total columns: {len(header)}")
print(f"Total data rows: {len(rows)}")

print("\n" + "="*80)
print("COLUMN NAMES (organized by prefix)")
print("="*80)

# Group columns by their prefix
prefixes = {}
for col in header:
    if '_' in col:
        prefix = col.split('_')[0]
    else:
        prefix = col
    
    if prefix not in prefixes:
        prefixes[prefix] = []
    prefixes[prefix].append(col)

for prefix, cols in sorted(prefixes.items()):
    print(f"\n{prefix.upper()} ({len(cols)} columns):")
    for col in cols[:5]:  # Show first 5 only
        print(f"  - {col}")
    if len(cols) > 5:
        print(f"  ... and {len(cols)-5} more")

print("\n" + "="*80)
print("DATA ROWS (labels only)")
print("="*80)
label_idx = header.index('label') if 'label' in header else -1
for i, row in enumerate(rows, 1):
    if label_idx >= 0 and label_idx < len(row):
        print(f"Row {i}: {row[label_idx]}")
    else:
        print(f"Row {i}: [no label]")

print("\n" + "="*80)
print("SAMPLE DATA (first row, first 5 columns)")
print("="*80)
if rows:
    for i in range(min(5, len(header))):
        value = rows[0][i] if i < len(rows[0]) else 'N/A'
        # Truncate long values
        if len(str(value)) > 50:
            value = str(value)[:50] + "..."
        print(f"{header[i]}: {value}")

print("\n" + "="*80)
print("\nTo see specific categories, use one of the other display scripts:")
print("  - display_simple.py (interactive)")
print("  - display_finalrun.py (organized by category)")
print("  - disentangle_finalrun_complete.ipynb (Jupyter notebook)")

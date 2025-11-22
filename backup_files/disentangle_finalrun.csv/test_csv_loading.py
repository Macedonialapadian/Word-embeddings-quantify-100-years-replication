#!/usr/bin/env python3
"""
Quick test to verify finalrun.csv can be loaded properly
"""
import csv
import sys

print("Testing CSV loading with field size limit fix...\n")

# Increase field size limit
csv.field_size_limit(sys.maxsize)
print(f"✓ Field size limit set to: {csv.field_size_limit()}")

# Try to load the CSV
try:
    with open('finalrun.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    
    print(f"✓ Successfully loaded CSV")
    print(f"✓ Rows: {len(data)}")
    print(f"✓ Columns: {len(data[0]) if data else 0}")
    
    # Show some basic info
    if len(data) > 0:
        header = data[0]
        print(f"\n✓ First few columns: {', '.join(header[:5])}")
        print(f"✓ Last few columns: {', '.join(header[-3:])}")
    
    if len(data) > 1:
        print(f"\n✓ First data row label: {data[1][1] if len(data[1]) > 1 else 'N/A'}")
    
    print("\n✅ All checks passed! Your CSV can be loaded successfully.")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("Please make sure finalrun.csv is in the same directory as this script.")

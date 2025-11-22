import pandas as pd

# Read CSV efficiently (pandas handles large fields automatically)
df = pd.read_csv('finalrun.csv')

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}\n")

# Organize columns by category
categories = {
    'Basic Info': ['datetime', 'label', 'counts_all', 'variance_over_time'],
    
    'Female Pairs': [col for col in df.columns if col.startswith('female_pairs_')],
    
    'Male Pairs': [col for col in df.columns if col.startswith('male_pairs_')],
    
    'Asian Names': [col for col in df.columns if col.startswith('names_asian_')],
    
    'Hispanic Names': [col for col in df.columns if col.startswith('names_hispanic_')],
    
    'White Names': [col for col in df.columns if col.startswith('names_white_')],
    
    'Individual Distances - Group': [col for col in df.columns if 'indiv_distances_group' in col],
    
    'Individual Distances - Neutral': [col for col in df.columns if 'indiv_distances_neutral' in col],
    
    'Occupations 1950': [col for col in df.columns if col.startswith('occupations1950_') and 'professional' not in col],
    
    'Occupations 1950 Professional': [col for col in df.columns if 'occupations1950_professional' in col],
    
    'Personality Traits': [col for col in df.columns if col.startswith('personalitytraits_')]
}

# Display each category
for category_name, columns in categories.items():
    if columns:
        print(f"\n{'='*80}")
        print(f"  {category_name.upper()}")
        print(f"{'='*80}")
        
        # Get subset of dataframe with these columns
        subset = df[columns]
        
        # Display with pandas to_string for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 30)
        
        print(subset.to_string(index=True))
        print()

# Optional: Save each category to separate CSV files
save_separate = input("\nDo you want to save each category to separate CSV files? (y/n): ")
if save_separate.lower() == 'y':
    for category_name, columns in categories.items():
        if columns:
            filename = f"{category_name.replace(' ', '_').lower()}.csv"
            df[columns].to_csv(filename, index=False)
            print(f"Saved {filename}")

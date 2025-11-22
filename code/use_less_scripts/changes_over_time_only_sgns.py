#!/usr/bin/env python3
"""
Script to re-run analysis only for SGNS embeddings and update finalrun.csv

This script:
1. Reads the existing finalrun.csv
2. Runs the analysis only for SGNS embeddings
3. Replaces only the SGNS rows in finalrun.csv
4. Keeps all other rows unchanged

REQUIREMENTS:
- changes_over_time.py (the main analysis script)
- run_params.csv (parameter configuration file)
- vectors/normalized_clean/vectors_sgns*.txt (SGNS embedding files)
- vectors/normalized_clean/vocab/vocab*.txt (vocabulary frequency files)
- data/*.txt (word list files: male_pairs.txt, female_pairs.txt, etc.)
"""

import csv
import sys
import os
import datetime

# Import functions from the original script
try:
    from changes_over_time import (
        load_vectors_over_time,
        load_vocab,
        get_counts_dictionary,
        get_vector_variance,
        single_set_distances_to_single_set,
    )
except ImportError:
    print("ERROR: changes_over_time.py not found in the current directory")
    print("Please ensure changes_over_time.py is in the same directory as this script")
    sys.exit(1)


def run_sgns_analysis(filenames_sgns, label, neutral_lists, group_lists, 
                      do_individual_neutral_words=False, do_individual_group_words=False):
    """
    Run the analysis for SGNS embeddings only.
    Returns a dictionary with the analysis results.
    """
    print(f"Loading SGNS embeddings...")
    print(f"  Files: {filenames_sgns}")
    
    # Load vocab files
    vocabs = [fi.replace('vectors/normalized_clean/vectors', 
                        'vectors/normalized_clean/vocab/vocab') 
              for fi in filenames_sgns]
    vocabd = [load_vocab(fi) for fi in vocabs]
    
    # Load vectors
    vectors_over_time = load_vectors_over_time(filenames_sgns)
    print(f'  Vocab size: {[len(v.keys()) for v in vectors_over_time]}')
    
    # Initialize results dictionary
    d = {}
    d['counts_all'] = {}
    d['variance_over_time'] = {}
    
    # Process group lists
    for grouplist in group_lists:
        with open(f'data/{grouplist}.txt', 'r') as f2:
            groupwords = [x.strip() for x in list(f2)]
            d['counts_all'][grouplist] = get_counts_dictionary(vocabd, groupwords)
            d['variance_over_time'][grouplist] = get_vector_variance(
                vectors_over_time, groupwords, vocabd=vocabd
            )
    
    # Process neutral lists
    for neut in neutral_lists:
        with open(f'data/{neut}.txt', 'r') as f:
            neutwords = [x.strip() for x in list(f)]
            
            d['counts_all'][neut] = get_counts_dictionary(vocabd, neutwords)
            d['variance_over_time'][neut] = get_vector_variance(
                vectors_over_time, neutwords, vocabd=vocabd
            )
            
            dloc_neutral = {}
            
            for grouplist in group_lists:
                with open(f'data/{grouplist}.txt', 'r') as f2:
                    print(f'  Processing {neut} vs {grouplist}')
                    groupwords = [x.strip() for x in list(f2)]
                    distances = single_set_distances_to_single_set(
                        vectors_over_time, neutwords, groupwords, vocabd
                    )
                    
                    d[f'{neut}_{grouplist}'] = distances
                    
                    if do_individual_neutral_words:
                        for word in neutwords:
                            dloc_neutral[word] = dloc_neutral.get(word, {})
                            dloc_neutral[word][grouplist] = single_set_distances_to_single_set(
                                vectors_over_time, [word], groupwords, vocabd
                            )
                    
                    if do_individual_group_words:
                        d_group_so_far = d.get(f'indiv_distances_group_{grouplist}', {})
                        for word in groupwords:
                            d_group_so_far[word] = d_group_so_far.get(word, {})
                            d_group_so_far[word][neut] = single_set_distances_to_single_set(
                                vectors_over_time, neutwords, [word], vocabd
                            )
                        d[f'indiv_distances_group_{grouplist}'] = d_group_so_far
            
            d[f'indiv_distances_neutral_{neut}'] = dloc_neutral
    
    # Add metadata
    d['label'] = label
    d['datetime'] = datetime.datetime.now()
    
    return d


def update_finalrun_csv(original_csv_path, output_csv_path):
    """
    Read finalrun.csv, run SGNS analysis, and update only SGNS rows.
    """
    # Read all lines from the original CSV
    print(f"\nReading original CSV: {original_csv_path}")
    with open(original_csv_path, 'r') as f:
        all_lines = f.readlines()
    
    print(f"  Total lines: {len(all_lines)}")
    
    # Find SGNS rows (line 7 is header, line 8 is data - 0-indexed: 6 and 7)
    sgns_header_idx = None
    sgns_data_idx = None
    
    for idx, line in enumerate(all_lines):
        if idx % 2 == 1 and ',sgns,' in line:  # Data rows are odd-indexed
            sgns_data_idx = idx
            sgns_header_idx = idx - 1
            break
    
    if sgns_data_idx is None:
        print("ERROR: Could not find SGNS row in finalrun.csv")
        return False
    
    print(f"  Found SGNS at line {sgns_data_idx + 1} (header at line {sgns_header_idx + 1})")
    
    # Parse the SGNS header to get column names
    sgns_header = all_lines[sgns_header_idx].strip()
    
    # Read parameters from run_params.csv
    param_filename = 'run_params.csv'
    
    if not os.path.exists(param_filename):
        print(f"\nERROR: {param_filename} not found")
        print("Please ensure run_params.csv is in the current directory")
        return False
    
    print(f"\nReading parameters from {param_filename}")
    with open(param_filename, 'r') as f:
        reader = csv.DictReader(f)
        next(reader, None)  # skip the first line
        for row in reader:
            if row['label'] == 'sgns':
                label = row['label']
                neutral_lists = eval(row['neutral_lists'])
                group_lists = eval(row['group_lists'])
                do_individual_neutral_words = (row['do_individual_neutral_words'] == "TRUE")
                do_individual_group_words = (row.get('do_individual_group_words', '') == "TRUE")
                break
        else:
            print("ERROR: SGNS parameters not found in run_params.csv")
            return False
    
    print(f"  neutral_lists: {neutral_lists}")
    print(f"  group_lists: {group_lists}")
    
    # Define SGNS files
    folder = 'vectors/normalized_clean/'
    filenames_sgns = [folder + f'vectors_sgns{x}.txt' for x in range(1910, 2000, 10)]
    
    # Run SGNS analysis
    print(f"\nRunning SGNS analysis...")
    try:
        results = run_sgns_analysis(
            filenames_sgns,
            label=label,
            neutral_lists=neutral_lists,
            group_lists=group_lists,
            do_individual_neutral_words=do_individual_neutral_words,
            do_individual_group_words=do_individual_group_words
        )
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  Analysis complete!")
    
    # Create a new CSV line with the results using csv.DictWriter to match original format
    from io import StringIO
    
    # Parse header to get column order
    header_row = csv.reader([sgns_header]).__next__()
    
    # Replace the SGNS data row using csv.DictWriter to get proper quoting
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=header_row)
    writer.writerow(results)
    new_data_row = output.getvalue()
    
    # Replace the SGNS data row
    all_lines[sgns_data_idx] = new_data_row
    
    # Write to output file
    print(f"\nWriting updated CSV to: {output_csv_path}")
    with open(output_csv_path, 'w') as f:
        f.writelines(all_lines)
    
    print("SUCCESS: SGNS data updated in finalrun.csv")
    return True


def main():
    """Main function"""
    # File paths - look in run_results folder where original script writes
    original_csv = 'run_results/finalrun.csv'
    output_csv = 'run_results/finalrun_updated.csv'
    
    # Check if run_results directory exists
    if not os.path.exists('run_results'):
        print("ERROR: run_results/ directory not found")
        print("\nThe original changes_over_time.py writes to run_results/ folder.")
        print("Please ensure you're in the correct working directory.")
        return 1
    
    # Check if input file exists
    if not os.path.exists(original_csv):
        print(f"ERROR: {original_csv} not found")
        print("\nPlease ensure the following are in place:")
        print("  - run_results/finalrun.csv (your original results)")
        print("  - changes_over_time.py")
        print("  - run_params.csv")
        print("  - vectors/normalized_clean/vectors_sgns*.txt")
        print("  - vectors/normalized_clean/vocab/vocab*.txt")
        print("  - data/*.txt")
        return 1
    
    # Run the update
    success = update_finalrun_csv(original_csv, output_csv)
    
    if success:
        print(f"\n{'='*60}")
        print("COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Original file: {original_csv}")
        print(f"Updated file:  {output_csv}")
        print(f"\nOnly the SGNS rows have been updated.")
        print(f"All other rows remain unchanged.")
        print(f"\nTo replace the original:")
        print(f"  cp {original_csv} run_results/finalrun_backup.csv")
        print(f"  mv {output_csv} {original_csv}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("FAILED - See errors above")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
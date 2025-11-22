import numpy as np
import os
import re
from collections import defaultdict
import glob
import warnings
warnings.filterwarnings('ignore')

def validate_vector(vec, word):
    """
    Validate that a vector is meaningful.
    Returns (is_valid, issues) where issues is a list of problems found.
    """
    issues = []
    
    # Check for NaN values
    if np.isnan(vec).any():
        issues.append("contains NaN")
    
    # Check for infinite values
    if np.isinf(vec).any():
        issues.append("contains infinity")
    
    # Check if all zeros
    if np.all(vec == 0):
        issues.append("all zeros")
    
    # Check if norm is too small (should be ~1.0 for normalized vectors)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        issues.append(f"norm too small ({norm:.2e})")
    
    # Check if norm is far from 1.0 (assuming vectors should be normalized)
    if abs(norm - 1.0) > 0.1:
        issues.append(f"norm not close to 1.0 ({norm:.4f})")
    
    # Check for extreme values
    if np.max(np.abs(vec)) > 10:
        issues.append(f"extreme values (max: {np.max(np.abs(vec)):.2f})")
    
    is_valid = len(issues) == 0
    return is_valid, issues

def load_vectors_with_validation(filename):
    """Load word vectors and validate their quality."""
    vectors = {}
    validation_stats = {
        'total_loaded': 0,
        'valid': 0,
        'invalid': 0,
        'issues': defaultdict(int)
    }
    invalid_words = []
    
    print(f"Loading and validating vectors from {os.path.basename(filename)}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.rstrip().split()
                if not parts or len(parts) < 2:
                    continue
                
                word = parts[0]
                try:
                    vec = np.array([float(x) for x in parts[1:]])
                    
                    # Validate the vector
                    is_valid, issues = validate_vector(vec, word)
                    
                    validation_stats['total_loaded'] += 1
                    
                    if is_valid:
                        vectors[word] = vec
                        validation_stats['valid'] += 1
                    else:
                        validation_stats['invalid'] += 1
                        invalid_words.append((word, issues))
                        # Track issue types
                        for issue in issues:
                            validation_stats['issues'][issue] += 1
                    
                except (ValueError, OverflowError) as e:
                    validation_stats['invalid'] += 1
                    validation_stats['issues']['parse_error'] += 1
                    invalid_words.append((word, [f"parse error: {str(e)}"]))
                    continue
                
                # Progress indicator
                if (i + 1) % 100000 == 0:
                    print(f"  Processed {i + 1} lines... ({validation_stats['valid']} valid)")
        
        # Print validation summary
        print(f"  ✓ Valid vectors: {validation_stats['valid']}")
        if validation_stats['invalid'] > 0:
            print(f"  ✗ Invalid vectors: {validation_stats['invalid']}")
            print(f"    Issues found:")
            for issue, count in sorted(validation_stats['issues'].items(), key=lambda x: x[1], reverse=True):
                print(f"      - {issue}: {count}")
            
            # Show some examples of invalid words
            if len(invalid_words) <= 10:
                print(f"    Invalid words: {', '.join([w for w, _ in invalid_words])}")
            else:
                print(f"    Sample invalid words: {', '.join([w for w, _ in invalid_words[:5]])}... (and {len(invalid_words)-5} more)")
        print()
        
    except Exception as e:
        print(f"  ERROR loading file: {e}\n")
        
    return vectors, validation_stats

def load_words_from_file(filepath):
    """Load words from a text file (one word per line)."""
    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            # Clean word to match normalization (only lowercase letters)
            word_clean = re.sub('[^a-z]+', '', word)
            if len(word_clean) >= 2:
                words.append(word_clean)
    return words

def discover_vector_files(vectors_dir):
    """Automatically discover all vector files in the directory."""
    
    vector_files = {
        'static': {},
        'temporal': {'sgns': {}, 'svd': {}}
    }
    
    # Static vectors
    static_patterns = {
        'GloVe': 'vectorscommoncrawlglove.txt',
        'GoogleNews': 'vectorsGoogleNews_exactclean.txt',
        'Wikipedia': 'vectorswikipedia.txt'
    }
    
    for name, pattern in static_patterns.items():
        filepath = os.path.join(vectors_dir, pattern)
        if os.path.exists(filepath):
            vector_files['static'][name] = filepath
    
    # Temporal vectors - SGNS
    sgns_files = glob.glob(os.path.join(vectors_dir, 'vectors_sgns*.txt'))
    for filepath in sorted(sgns_files):
        basename = os.path.basename(filepath)
        # Extract year from filename
        match = re.search(r'sgns(\d{4})', basename)
        if match:
            year = int(match.group(1))
            vector_files['temporal']['sgns'][year] = filepath
    
    # Temporal vectors - SVD
    svd_files = glob.glob(os.path.join(vectors_dir, 'vectors_svd*.txt'))
    for filepath in sorted(svd_files):
        basename = os.path.basename(filepath)
        # Extract year from filename
        match = re.search(r'svd(\d{4})', basename)
        if match:
            year = int(match.group(1))
            vector_files['temporal']['svd'][year] = filepath
    
    return vector_files

def check_word_coverage_with_validation(data_folder, vector_files_dict, load_mode='all'):
    """
    Check which words from data files appear in which vector files,
    with validation of vector quality.
    
    load_mode options:
    - 'all': Load all vectors
    - 'static': Load only static vectors
    - 'sgns': Load only SGNS temporal vectors
    - 'svd': Load only SVD temporal vectors
    - 'sample_temporal': Load static + sample of temporal
    """
    
    all_vectors = {}
    all_validation_stats = {}
    
    print(f"Load mode: {load_mode}\n")
    print("="*80)
    
    # Load static vectors
    if load_mode in ['all', 'static', 'sample_temporal']:
        print("\nLoading STATIC vectors:")
        print("-"*80)
        for name, filepath in vector_files_dict['static'].items():
            vecs, stats = load_vectors_with_validation(filepath)
            all_vectors[f'static_{name}'] = vecs
            all_validation_stats[f'static_{name}'] = stats
    
    # Load SGNS temporal vectors
    if load_mode in ['all', 'sgns']:
        print("\nLoading SGNS TEMPORAL vectors:")
        print("-"*80)
        for year, filepath in sorted(vector_files_dict['temporal']['sgns'].items()):
            vecs, stats = load_vectors_with_validation(filepath)
            all_vectors[f'sgns_{year}'] = vecs
            all_validation_stats[f'sgns_{year}'] = stats
    elif load_mode == 'sample_temporal':
        print("\nLoading SGNS TEMPORAL vectors (sample):")
        print("-"*80)
        sgns_years = sorted(vector_files_dict['temporal']['sgns'].keys())
        # Sample every 3rd year, or ~30 year intervals
        if len(sgns_years) > 0:
            step = max(1, len(sgns_years) // 4)
            sample_years = sgns_years[::step]
            for year in sample_years:
                filepath = vector_files_dict['temporal']['sgns'][year]
                vecs, stats = load_vectors_with_validation(filepath)
                all_vectors[f'sgns_{year}'] = vecs
                all_validation_stats[f'sgns_{year}'] = stats
    
    # Load SVD temporal vectors
    if load_mode in ['all', 'svd']:
        print("\nLoading SVD TEMPORAL vectors:")
        print("-"*80)
        for year, filepath in sorted(vector_files_dict['temporal']['svd'].items()):
            vecs, stats = load_vectors_with_validation(filepath)
            all_vectors[f'svd_{year}'] = vecs
            all_validation_stats[f'svd_{year}'] = stats
    elif load_mode == 'sample_temporal':
        print("\nLoading SVD TEMPORAL vectors (sample):")
        print("-"*80)
        svd_years = sorted(vector_files_dict['temporal']['svd'].keys())
        if len(svd_years) > 0:
            step = max(1, len(svd_years) // 4)
            sample_years = svd_years[::step]
            for year in sample_years:
                filepath = vector_files_dict['temporal']['svd'][year]
                vecs, stats = load_vectors_with_validation(filepath)
                all_vectors[f'svd_{year}'] = vecs
                all_validation_stats[f'svd_{year}'] = stats
    
    # Get all .txt files from data folder
    txt_files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    txt_files.sort()
    
    print("\n" + "="*80)
    print(f"Found {len(txt_files)} .txt files in {data_folder}")
    print(f"Checking coverage across {len(all_vectors)} vector files")
    print("="*80)
    
    # Results storage
    results = {}
    
    # Process each file
    for txt_file in txt_files:
        filepath = os.path.join(data_folder, txt_file)
        words = load_words_from_file(filepath)
        
        if not words:
            continue
        
        print(f"\n{txt_file}: {len(words)} words")
        print("-" * 80)
        
        # Check coverage in each vector file
        coverage = {}
        for vec_name, vectors in all_vectors.items():
            found_words = [w for w in words if w in vectors]
            coverage[vec_name] = {
                'found': len(found_words),
                'total': len(words),
                'percentage': 100 * len(found_words) / len(words) if words else 0,
                'found_words': found_words,
                'missing_words': [w for w in words if w not in vectors]
            }
        
        # Print summary (grouped by type)
        static_vecs = {k: v for k, v in coverage.items() if k.startswith('static_')}
        sgns_vecs = {k: v for k, v in coverage.items() if k.startswith('sgns_')}
        svd_vecs = {k: v for k, v in coverage.items() if k.startswith('svd_')}
        
        if static_vecs:
            print("  Static vectors:")
            for vec_name, stats in sorted(static_vecs.items()):
                validation = all_validation_stats[vec_name]
                quality_pct = 100 * validation['valid'] / validation['total_loaded'] if validation['total_loaded'] > 0 else 0
                print(f"    {vec_name:25s}: {stats['found']:4d}/{stats['total']:4d} ({stats['percentage']:5.1f}%) [Quality: {quality_pct:.1f}%]")
        
        if sgns_vecs:
            print("  SGNS temporal:")
            for vec_name, stats in sorted(sgns_vecs.items()):
                validation = all_validation_stats[vec_name]
                quality_pct = 100 * validation['valid'] / validation['total_loaded'] if validation['total_loaded'] > 0 else 0
                print(f"    {vec_name:25s}: {stats['found']:4d}/{stats['total']:4d} ({stats['percentage']:5.1f}%) [Quality: {quality_pct:.1f}%]")
        
        if svd_vecs:
            print("  SVD temporal:")
            for vec_name, stats in sorted(svd_vecs.items()):
                validation = all_validation_stats[vec_name]
                quality_pct = 100 * validation['valid'] / validation['total_loaded'] if validation['total_loaded'] > 0 else 0
                print(f"    {vec_name:25s}: {stats['found']:4d}/{stats['total']:4d} ({stats['percentage']:5.1f}%) [Quality: {quality_pct:.1f}%]")
        
        results[txt_file] = {
            'words': words,
            'coverage': coverage
        }
    
    return results, all_vectors, all_validation_stats

def save_detailed_report_with_validation(results, all_vectors, all_validation_stats, output_file):
    """Save a detailed report with vector information and quality metrics."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("WORD VECTOR COVERAGE REPORT WITH QUALITY VALIDATION\n")
        f.write("="*80 + "\n\n")
        
        # Overall validation summary
        f.write("OVERALL VECTOR QUALITY:\n")
        f.write("-"*80 + "\n")
        for vec_name in sorted(all_validation_stats.keys()):
            stats = all_validation_stats[vec_name]
            quality_pct = 100 * stats['valid'] / stats['total_loaded'] if stats['total_loaded'] > 0 else 0
            f.write(f"\n{vec_name}:\n")
            f.write(f"  Total vectors loaded: {stats['total_loaded']}\n")
            f.write(f"  Valid vectors: {stats['valid']} ({quality_pct:.2f}%)\n")
            f.write(f"  Invalid vectors: {stats['invalid']}\n")
            if stats['invalid'] > 0:
                f.write(f"  Issues found:\n")
                for issue, count in sorted(stats['issues'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"    - {issue}: {count}\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("WORD COVERAGE BY DATA FILE\n")
        f.write("="*80 + "\n")
        
        for txt_file, data in sorted(results.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"FILE: {txt_file}\n")
            f.write(f"{'='*80}\n\n")
            
            words = data['words']
            coverage = data['coverage']
            
            # Group by type
            static_vecs = {k: v for k, v in coverage.items() if k.startswith('static_')}
            sgns_vecs = {k: v for k, v in coverage.items() if k.startswith('sgns_')}
            svd_vecs = {k: v for k, v in coverage.items() if k.startswith('svd_')}
            
            # Summary table
            f.write("COVERAGE SUMMARY:\n")
            f.write("-"*80 + "\n")
            
            if static_vecs:
                f.write("\nStatic Vectors:\n")
                for vec_name, stats in sorted(static_vecs.items()):
                    validation = all_validation_stats[vec_name]
                    quality_pct = 100 * validation['valid'] / validation['total_loaded'] if validation['total_loaded'] > 0 else 0
                    f.write(f"  {vec_name:30s}: {stats['found']:4d}/{stats['total']:4d} ({stats['percentage']:5.1f}%) [Quality: {quality_pct:.1f}%]\n")
            
            if sgns_vecs:
                f.write("\nSGNS Temporal Vectors:\n")
                for vec_name, stats in sorted(sgns_vecs.items()):
                    validation = all_validation_stats[vec_name]
                    quality_pct = 100 * validation['valid'] / validation['total_loaded'] if validation['total_loaded'] > 0 else 0
                    f.write(f"  {vec_name:30s}: {stats['found']:4d}/{stats['total']:4d} ({stats['percentage']:5.1f}%) [Quality: {quality_pct:.1f}%]\n")
            
            if svd_vecs:
                f.write("\nSVD Temporal Vectors:\n")
                for vec_name, stats in sorted(svd_vecs.items()):
                    validation = all_validation_stats[vec_name]
                    quality_pct = 100 * validation['valid'] / validation['total_loaded'] if validation['total_loaded'] > 0 else 0
                    f.write(f"  {vec_name:30s}: {stats['found']:4d}/{stats['total']:4d} ({stats['percentage']:5.1f}%) [Quality: {quality_pct:.1f}%]\n")
            
            # Missing words section
            f.write("\n" + "-"*80 + "\n")
            f.write("MISSING WORDS (Static Vectors Only):\n")
            f.write("-"*80 + "\n")
            
            for vec_name, stats in sorted(static_vecs.items()):
                if stats['missing_words']:
                    f.write(f"\n{vec_name}:\n")
                    for word in sorted(stats['missing_words'])[:20]:  # Limit to 20
                        f.write(f"  - {word}\n")
                    if len(stats['missing_words']) > 20:
                        f.write(f"  ... and {len(stats['missing_words']) - 20} more\n")
            
            # Sample vectors for found words
            f.write("\n" + "-"*80 + "\n")
            f.write("SAMPLE VECTOR VALIDATION (first 3 words from static vectors):\n")
            f.write("-"*80 + "\n")
            
            static_vector_objs = {k.replace('static_', ''): v for k, v in all_vectors.items() if k.startswith('static_')}
            
            for i, word in enumerate(words[:3]):
                f.write(f"\nWord: '{word}'\n")
                for vec_name, vectors in sorted(static_vector_objs.items()):
                    if word in vectors:
                        vec = vectors[word]
                        is_valid, issues = validate_vector(vec, word)
                        norm = np.linalg.norm(vec)
                        
                        status = "✓ VALID" if is_valid else "✗ INVALID"
                        f.write(f"  {vec_name:30s}: {status}\n")
                        f.write(f"    Dimension: {len(vec)}, Norm: {norm:.6f}\n")
                        f.write(f"    Value range: [{vec.min():.6f}, {vec.max():.6f}]\n")
                        f.write(f"    Mean: {vec.mean():.6f}, Std: {vec.std():.6f}\n")
                        f.write(f"    First 10 components: {vec[:10]}\n")
                        if not is_valid:
                            f.write(f"    Issues: {', '.join(issues)}\n")
                    else:
                        f.write(f"  {vec_name:30s}: NOT FOUND\n")
            
            f.write("\n")

def save_quality_summary_csv(all_validation_stats, output_file):
    """Save a CSV summary of vector quality statistics."""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['vector_file', 'total_loaded', 'valid', 'invalid', 'quality_percentage']
        # Add common issue types
        all_issues = set()
        for stats in all_validation_stats.values():
            all_issues.update(stats['issues'].keys())
        header.extend(sorted(all_issues))
        
        writer.writerow(header)
        
        # Data rows
        for vec_name in sorted(all_validation_stats.keys()):
            stats = all_validation_stats[vec_name]
            quality_pct = 100 * stats['valid'] / stats['total_loaded'] if stats['total_loaded'] > 0 else 0
            
            row = [
                vec_name,
                stats['total_loaded'],
                stats['valid'],
                stats['invalid'],
                f"{quality_pct:.2f}"
            ]
            
            # Add issue counts
            for issue in sorted(all_issues):
                row.append(stats['issues'].get(issue, 0))
            
            writer.writerow(row)

if __name__ == "__main__":
    # Configuration
    data_folder = "../../data/word_lists"  # Your data folder path
    vectors_dir = "../../data/vectors/normalized_clean"  # Your vectors directory
    
    # Discover all vector files
    print("Discovering vector files...")
    vector_files_dict = discover_vector_files(vectors_dir)
    
    print(f"\nFound vectors:")
    print(f"  Static: {len(vector_files_dict['static'])} files")
    print(f"  SGNS temporal: {len(vector_files_dict['temporal']['sgns'])} files (years: {sorted(vector_files_dict['temporal']['sgns'].keys())})")
    print(f"  SVD temporal: {len(vector_files_dict['temporal']['svd'])} files (years: {sorted(vector_files_dict['temporal']['svd'].keys())})")
    print()
    
    # Choose load mode
    # Options: 'all', 'static', 'sgns', 'svd', 'sample_temporal'
    load_mode = 'all'  # Change this as needed
    
    print("="*80)
    print("TEMPORAL WORD VECTOR COVERAGE CHECKER WITH QUALITY VALIDATION")
    print("="*80 + "\n")
    
    # Run the check with validation
    results, all_vectors, all_validation_stats = check_word_coverage_with_validation(
        data_folder, vector_files_dict, load_mode=load_mode
    )
    
    # Save detailed report
    print("\n" + "="*80)
    print("Saving detailed report with validation...")
    save_detailed_report_with_validation(
        results, all_vectors, all_validation_stats, 
        'word_coverage_report_validated.txt'
    )
    print("Saved: word_coverage_report_validated.txt")
    
    # Save quality summary
    print("Saving vector quality summary...")
    save_quality_summary_csv(all_validation_stats, 'vector_quality_summary.csv')
    print("Saved: vector_quality_summary.csv")
    
    # Also save regular coverage summary (for compatibility)
    print("Saving coverage summary...")
    import csv
    with open('word_coverage_summary.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['filename', 'total_words']
        if results:
            first_file = list(results.values())[0]
            vec_names = sorted(first_file['coverage'].keys())
            for vec_name in vec_names:
                header.extend([f'{vec_name}_found', f'{vec_name}_pct'])
        writer.writerow(header)
        
        # Data rows
        for txt_file, data in sorted(results.items()):
            row = [txt_file, len(data['words'])]
            for vec_name in vec_names:
                stats = data['coverage'][vec_name]
                row.extend([stats['found'], f"{stats['percentage']:.1f}"])
            writer.writerow(row)
    print("Saved: word_coverage_summary.csv")
    
    print("\n" + "="*80)
    print("SUMMARY OF VECTOR QUALITY:")
    print("="*80)
    for vec_name in sorted(all_validation_stats.keys()):
        stats = all_validation_stats[vec_name]
        quality_pct = 100 * stats['valid'] / stats['total_loaded'] if stats['total_loaded'] > 0 else 0
        status = "✓ EXCELLENT" if quality_pct >= 99 else "⚠ CHECK" if quality_pct >= 95 else "✗ ISSUES"
        print(f"{vec_name:30s}: {stats['valid']:7d} valid / {stats['total_loaded']:7d} total ({quality_pct:5.2f}%) {status}")
        if stats['invalid'] > 0 and stats['invalid'] <= 10:
            # Show issues for files with few problems
            issues_str = ', '.join([f"{issue}: {count}" for issue, count in sorted(stats['issues'].items())])
            print(f"  Issues: {issues_str}")
    
    print("\nDone!")
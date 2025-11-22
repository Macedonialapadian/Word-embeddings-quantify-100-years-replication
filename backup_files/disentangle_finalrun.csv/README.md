# Finalrun.csv Display Scripts

Your finalrun.csv contains **57 columns** and **12 rows** (1 header + 11 data rows) with bias metrics for different demographic groups.

## 🚀 Getting Started

**First time?** Run this to verify everything works:
```bash
python test_csv_loading.py
```

**Want a quick overview?** Run this to see the structure:
```bash
python explore_structure.py
```

**Ready to explore the data?** Use any of the three main display scripts below.

## ⚠️ Important Note
Your CSV has very large fields that exceed Python's default CSV field size limit. All scripts have been configured to handle this automatically using:
```python
csv.field_size_limit(sys.maxsize)
```

## File Structure

Your CSV has these categories of data:
- **Basic Info**: datetime, label, counts_all, variance_over_time
- **Female/Male Pairs**: Gender-based pair comparisons
- **Names (Asian/Hispanic/White)**: Ethnic name-based metrics
- **Individual Distances**: Group and neutral distance measurements
- **Occupations**: 1950 occupation metrics (general and professional)
- **Personality Traits**: Original personality trait metrics

## Three Ways to Display Your Data

### 1. **disentangle_finalrun_complete.ipynb** (Recommended)
Complete Jupyter notebook that extends your original work. It includes:
- Your original code
- Proper CSV parsing with field size limit fix
- Organized display by category
- Both manual and pandas-based approaches

**Usage**: Open in Jupyter Notebook and run cells sequentially

### 2. **display_finalrun.py** (Pandas-based)
Organized, categorical display using pandas for clean formatting.

**Usage**:
```bash
python display_finalrun.py
```

Features:
- Automatically organizes data into 11 categories
- Clean pandas formatting
- Option to save each category to separate CSV files

### 3. **display_simple.py** (Interactive)
Simple, interactive script following your original approach.

**Usage**:
```bash
python display_simple.py
```

Features:
- Shows basic info for each row
- Asks if you want to see all columns for each row
- Can quit anytime with 'q'

## 🔍 Helper Scripts

### **explore_structure.py** (Quick Overview)
Get a quick overview of your data structure without overwhelming output.

**Usage**:
```bash
python explore_structure.py
```

Shows column organization, row labels, and sample data.

### **test_csv_loading.py** (Verify Setup)
Quick test to verify the CSV loads correctly with the field size fix.

**Usage**:
```bash
python test_csv_loading.py
```

## Quick Stats

- **Total Columns**: 57
- **Total Rows**: 11 (data rows, excluding header)
- **File Size**: ~10MB (each row has lots of data)

## Tips for Memory Efficiency

Since the CSV is large, the scripts are designed to:
1. Load data once
2. Display in organized chunks
3. Not duplicate data in memory
4. Use pandas efficiently when possible

All scripts work with the finalrun.csv file in the same directory.

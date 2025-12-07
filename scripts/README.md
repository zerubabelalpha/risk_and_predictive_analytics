# Scripts Directory

This directory contains the core data processing and analysis modules for the Risk and Predictive Analytics project.

## Modules

### `clean_data.py`

**DataCleaner Class** - Comprehensive data cleaning and preprocessing.

**Key Features:**
- Load pipe-delimited data files
- Convert columns to proper numeric types
- Normalize dates to yyyy-mm-dd format
- Calculate loss ratios
- Save cleaned data in same directory as raw data with `_cleaned` suffix

**Usage:**
```python
from scripts.clean_data import DataCleaner

cleaner = DataCleaner()
df = cleaner.process_file("data/raw.txt")  # Saves to data/raw_cleaned.csv
```

### `eda_inline.py`

**EDAAnalyzer Class** - File-based exploratory data analysis with saved outputs.

**Key Features:**
- Data structure summarization
- Missing value assessment
- Descriptive statistics with skewness
- Distribution plots (histograms, boxplots)
- Outlier detection (IQR, Z-score)
- Correlation analysis
- Scatter plots
- Group-level KPIs
- Temporal trend analysis


**Usage:**
```python
from scripts.eda import EDAAnalyzer, analyze_data

# Quick start - run full analysis (saves outputs)
analyzer = analyze_data(df, output_dir="outputs", run_full=True)

# Or run individual methods
analyzer = EDAAnalyzer(df, output_dir="outputs")
analyzer.correlation_analysis()
analyzer.temporal_analysis()
```

### `eda_inline.py` 

**Inline EDA Functions** - Notebook-friendly analysis with inline display (no file outputs).

**Key Features:**
- All visualizations display inline using `plt.show()`
- All tables display using `print()` and `display()`
- **No files saved to disk** - perfect for Jupyter notebooks
- Same analysis capabilities as `eda.py`
- Individual functions for each analysis type
- Master `run_full_inline_analysis()` function

**Available Functions:**
- `show_data_summary()` - Display data structure
- `show_missing_values()` - Missing value analysis
- `show_descriptive_stats()` - Statistical summaries
- `show_distributions()` - Histograms and boxplots
- `show_outliers()` - Outlier detection
- `show_correlation()` - Correlation matrix and heatmap
- `show_scatter()` - Scatter plots
- `show_group_analysis()` - Group-level KPIs
- `show_temporal_trends()` - Time series analysis
- `run_full_inline_analysis()` - Complete EDA pipeline

**Usage:**
```python
from scripts.eda_inline import run_full_inline_analysis, show_correlation

# Quick start - run full analysis (displays inline)
run_full_inline_analysis(df)

# Or run individual functions
show_correlation(df)
show_temporal_trends(df)
```

## Quick Start

### Inline EDA 

```python
# Complete pipeline with inline display
from scripts.clean_data import DataCleaner
from scripts.eda_inline import run_full_inline_analysis

cleaner = DataCleaner()
df = cleaner.process_file("data/MachineLearningRating_v3.txt")
run_full_inline_analysis(df)  # All results display inline
```


## Output Structure

### Cleaned Data
```
data/
└── *_cleaned.csv         # Cleaned data saved alongside raw data
```



- **Analysis Notebooks**: 
  - `notebooks/analysis.ipynb` 
  
- **Tests**: `tests/test_scripts.py` 

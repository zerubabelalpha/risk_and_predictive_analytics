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

### `preprocessing.py`

**DataPreprocessor Class** - Comprehensive preprocessing for machine learning models.

**Key Features:**
- Remove duplicate rows
- Handle missing values (configurable threshold, default 80%)
- Smart imputation (median for numeric, mode for categorical)
- Categorical encoding (label encoding and one-hot encoding)
- Multiple scaling methods (StandardScaler, MinMaxScaler, log transformation)
- Save/load fitted transformers for production use
- Detailed preprocessing reports

**Usage:**
```python
from scripts.preprocessing import DataPreprocessor

# Initialize with 80% missing value threshold
preprocessor = DataPreprocessor(missing_threshold=0.80)

# Full pipeline
df_processed = preprocessor.preprocess_pipeline(
    df,
    encoding_method='label',  # or 'onehot'
    scaling_method='standard'  # or 'minmax' or 'log'
)

# Save artifacts for production
preprocessor.save_artifacts('artifacts/', prefix='model_v1')

# Load artifacts for new data
new_preprocessor = DataPreprocessor()
new_preprocessor.load_artifacts('artifacts/model_v1_artifacts.pkl')
df_new_processed = new_preprocessor.preprocess_pipeline(df_new)
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

### `ab_hypothesis_testing.py`

**ABHypothesisTester Class** - framework for A/B testing on insurance data.

**Key Features:**
- Metrics calculation: Claim Frequency, Severity, Margin
- Statistical tests: Chi-Squared, T-Test, Mann-Whitney U
- Visualization of test results
- Built-in methods for specific hypotheses (Province, Zip Code, Gender, Margin)

**Usage:**
```python
from scripts.ab_hypothesis_testing import ABHypothesisTester

tester = ABHypothesisTester(df)
tester.test_risk_gender("Gender", "Female", "Male")
```

### `model.py`

**InsuranceModeler Class** - Machine learning workflow for Risk and Pricing.

**Key Features:**
- **Claim Severity Model**: Predicts `TotalClaims` (Regression: LR, RF, XGB)
- **Claim Probability Model**: Predicts `HasClaim` (Classification: LR, RF, XGB)
- **Premium Optimization**: Calculates risk-based premiums combining probability and severity
- **Interpretability**: SHAP integration for model explanation
- Evaluation metrics included (RMSE, R2, Accuracy, F1, etc.)

**Usage:**
```python
from scripts.model import InsuranceModeler

modeler = InsuranceModeler(df)

# Train Severity Model
modeler.train_severity_models(X_train, y_train)

# Calculate Premium
premium_df = modeler.calculate_risk_premium(X_new)
```

## Quick Start

### Complete ML Pipeline

```python
from scripts.clean_data import DataCleaner
from scripts.model import InsuranceModeler

# 1. Clean Data
cleaner = DataCleaner()
df = cleaner.process_file("data/MachineLearningRating_v3.txt")

# 2. Model Risk & Price
modeler = InsuranceModeler(df)
# ... preprocess & split ...
modeler.train_severity_models(X_train, y_train)
```

## Output Structure

### Cleaned Data
```
data/
├── *_cleaned.csv                    # Cleaned data saved alongside raw data
└── preprocessing_artifacts/         # Fitted transformers for production
    └── *_artifacts.pkl              # Encoders, scalers, imputers
```

- **Analysis Notebooks**: 
  - `notebooks/ab_testing_demo.ipynb`
  - `notebooks/model_training.ipynb`
 

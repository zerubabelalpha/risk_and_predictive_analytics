# Risk and Predictive Analytics

A comprehensive insurance data analysis toolkit featuring professional class-based implementations for data cleaning, exploratory data analysis (EDA), and predictive modeling.

## 🚀 Features

- **DataCleaner Class**: Automated data cleaning and preprocessing
  - Load pipe-delimited insurance data
  - Convert to proper numeric types with error handling
  - Normalize dates to yyyy-mm-dd format
  - Calculate loss ratios automatically
  - Save cleaned data in same directory as raw data with `_cleaned` suffix

- **EDA Module**: Approaches for exploratory data analysis
  
  - **Inline Functions** (`eda_inline.py`): Notebook-friendly inline display ⭐
    - All visualizations displayed inline with `plt.show()`
    - No files saved to disk
    - Perfect for Jupyter notebooks
  - Include: data summarization, missing values, statistics, distributions, outliers, correlation, scatter plots, group KPIs, temporal trends

## 📁 Project Structure

```
risk_and_predictive_analytics/
|
├──.dvc
|    ├──.gitignore                           # dvc
|    └──config
|
├──.github/workflows/ci-cd.yaml
├── data/
│   ├── MachineLearningRating_v3.txt         # Raw insurance data
│   └── MachineLearningRating_v3_cleaned.csv # Auto-generated cleaned data
├── scripts/
│   ├── clean_data.py                        # DataCleaner class
│   ├── eda_inline.py                        # Inline EDA functions
│   └── README.md                            # Scripts documentation
├── notebooks/
│   └── analysis.ipynb                       # analysis notebook
├── tests/
│   └── test_scripts.py                      # Test suite
|
├── .dcvignore
├── .gitignore
├── pyproject.toml                           # py configuration
|
├── requirements.txt                         # Python dependencies
└── README.md                                # project discription
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd risk_and_predictive_analytics
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Quick Start

###  Inline EDA (Notebook-Friendly) ⭐ Recommended

```python
from scripts.clean_data import DataCleaner
from scripts.eda_inline import run_full_inline_analysis

# Clean data (saves to data/filename_cleaned.csv)
cleaner = DataCleaner()
df = cleaner.process_file("data/MachineLearningRating_v3.txt")

# Run EDA with inline display 
run_full_inline_analysis(df)
```


###  Step-by-Step Inline EDA

```python
from scripts.clean_data import DataCleaner
from scripts.eda_inline import (
    show_data_summary,
    show_missing_values,
    show_descriptive_stats,
    show_distributions,
    show_correlation,
    show_temporal_trends
)

# 1. Clean the data
cleaner = DataCleaner()
df = cleaner.process_file(
    input_path="data/MachineLearningRating_v3.txt",
    save_to_cleaned=True  # Saves to data/filename_cleaned.csv
)

# 2. Run individual EDA steps (displays inline)
show_data_summary(df)
show_missing_values(df)
show_descriptive_stats(df)
show_distributions(df, ['TotalPremium', 'TotalClaims'])
show_correlation(df)
show_temporal_trends(df)
```

## 📊 Usage Examples

### Data Cleaning

```python
from scripts.clean_data import DataCleaner

# Initialize with custom configuration
cleaner = DataCleaner(
    numeric_columns=["TotalPremium", "TotalClaims", "SumInsured"],
    date_columns=["TransactionMonth", "VehicleIntroDate"],
    separator="|"
)

# Process file (auto-saves to data/raw_data_cleaned.csv)
df = cleaner.process_file("data/raw_data.txt")

# Or specify custom output path
df = cleaner.process_file(
    "data/raw_data.txt",
    output_path="custom/location.csv"
)

# Don't save, just return cleaned DataFrame
df = cleaner.process_file("data/raw_data.txt", save_to_cleaned=False)
```

### Exploratory Data Analysis

**Inline EDA :**
```python
from scripts.eda_inline import (
    show_missing_values,
    show_descriptive_stats,
    show_distributions,
    show_outliers,
    show_correlation,
    show_scatter,
    show_group_analysis,
    show_temporal_trends
)

# All results display inline in notebook
show_missing_values(df, top_n=20)
show_descriptive_stats(df)
show_distributions(df, ['TotalPremium', 'TotalClaims'])
show_outliers(df, 'TotalPremium', method='iqr')
show_correlation(df)
show_scatter(df, 'TotalPremium', 'TotalClaims')
show_group_analysis(df, group_cols=['Province', 'Gender'])
show_temporal_trends(df)
```



## 📓 Jupyter Notebooks

**Analysis:**
- `notebooks/analysis.ipynb` - Displays all results inline 
  - Perfect for interactive analysis
  - Clean notebook output


## 🧪 Testing

Run the test suite:
```bash
pytest tests/test_scripts.py -v
```


## 🔧 Configuration

### DataCleaner Configuration

```python
cleaner = DataCleaner(
    numeric_columns=[...],      # Columns to convert to numeric
    date_columns=[...],          # Columns to normalize as dates
    separator="|",               # Input file delimiter
    encoding="utf-8"             # File encoding
)
```

### EDAAnalyzer Configuration

```python
analyzer = EDAAnalyzer(
    df=cleaned_df,               # Cleaned DataFrame
    output_dir="outputs",        # Output directory
    numeric_cols=[...],          # Numeric columns to analyze
    group_cols=[...]             # Categorical columns for grouping
)
```

## 📋 Data Schema

The insurance dataset includes:

**Identifiers**: UnderwrittenCoverID, PolicyID

**Dates**: TransactionMonth, VehicleIntroDate (normalized to yyyy-mm-dd)

**Numeric**: TotalPremium, TotalClaims, SumInsured, CustomValueEstimate, CapitalOutstanding, CalculatedPremiumPerTerm, cubiccapacity, kilowatts, Cylinders, NumberOfDoors, RegistrationYear

**Categorical**: Province, VehicleType, Gender, MaritalStatus, Citizenship, LegalType, Bank, AccountType, etc.

**Calculated**: LossRatio (TotalClaims / TotalPremium)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


**Note**: This project uses DVC (Data Version Control) for managing large data files. The raw data file `MachineLearningRating_v3.txt` is tracked with DVC.
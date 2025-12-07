"""
Inline EDA Module for Jupyter Notebooks

This module provides functions for exploratory data analysis that display
results directly in notebooks without saving to files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from scipy import stats

# Set default plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def show_data_summary(df: pd.DataFrame) -> None:
    """
    Display data structure and basic information.
    
    Args:
        df: DataFrame to summarize
    """
    print("=" * 60)
    print("DATA STRUCTURE SUMMARY")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    print(f"\nFirst few rows:")
    display(df.head())
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def show_missing_values(df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """
    Display missing values analysis.
    
    Args:
        df: DataFrame to analyze
        top_n: Number of top columns with missing values to display
    
    Returns:
        DataFrame with missing value statistics
    """
    print("\n" + "=" * 60)
    print("MISSING VALUES ASSESSMENT")
    print("=" * 60)
    
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    
    missing_with_values = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_with_values) > 0:
        print(f"\nTop {top_n} columns with missing values:")
        display(missing_with_values.head(top_n))
    else:
        print("\n✓ No missing values found!")
    
    return missing_df


def show_descriptive_stats(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Display descriptive statistics for numeric columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to analyze. If None, uses all numeric columns.
    
    Returns:
        DataFrame with descriptive statistics
    """
    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    available_cols = [col for col in columns if col in df.columns]
    
    if not available_cols:
        print("\nNo numeric columns found!")
        return pd.DataFrame()
    
    # Basic statistics
    stats_df = df[available_cols].describe().T
    
    # Add skewness
    stats_df['skewness'] = df[available_cols].skew()
    
    print("\nNumeric Column Statistics:")
    display(stats_df)
    
    return stats_df


def show_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (12, 5)
) -> None:
    """
    Display histogram and boxplot visualizations for numeric columns.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to plot. If None, uses all numeric columns.
        figsize: Figure size for each plot
    """
    print("\n" + "=" * 60)
    print("DISTRIBUTION PLOTS")
    print("=" * 60)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    available_cols = [col for col in columns if col in df.columns]
    
    for col in available_cols:
        data = df[col].dropna()
        
        if len(data) == 0:
            print(f"\n⚠ Skipping {col} - no data available")
            continue
        
        print(f"\n{col}:")
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram with KDE
        sns.histplot(data, kde=True, bins=50, ax=axes[0])
        axes[0].set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")
        
        # Use symlog scale for highly skewed data
        if data.max() / data.median() > 100:
            axes[0].set_xscale('symlog')
        
        # Boxplot
        sns.boxplot(x=data, ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}", fontsize=12, fontweight='bold')
        axes[1].set_xlabel(col)
        
        plt.tight_layout()
        plt.show()


def show_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0
) -> pd.Series:
    """
    Detect and display outliers in a column.
    
    Args:
        df: DataFrame to analyze
        column: Column name to analyze
        method: "iqr" or "zscore"
        k: IQR multiplier (default: 1.5)
        z_threshold: Z-score threshold (default: 3.0)
    
    Returns:
        Series containing outlier values
    """
    print(f"\n{column} - Outlier Detection ({method.upper()} method):")
    
    data = df[column].dropna()
    
    if method == "iqr":
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        print(f"  IQR Outliers: {len(outliers):,} ({len(outliers)/len(data)*100:.2f}%)")
        
    elif method == "zscore":
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > z_threshold]
        print(f"  Z-score Outliers (|z| > {z_threshold}): {len(outliers):,} ({len(outliers)/len(data)*100:.2f}%)")
    
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return outliers


def show_correlation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (10, 8)
) -> pd.DataFrame:
    """
    Display correlation analysis and heatmap.
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to analyze. If None, uses all numeric columns.
        figsize: Figure size for heatmap
    
    Returns:
        Correlation matrix DataFrame
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    available_cols = [col for col in columns if col in df.columns]
    
    if len(available_cols) < 2:
        print("\nNeed at least 2 numeric columns for correlation analysis!")
        return pd.DataFrame()
    
    corr_matrix = df[available_cols].corr()
    
    print("\nCorrelation Matrix:")
    display(corr_matrix)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Heatmap", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def show_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    figsize: tuple = (10, 6)
) -> None:
    """
    Display scatter plot for bivariate analysis.
    
    Args:
        df: DataFrame to analyze
        x_col: Column for x-axis
        y_col: Column for y-axis
        figsize: Figure size
    """
    print(f"\n" + "=" * 60)
    print(f"SCATTER PLOT: {x_col} vs {y_col}")
    print("=" * 60)
    
    plt.figure(figsize=figsize)
    plt.scatter(df[x_col], df[y_col], alpha=0.5, s=10)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.title(f"{x_col} vs {y_col}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Calculate correlation
    corr = df[[x_col, y_col]].corr().iloc[0, 1]
    print(f"\nCorrelation: {corr:.4f}")


def show_group_analysis(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    top_n: int = 10
) -> None:
    """
    Display group-level KPI analysis (loss ratio by groups).
    
    Args:
        df: DataFrame to analyze
        group_cols: List of columns to group by
        top_n: Number of top groups to display
    """
    print("\n" + "=" * 60)
    print("GROUP-LEVEL KPI ANALYSIS")
    print("=" * 60)
    
    if group_cols is None:
        group_cols = ['Province', 'VehicleType', 'Gender']
    
    # Check if required columns exist
    if 'TotalPremium' not in df.columns or 'TotalClaims' not in df.columns:
        print("\n⚠ TotalPremium and TotalClaims columns required for KPI analysis")
        return
    
    for group_col in group_cols:
        if group_col not in df.columns:
            print(f"\n⚠ Skipping {group_col} - column not found")
            continue
        
        print(f"\n{group_col} - Top {top_n} by Loss Ratio:")
        
        grouped = df.groupby(group_col).agg({
            'PolicyID': 'count',
            'TotalPremium': 'sum',
            'TotalClaims': 'sum'
        }).rename(columns={'PolicyID': 'policies'})
        
        grouped.columns = ['policies', 'total_premium', 'total_claims']
        grouped['loss_ratio'] = grouped['total_claims'] / grouped['total_premium']
        grouped = grouped.sort_values('loss_ratio', ascending=False)
        
        display(grouped.head(top_n))


def show_temporal_trends(
    df: pd.DataFrame,
    date_col: str = "TransactionMonth",
    figsize: tuple = (14, 6)
) -> pd.DataFrame:
    """
    Display temporal trend analysis.
    
    Args:
        df: DataFrame to analyze
        date_col: Date column for temporal analysis
        figsize: Figure size for plots
    
    Returns:
        DataFrame with temporal trends
    """
    print("\n" + "=" * 60)
    print("TEMPORAL TREND ANALYSIS")
    print("=" * 60)
    
    if date_col not in df.columns:
        print(f"\n⚠ {date_col} column not found")
        return pd.DataFrame()
    
    # Convert to datetime if needed
    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
    
    # Group by date
    temporal = df_temp.groupby(date_col).agg({
        'PolicyID': 'count',
        'TotalPremium': 'sum',
        'TotalClaims': 'sum'
    }).rename(columns={'PolicyID': 'policies'})
    
    temporal.columns = ['policies', 'total_premium', 'total_claims']
    temporal['loss_ratio'] = temporal['total_claims'] / temporal['total_premium']
    
    print("\nMonthly Trends:")
    display(temporal)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Policies over time
    axes[0, 0].plot(temporal.index, temporal['policies'], marker='o')
    axes[0, 0].set_title('Policies Over Time', fontweight='bold')
    axes[0, 0].set_ylabel('Number of Policies')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Premium over time
    axes[0, 1].plot(temporal.index, temporal['total_premium'], marker='o', color='green')
    axes[0, 1].set_title('Total Premium Over Time', fontweight='bold')
    axes[0, 1].set_ylabel('Total Premium')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Claims over time
    axes[1, 0].plot(temporal.index, temporal['total_claims'], marker='o', color='red')
    axes[1, 0].set_title('Total Claims Over Time', fontweight='bold')
    axes[1, 0].set_ylabel('Total Claims')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss ratio over time
    axes[1, 1].plot(temporal.index, temporal['loss_ratio'], marker='o', color='purple')
    axes[1, 1].set_title('Loss Ratio Over Time', fontweight='bold')
    axes[1, 1].set_ylabel('Loss Ratio')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return temporal


def run_full_inline_analysis(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    group_cols: Optional[List[str]] = None
) -> None:
    """
    Run complete EDA pipeline with inline display.
    
    Args:
        df: DataFrame to analyze
        numeric_cols: List of numeric columns to analyze
        group_cols: List of categorical columns for grouping
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EDA ANALYSIS")
    print("=" * 60)
    
    # Default columns
    if numeric_cols is None:
        numeric_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'SumInsured']
    
    if group_cols is None:
        group_cols = ['Province', 'VehicleType', 'Gender']
    
    # 1. Data summary
    show_data_summary(df)
    
    # 2. Missing values
    show_missing_values(df)
    
    # 3. Descriptive statistics
    show_descriptive_stats(df, numeric_cols)
    
    # 4. Distributions
    show_distributions(df, numeric_cols)
    
    # 5. Outlier detection
    print("\n" + "=" * 60)
    print("OUTLIER DETECTION")
    print("=" * 60)
    for col in numeric_cols:
        if col in df.columns:
            show_outliers(df, col, method="iqr")
            show_outliers(df, col, method="zscore")
    
    # 6. Correlation
    show_correlation(df, numeric_cols)
    
    # 7. Scatter plot
    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        show_scatter(df, 'TotalPremium', 'TotalClaims')
    
    # 8. Group analysis
    show_group_analysis(df, group_cols)
    
    # 9. Temporal trends
    if 'TransactionMonth' in df.columns:
        show_temporal_trends(df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

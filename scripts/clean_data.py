import pandas as pd
import re
from datetime import datetime
from typing import List, Optional, Union
from pathlib import Path


class DataCleaner:
    """
    A class for cleaning and processing insurance data.
    
    This class handles:
    - Loading pipe-delimited data files
    - Converting columns to proper numeric types
    - Normalizing dates to yyyy-mm-dd format
    - Calculating derived metrics like loss ratio
    - Saving cleaned data to CSV
    
    Attributes:
        numeric_columns (List[str]): List of columns to convert to numeric
        date_columns (List[str]): List of columns to normalize as dates
        separator (str): Delimiter used in input files
        encoding (str): File encoding
    """
    
    DEFAULT_NUMERIC_COLUMNS = [
        "CustomValueEstimate", "CapitalOutstanding", "SumInsured",
        "CalculatedPremiumPerTerm", "TotalPremium", "TotalClaims",
        "cubiccapacity", "kilowatts", "Cylinders", "NumberOfDoors",
        "RegistrationYear", "NumberOfVehiclesInFleet"
    ]
    
    DEFAULT_DATE_COLUMNS = ["TransactionMonth", "VehicleIntroDate"]
    
    def __init__(
        self,
        numeric_columns: Optional[List[str]] = None,
        date_columns: Optional[List[str]] = None,
        separator: str = "|",
        encoding: str = "utf-8"
    ):
        """
        Initialize the DataCleaner.
        
        Args:
            numeric_columns: List of column names to convert to numeric.
                           If None, uses DEFAULT_NUMERIC_COLUMNS.
            date_columns: List of column names to normalize as dates.
                         If None, uses DEFAULT_DATE_COLUMNS.
            separator: Delimiter used in input files (default: "|")
            encoding: File encoding (default: "utf-8")
        """
        self.numeric_columns = numeric_columns or self.DEFAULT_NUMERIC_COLUMNS
        self.date_columns = date_columns or self.DEFAULT_DATE_COLUMNS
        self.separator = separator
        self.encoding = encoding
    
    def load_data(
        self,
        filepath: Union[str, Path],
        sep: Optional[str] = None,
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data from a delimited file.
        
        Args:
            filepath: Path to the input file
            sep: Delimiter (overrides instance separator if provided)
            encoding: File encoding (overrides instance encoding if provided)
        
        Returns:
            DataFrame with loaded data
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.ParserError: If the file cannot be parsed
        """
        sep = sep or self.separator
        encoding = encoding or self.encoding
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load data as strings initially
        df = pd.read_csv(
            filepath,
            sep=sep,
            dtype=str,
            encoding=encoding,
            engine="python"
        )
        
        # Trim whitespace from all string columns
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        print(f"✓ Loaded {len(df):,} rows from {filepath.name}")
        return df
    
    @staticmethod
    def _to_numeric_safe(value) -> Union[float, pd.NA]:
        """
        Safely convert a value to numeric, handling various formats.
        
        Args:
            value: Value to convert
        
        Returns:
            Float value or pd.NA if conversion fails
        """
        if pd.isna(value):
            return pd.NA
        
        # Remove non-numeric characters except decimal point, minus, and scientific notation
        cleaned = re.sub(r"[^0-9eE\.\-]", "", str(value))
        
        try:
            return float(cleaned) if cleaned else pd.NA
        except (ValueError, TypeError):
            return pd.NA
    
    def convert_to_numeric(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert specified columns to numeric type.
        
        Args:
            df: Input DataFrame
            columns: List of columns to convert. If None, uses instance numeric_columns.
        
        Returns:
            DataFrame with converted numeric columns
        """
        df = df.copy()
        columns = columns or self.numeric_columns
        
        converted_count = 0
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(self._to_numeric_safe)
                df[col] = df[col].astype("Float64")  # Nullable float type
                converted_count += 1
        
        print(f"✓ Converted {converted_count} columns to numeric")
        return df
    
    def normalize_dates(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        output_format: str = "%Y-%m-%d"
    ) -> pd.DataFrame:
        """
        Normalize date columns to a consistent format (yyyy-mm-dd by default).
        
        Args:
            df: Input DataFrame
            columns: List of date columns. If None, uses instance date_columns.
            output_format: Output date format (default: "%Y-%m-%d")
        
        Returns:
            DataFrame with normalized date columns
        """
        df = df.copy()
        columns = columns or self.date_columns
        
        converted_count = 0
        for col in columns:
            if col in df.columns:
                # Parse dates
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)
                
                # Convert to string in specified format (yyyy-mm-dd)
                # Keep as NaT for missing values
                df[col] = df[col].dt.strftime(output_format)
                converted_count += 1
        
        print(f"✓ Normalized {converted_count} date columns to {output_format}")
        return df
    
    def calculate_loss_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate loss ratio from TotalPremium and TotalClaims.
        
        Loss Ratio = TotalClaims / TotalPremium
        
        Args:
            df: Input DataFrame with TotalPremium and TotalClaims columns
        
        Returns:
            DataFrame with LossRatio column added
        """
        df = df.copy()
        
        if "TotalPremium" in df.columns and "TotalClaims" in df.columns:
            df["LossRatio"] = pd.NA
            
            # Calculate only where TotalPremium is not null and not zero
            mask = df["TotalPremium"].notna() & (df["TotalPremium"] != 0)
            df.loc[mask, "LossRatio"] = (
                df.loc[mask, "TotalClaims"] / df.loc[mask, "TotalPremium"]
            ).astype("Float64")
            
            print(f"✓ Calculated LossRatio for {mask.sum():,} rows")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning transformations to the DataFrame.
        
        This method:
        1. Trims whitespace from string columns
        2. Converts numeric columns to proper types
        3. Normalizes date columns to yyyy-mm-dd format
        4. Calculates loss ratio
        5. Cleans categorical columns
        
        Args:
            df: Input DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        print("\n=== Starting Data Cleaning ===")
        
        # Trim whitespace (already done in load_data, but ensure it's done)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Convert numeric columns
        df = self.convert_to_numeric(df)
        
        # Normalize dates
        df = self.normalize_dates(df)
        
        # Calculate loss ratio
        df = self.calculate_loss_ratio(df)
        
        # Clean categorical columns
        if "Province" in df.columns:
            df["Province"] = df["Province"].str.title().replace({"Gauteng ": "Gauteng"})
        
        print("=== Data Cleaning Complete ===\n")
        return df
    
    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_to_cleaned: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Master function to load, clean, and optionally save data.
        
        This is the main entry point for processing a data file.
        
        Args:
            input_path: Path to input file
            output_path: Optional path to save cleaned data. If None and save_to_cleaned=True,
                        saves to same directory as input with "_cleaned" suffix
            save_to_cleaned: If True and output_path is None, automatically saves to same directory as input
            **kwargs: Additional arguments passed to load_data
        
        Returns:
            Cleaned DataFrame
        
        Example:
            >>> cleaner = DataCleaner()
            >>> # Saves to data/raw_data_cleaned.csv by default
            >>> df = cleaner.process_file("data/raw_data.txt")
            >>> # Custom output path
            >>> df = cleaner.process_file("data/raw_data.txt", "output/custom.csv")
            >>> # Don't save
            >>> df = cleaner.process_file("data/raw_data.txt", save_to_cleaned=False)
        """
        # Load data
        df = self.load_data(input_path, **kwargs)
        
        # Clean data
        df = self.clean_data(df)
        
        # Determine output path
        if output_path is None and save_to_cleaned:
            input_path = Path(input_path)
            # Save in the same directory as the input file with "_cleaned" suffix
            output_filename = input_path.stem + "_cleaned.csv"
            output_path = input_path.parent / output_filename
        
        # Save if output path is determined
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✓ Saved cleaned data to {output_path}")
        
        return df


# Convenience function for backward compatibility
def to_csv(infile: Union[str, Path], sep: str = "|", encoding: str = "utf-8") -> pd.DataFrame:
    """
    Legacy function for loading delimited files.
    
    Args:
        infile: Path to input file
        sep: Delimiter
        encoding: File encoding
    
    Returns:
        DataFrame with loaded data
    """
    cleaner = DataCleaner(separator=sep, encoding=encoding)
    return cleaner.load_data(infile)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function for cleaning a DataFrame.
    
    Args:
        df: Input DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner()
    return cleaner.clean_data(df)

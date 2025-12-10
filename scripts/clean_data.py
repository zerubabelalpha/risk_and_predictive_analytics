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
    
    def convert_booleans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert boolean-like columns to binary 0/1.
        """
        df = df.copy()
        bool_cols = [
            'IsVATRegistered', 'AlarmImmobiliser', 'TrackingDevice',
            'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder'
        ]
        
        mapping = {'Yes': 1, 'No': 0, True: 1, False: 0, 'True': 1, 'False': 0}
        
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0) # Assume 0 if missing/unknown after map
                
        print(f"✓ Converted boolean columns to binary")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features like VehicleAge.
        """
        df = df.copy()
        
        # Vehicle Age
        # Ensure we have numeric years. RegistrationYear should be numeric from convert_to_numeric
        # TransactionMonth (string) -> extract year
        if 'TransactionMonth' in df.columns and 'RegistrationYear' in df.columns:
            # Create temp TransactionYear
            try:
                # Handle YYYY-MM-DD format from normalize_dates
                df['TransactionYear'] = pd.to_datetime(df['TransactionMonth'], errors='coerce').dt.year
                df['VehicleAge'] = df['TransactionYear'] - df['RegistrationYear']
                # Drop temp column if not needed or keep it
                # df = df.drop('TransactionYear', axis=1)
                print(f"✓ Created VehicleAge feature")
            except Exception as e:
                print(f"⚠ Could not create VehicleAge: {e}")
                
        return df

    def impute_gender_from_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create/Impute 'Gender' column from 'Title' column.
        """
        df = df.copy()
        if 'Title' in df.columns:
            # Map titles to gender
            title_map = {
                'Mr': 'Male',
                'Mrs': 'Female',
                'Ms': 'Female',
                'Miss': 'Female',
                'Dr': 'Unknown',
            }
            
            def get_gender(title):
                if pd.isna(title):
                    return pd.NA
                t = str(title).strip().replace('.', '')
                if t in title_map:
                    return title_map[t]
                # Fallback for common prefixes
                if t.startswith('Mr'): return 'Male'
                if t.startswith('Mrs') or t.startswith('Ms') or t.startswith('Miss'): return 'Female'
                return pd.NA
            
            # If Gender column doesn't exist, create it
            if 'Gender' not in df.columns:
                df['Gender'] = pd.NA
            else:
                # Treat empty strings and whitespace as NA
                df['Gender'] = df['Gender'].replace(r'^\s*$', pd.NA, regex=True)
            
            # Fill missing Gender values
            # We use a temporary series to map
            inferred = df['Title'].apply(get_gender)
            df['Gender'] = df['Gender'].fillna(inferred)
            
            print(f"✓ Imputed Gender from Title")
            
        return df

    def handle_missing_values(self, df: pd.DataFrame, drop_threshold_pct: float = 0.2) -> pd.DataFrame:
        """
        Handle missing values:
        1. Replace strings like '', 'Not specified' with NaN
        2. Drop columns with > (1-drop_threshold_pct) missing values (e.g. >80% missing)
        3. Fill categorical with 'Unknown'
        4. Fill numeric with median
        """
        df = df.copy()
        
        # 1. Standardize missing
        df.replace(['', 'Not specified', 'nan', 'None'], pd.NA, inplace=True)
        
        # 2. Drop sparse columns
        # thresh = require at least N non-NA values
        thresh = len(df) * drop_threshold_pct
        start_cols = len(df.columns)
        df = df.dropna(axis=1, thresh=thresh)
        dropped_count = start_cols - len(df.columns)
        if dropped_count > 0:
            print(f"✓ Dropped {dropped_count} columns with >{100*(1-drop_threshold_pct)}% missing values")

        # 3. Fill Categorical
        cat_cols = df.select_dtypes(include=['object', 'string']).columns
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
            
        # 4. Fill Numeric
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
        print(f"✓ Imputed missing values (Cat: Unknown, Num: Median)")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        orig_len = len(df)
        df = df.drop_duplicates(keep='first')
        dropped = orig_len - len(df)
        if dropped > 0:
            print(f"✓ Removed {dropped:,} duplicate rows")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning transformations to the DataFrame.
        
        Pipeline:
        1. Trim whitespace
        2. Replace missing-like strings
        3. Convert numeric
        4. Normalize dates
        5. Convert booleans
        6. Feature engineering (VehicleAge)
        7. Handle missing values (Drop sparse, Impute)
        8. Remove duplicates
        9. Calculate loss ratio
        """
        print("\n=== Starting Data Cleaning ===")
        
        # 1. Trim
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        
       #2.
        
        # 3. Numeric
        df = self.convert_to_numeric(df)
        
        # 4. Dates
        df = self.normalize_dates(df)
        
        # 5. Booleans
        df = self.convert_booleans(df)
        
        # 6. Features
        df = self.engineer_features(df)
        
        # 6b. Impute Gender from Title (before handling missing values)
        df = self.impute_gender_from_title(df)
        
        # 7. Missing Values
        df = self.handle_missing_values(df)
        
        # 8. Duplicates
        df = self.remove_duplicates(df)
        
        # 9. Loss Ratio
        df = self.calculate_loss_ratio(df)
        
        # Extra cleaning
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

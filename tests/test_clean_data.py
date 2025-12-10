
import pandas as pd
import pytest
from scripts.clean_data import DataCleaner

class TestDataCleaner:
    def test_impute_gender_missing_column(self):
        """Test imputation when Gender column is missing."""
        df = pd.DataFrame({
            'Title': ['Mr', 'Mrs', 'Dr', 'Miss', 'Unknown']
        })
        cleaner = DataCleaner()
        df_out = cleaner.impute_gender_from_title(df)
        
        expected_gender = ['Male', 'Female', 'Unknown', 'Female', pd.NA]
        pd.testing.assert_series_equal(
            df_out['Gender'], 
            pd.Series(expected_gender, name='Gender', dtype='object'),  # Adjust dtype if needed
            check_dtype=False
        )
        
    def test_impute_gender_na_values(self):
        """Test imputation when Gender column exists but has NA values."""
        df = pd.DataFrame({
            'Title': ['Mr', 'Mrs'],
            'Gender': [pd.NA, None]
        })
        cleaner = DataCleaner()
        df_out = cleaner.impute_gender_from_title(df)
        
        expected_gender = ['Male', 'Female']
        pd.testing.assert_series_equal(
            df_out['Gender'], 
            pd.Series(expected_gender, name='Gender'),
            check_dtype=False
        )

    def test_impute_gender_empty_strings(self):
        """Test imputation when Gender column has empty strings (The Bug Fix)."""
        df = pd.DataFrame({
            'Title': ['Mr', 'Mrs', 'Mr'],
            'Gender': ['', '  ', 'Existing']
        })
        cleaner = DataCleaner()
        df_out = cleaner.impute_gender_from_title(df)
        
        # 'Existing' should be preserved
        expected_gender = ['Male', 'Female', 'Existing']
        pd.testing.assert_series_equal(
            df_out['Gender'], 
            pd.Series(expected_gender, name='Gender'),
            check_dtype=False
        )

    def test_impute_gender_priority(self):
        """Ensure existing valid values are NOT overwritten."""
        df = pd.DataFrame({
            'Title': ['Mr'],
            'Gender': ['Female']  # Wrong gender, but should be preserved
        })
        cleaner = DataCleaner()
        df_out = cleaner.impute_gender_from_title(df)
        
        assert df_out.iloc[0]['Gender'] == 'Female'

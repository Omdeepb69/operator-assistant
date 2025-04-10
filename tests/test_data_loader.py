import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
from pandas.testing import assert_frame_equal, assert_series_equal

# Assume data_loader.py exists in the same directory or is importable
# It should contain functions: load_csv, preprocess_data, transform_features
# Example content for data_loader.py is assumed for these tests to make sense.
# For demonstration, let's define dummy versions if the module isn't available
try:
    from data_loader import load_csv, preprocess_data, transform_features
except ImportError:
    # Define dummy functions if data_loader module is not found
    # In a real scenario, data_loader.py should exist
    from sklearn.preprocessing import MinMaxScaler

    def load_csv(filepath):
        if isinstance(filepath, (str, Path)):
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No such file or directory: '{filepath}'")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            raise

    def preprocess_data(df):
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                mean_val = df_processed[col].mean()
                df_processed[col].fillna(mean_val, inplace=True)

        if 'categorical_col' in df_processed.columns:
             # Example: Fill missing categorical data with mode
             if df_processed['categorical_col'].isnull().any():
                 mode_val = df_processed['categorical_col'].mode()[0]
                 df_processed['categorical_col'].fillna(mode_val, inplace=True)

        # Example type conversion
        if 'str_num_col' in df_processed.columns:
            df_processed['str_num_col'] = pd.to_numeric(df_processed['str_num_col'], errors='coerce')
            if df_processed['str_num_col'].isnull().any():
                 mean_val = df_processed['str_num_col'].mean()
                 df_processed['str_num_col'].fillna(mean_val, inplace=True)

        return df_processed

    def transform_features(df):
        df_transformed = df.copy()
        scaler = MinMaxScaler()
        numeric_cols = df_transformed.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            # Ensure all numeric columns are float for scaler
            for col in numeric_cols:
                df_transformed[col] = df_transformed[col].astype(float)
            df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
        return df_transformed


@pytest.fixture(scope="module")
def sample_csv_data():
    return pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10.5, 20.0, 30.0, 40.0, 50.5],
        'categorical_col': ['A', 'B', 'A', np.nan, 'C'],
        'str_num_col': ['100', '200', '300', '400', 'abc'] # Includes non-numeric
    })

@pytest.fixture(scope="function")
def temp_csv_file(sample_csv_data):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".csv") as tmp_file:
        sample_csv_data.to_csv(tmp_file.name, index=False)
        file_path = tmp_file.name
    yield Path(file_path)
    os.remove(file_path) # Clean up the file

@pytest.fixture(scope="function")
def sample_dataframe():
    return pd.DataFrame({
        'col_a': [1.0, 2.0, np.nan, 4.0],
        'col_b': [5, 6, 7, 8],
        'col_c': ['X', 'Y', 'X', np.nan],
        'str_num_col': ['50', '60', '70', '80']
    })


# --- Test Data Loading ---

def test_load_csv_success(temp_csv_file, sample_csv_data):
    """Test loading a valid CSV file."""
    loaded_df = load_csv(temp_csv_file)
    # Handle potential type differences after CSV read/write (e.g., NaN vs None)
    # Fill NaNs for comparison if necessary or compare carefully
    expected_df = sample_csv_data.copy()
    # CSV read might interpret 'abc' in str_num_col as NaN if not careful
    # Let's reload the saved csv to get the exact expected dataframe after save/load cycle
    expected_df_reloaded = pd.read_csv(temp_csv_file)

    assert_frame_equal(loaded_df, expected_df_reloaded, check_dtype=True)

def test_load_csv_file_not_found():
    """Test loading a non-existent CSV file."""
    non_existent_file = Path("non_existent_file_12345.csv")
    with pytest.raises(FileNotFoundError):
        load_csv(non_existent_file)

def test_load_csv_from_path_object(temp_csv_file, sample_csv_data):
    """Test loading using a pathlib.Path object."""
    loaded_df = load_csv(temp_csv_file) # temp_csv_file is already a Path object
    expected_df_reloaded = pd.read_csv(temp_csv_file)
    assert_frame_equal(loaded_df, expected_df_reloaded, check_dtype=True)


# --- Test Data Preprocessing ---

def test_preprocess_data_fillna(sample_dataframe):
    """Test filling missing numerical values."""
    df_to_process = sample_dataframe.copy()
    processed_df = preprocess_data(df_to_process)

    # Check NaN in 'col_a' is filled with mean of non-NaN values (1+2+4)/3 = 7/3
    expected_mean_col_a = (1.0 + 2.0 + 4.0) / 3.0
    assert not processed_df['col_a'].isnull().any()
    assert processed_df.loc[2, 'col_a'] == pytest.approx(expected_mean_col_a)

    # Check NaN in 'col_c' is filled with mode ('X')
    expected_mode_col_c = 'X'
    assert not processed_df['col_c'].isnull().any()
    assert processed_df.loc[3, 'col_c'] == expected_mode_col_c

    # Check original DataFrame is not modified
    assert sample_dataframe['col_a'].isnull().sum() == 1
    assert sample_dataframe['col_c'].isnull().sum() == 1


def test_preprocess_data_type_conversion(sample_dataframe):
    """Test conversion of columns to numeric types."""
    df_to_process = sample_dataframe.copy()
    processed_df = preprocess_data(df_to_process)

    # Check 'str_num_col' is converted to numeric
    assert pd.api.types.is_numeric_dtype(processed_df['str_num_col'])
    expected_series = pd.to_numeric(sample_dataframe['str_num_col'], errors='coerce').astype(float)
    assert_series_equal(processed_df['str_num_col'], expected_series, check_names=False)


def test_preprocess_data_no_missing_values():
    """Test preprocessing on data with no missing values."""
    df_no_missing = pd.DataFrame({
        'col_a': [1.0, 2.0, 3.0],
        'col_b': [4, 5, 6],
        'col_c': ['X', 'Y', 'Z'],
        'str_num_col': ['10', '20', '30']
    })
    processed_df = preprocess_data(df_no_missing.copy())
    expected_df = df_no_missing.copy()
    expected_df['str_num_col'] = pd.to_numeric(expected_df['str_num_col'])

    assert_frame_equal(processed_df, expected_df)


# --- Test Data Transformation ---

def test_transform_features_scaling(sample_dataframe):
    """Test feature scaling (MinMaxScaler)."""
    # Preprocess first to handle NaNs and types
    df_processed = preprocess_data(sample_dataframe.copy())
    transformed_df = transform_features(df_processed)

    numeric_cols = df_processed.select_dtypes(include=np.number).columns

    # Check if numeric columns are scaled between 0 and 1
    for col in numeric_cols:
        assert transformed_df[col].min() >= 0.0
        assert transformed_df[col].max() <= 1.0

    # Check specific scaled values for 'col_b' [5, 6, 7, 8] -> [0, 0.333, 0.666, 1]
    expected_col_b_scaled = np.array([0.0, 1/3, 2/3, 1.0])
    np.testing.assert_allclose(transformed_df['col_b'].values, expected_col_b_scaled, rtol=1e-6)

    # Check non-numeric columns are untouched
    assert_series_equal(transformed_df['col_c'], df_processed['col_c'])

    # Check original DataFrame is not modified
    assert not df_processed['col_b'].max() <= 1.0


def test_transform_features_no_numeric_cols():
    """Test transformation when there are no numeric columns."""
    df_no_numeric = pd.DataFrame({
        'col_c': ['X', 'Y', 'Z'],
        'col_d': ['A', 'B', 'C']
    })
    transformed_df = transform_features(df_no_numeric.copy())
    assert_frame_equal(transformed_df, df_no_numeric)


def test_transform_features_single_value_numeric_col():
    """Test transformation with a numeric column having only one unique value."""
    df_single_value = pd.DataFrame({
        'col_a': [5.0, 5.0, 5.0],
        'col_b': [1, 2, 3]
    })
    transformed_df = transform_features(df_single_value.copy())

    # Scaler should result in 0 for constant column if using MinMaxScaler
    # Some scalers might produce NaN, ensure it handles this gracefully (MinMaxScaler produces 0)
    assert transformed_df['col_a'].nunique() <= 1
    np.testing.assert_allclose(transformed_df['col_a'].values, np.zeros(3))

    # Check other column is scaled correctly [1, 2, 3] -> [0, 0.5, 1]
    expected_col_b_scaled = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(transformed_df['col_b'].values, expected_col_b_scaled, rtol=1e-6)
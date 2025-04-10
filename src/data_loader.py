```python
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import logging # Keep import for potential future use, but remove calls

DEFAULT_DATA_PATH = 'data/commands.csv'
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_RANDOM_STATE = 42

def load_environment_variables(env_path=None):
    try:
        load_dotenv(dotenv_path=env_path)
        return True
    except Exception as e:
        # In a real app, log this error: logging.error(f"Error loading environment variables: {e}")
        print(f"Error loading environment variables: {e}") # Basic feedback if logging is removed
        return False

def load_raw_data(file_path=DEFAULT_DATA_PATH):
    if not os.path.exists(file_path):
        # logging.warning(f"Data file not found at {file_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=['command_text', 'intent'])

    try:
        df = pd.read_csv(file_path)
        # logging.info(f"Raw data loaded successfully from {file_path}.")
        if 'command_text' not in df.columns or 'intent' not in df.columns:
             # logging.error(f"Missing required columns ('command_text', 'intent') in {file_path}")
             return pd.DataFrame(columns=['command_text', 'intent'])
        return df
    except pd.errors.EmptyDataError:
        # logging.warning(f"Data file {file_path} is empty. Returning empty DataFrame.")
        return pd.DataFrame(columns=['command_text', 'intent'])
    except Exception as e:
        # logging.error(f"Error loading data from {file_path}: {e}")
        print(f"Error loading data from {file_path}: {e}") # Basic feedback
        return pd.DataFrame(columns=['command_text', 'intent'])


def preprocess_data(df):
    if df.empty:
        # logging.warning("Preprocessing skipped: Input DataFrame is empty.")
        return df

    try:
        if 'command_text' in df.columns:
            df['command_text'] = df['command_text'].fillna('')
        else:
            # logging.warning("Column 'command_text' not found for preprocessing.")
            return df

        if 'intent' in df.columns:
            df.dropna(subset=['intent'], inplace=True)
        else:
            # logging.warning("Column 'intent' not found for preprocessing (dropna step).")
            pass # Continue processing other columns if needed

        def clean_text(text):
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        if 'command_text' in df.columns:
            df['processed_text'] = df['command_text'].apply(clean_text)
            # logging.info("Text preprocessing (lowercase, punctuation removal) applied.")
        else:
             # logging.warning("Column 'command_text' not found for text cleaning.")
             pass

        # logging.info("Data preprocessing completed.")
        return df
    except Exception as e:
        # logging.error(f"Error during data preprocessing: {e}")
        print(f"Error during data preprocessing: {e}") # Basic feedback
        return df

def transform_features(df):
    if df.empty:
        # logging.warning("Feature transformation skipped: Input DataFrame is empty.")
        return df
    if 'processed_text' not in df.columns:
        # logging.warning("Feature transformation skipped: 'processed_text' column not found.")
        return df

    try:
        df['tokens'] = df['processed_text'].apply(lambda x: x.split())
        # logging.info("Feature transformation (tokenization) applied.")
        return df
    except Exception as e:
        # logging.error(f"Error during feature transformation: {e}")
        print(f"Error during feature transformation: {e}") # Basic feedback
        return df


def split_data(df, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE):
    if df.empty or 'intent' not in df.columns:
        # logging.warning("Data splitting skipped: DataFrame is empty or missing 'intent' column.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    target = df['intent'] if 'intent' in df.columns else None
    stratify_param = target if target is not None and len(df) > 1 else None

    if (1 - test_size) <= 0:
         # logging.error("test_size must be less than 1.")
         print("Error: test_size must be less than 1.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if test_size + val_size >= 1.0:
        # logging.error("Sum of test_size and val_size must be less than 1.0")
        print("Error: Sum of test_size and val_size must be less than 1.0")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    min_samples_for_split = 3
    if len(df) < min_samples_for_split:
         # logging.warning(f"Insufficient data for splitting (found {len(df)} samples, need at least {min_samples_for_split}). Returning entire dataset as training set.")
         print(f"Warning: Insufficient data for splitting (found {len(df)} samples). Returning entire dataset as training set.")
         return df.copy(), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    try:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        if len(train_val_df) == 0:
             # logging.warning("No data left for training/validation after test split. Returning empty train/val sets.")
             print("Warning: No data left for training/validation after test split.")
             return pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns), test_df

        val_size_relative = val_size / (1 - test_size)

        if val_size_relative >= 1.0 or len(train_val_df) < 2:
             # logging.warning(f"Insufficient data ({len(train_val_df)} samples) for validation split or invalid val_size_relative ({val_size_relative:.2f}). Assigning all remaining to training set.")
             print(f"Warning: Insufficient data ({len(train_val_df)}) or invalid relative val size ({val_size_relative:.2f}) for validation split. Assigning remaining to training.")
             train_df = train_val_df
             val_df = pd.DataFrame(columns=df.columns)
        else:
             stratify_train_val = train_val_df['intent'] if 'intent' in train_val_df.columns else None
             train_df, val_df = train_test_split(
                 train_val_df,
                 test_size=val_size_relative,
                 random_state=random_state,
                 stratify=stratify_train_val
             )

        # logging.info(f"Data split into Train ({len(train_df)}), Validation ({len(val_df)}), Test ({len(test_df)}) sets.")
        return train_df, val_df, test_df

    except ValueError as e:
         # logging.error(f"Error during data splitting: {e}. Check data size and class distribution.")
         print(f"Error during data splitting: {e}. Check data size/distribution.") # Basic feedback
         if len(df) > 0:
             # logging.warning("Splitting failed, returning entire dataset as training set.")
             print("Warning: Splitting failed, returning entire dataset as training set.")
             return df.copy(), pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)
         else:
             return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        # logging.error(f"An unexpected error occurred during data splitting: {e}")
        print(f"An unexpected error occurred during data splitting: {e}") # Basic feedback
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def load_and_prepare_data(file_path=DEFAULT_DATA_PATH, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE, random_state=DEFAULT_RANDOM_STATE, env_path=None):
    load_environment_variables(env_path)

    raw_df = load_raw_data(file_path)
    if raw_df.empty:
        # logging.error("Failed to load or data is empty. Cannot proceed.")
        print("Error: Failed to
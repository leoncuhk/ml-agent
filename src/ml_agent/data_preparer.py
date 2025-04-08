import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
import numpy as np
import logging
from typing import List, Union, Tuple, Optional
import re
import inspect # Import inspect module
import sys # Import sys module

# Get the logger instance from the main agent module or configure a specific one
logger = logging.getLogger('ml_agent.data_preparer')

def impute_missing(df: pd.DataFrame, column: str, strategy: str = 'mean') -> pd.DataFrame:
    """
    Imputes missing values in a specified column using mean, median, or mode.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): The column to impute.
        strategy (str): Imputation strategy ('mean', 'median', 'mode').

    Returns:
        pd.DataFrame: DataFrame with imputed values (modified in place).
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for imputation. Skipping.")
        return df

    if df[column].isnull().sum() == 0:
        logger.info(f"No missing values found in column '{column}'. Skipping imputation.")
        return df

    original_dtype = df[column].dtype

    try:
        fill_value = None
        if strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].mean()
                logger.info(f"Imputing missing values in '{column}' with mean: {fill_value}")
            else:
                logger.warning(f"Mean imputation requested for non-numeric column '{column}'. Skipping.")
                return df
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df[column]):
                fill_value = df[column].median()
                logger.info(f"Imputing missing values in '{column}' with median: {fill_value}")
            else:
                logger.warning(f"Median imputation requested for non-numeric column '{column}'. Skipping.")
                return df
        elif strategy == 'mode':
            # Check if mode() returns an empty series (can happen)
            mode_values = df[column].mode()
            if not mode_values.empty:
                fill_value = mode_values[0]
                logger.info(f"Imputing missing values in '{column}' with mode: {fill_value}")
            else:
                 logger.warning(f"Could not determine mode for column '{column}'. Skipping imputation.")
                 return df
        else:
            logger.warning(f"Invalid imputation strategy '{strategy}' for column '{column}'. Skipping.")
            return df

        df[column].fillna(fill_value, inplace=True) # Modify in place

        # Attempt to restore original dtype if possible (e.g., int after mean imputation)
        # Check if original dtype was integer-like and current is float
        if pd.api.types.is_integer_dtype(original_dtype) and pd.api.types.is_float_dtype(df[column].dtype):
             # Check if all values are effectively integers
             if (df[column].dropna() % 1 == 0).all():
                 try:
                     # Convert to nullable integer type first to handle potential NaNs introduced if fill_value was NaN
                     df[column] = df[column].astype(pd.Int64Dtype())
                     # Then try converting back to the original specific integer type if needed
                     df[column] = df[column].astype(original_dtype)
                     logger.debug(f"Restored dtype for column '{column}' to {original_dtype}")
                 except Exception as e:
                      logger.debug(f"Could not fully restore original integer dtype for column '{column}': {e}")

    except Exception as e:
        logger.error(f"Error during imputation for column '{column}' with strategy '{strategy}': {e}", exc_info=True)

    return df # Return df even though modified in place for potential chaining


def encode_categorical(df: pd.DataFrame, column: str, strategy: str = 'one-hot', drop_original: bool = True, drop_first: bool = False) -> pd.DataFrame:
    """
    Encodes a categorical column using one-hot encoding.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): The categorical column to encode.
        strategy (str): Encoding strategy (currently only 'one-hot').
        drop_original (bool): Whether to drop the original column after encoding.
        drop_first (bool): Whether to drop the first category in one-hot encoding
                           to avoid multicollinearity.

    Returns:
        pd.DataFrame: DataFrame with the encoded column(s).
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for encoding. Skipping.")
        return df

    # Allow encoding for object or category types
    if not pd.api.types.is_categorical_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
         logger.warning(f"Column '{column}' is not categorical or object type. Skipping encoding.")
         return df

    try:
        if strategy == 'one-hot':
            logger.info(f"Applying one-hot encoding to column '{column}' (drop_first={drop_first}).")
            # Ensure the column is treated as string before getting dummies
            df_encoded = pd.get_dummies(df[[column]].astype(str), columns=[column], prefix=column, prefix_sep='_', drop_first=drop_first)

            # Clean generated column names (replace spaces, remove special chars)
            df_encoded.columns = [re.sub(r'[^A-Za-z0-9_]+', '', col.replace(' ', '_')) for col in df_encoded.columns]

            # Concatenate and optionally drop original
            df = pd.concat([df, df_encoded], axis=1)
            if drop_original:
                df.drop(column, axis=1, inplace=True)
                logger.info(f"Original column '{column}' dropped after one-hot encoding.")
        else:
            logger.warning(f"Invalid encoding strategy '{strategy}' for column '{column}'. Skipping.")

    except Exception as e:
        logger.error(f"Error during encoding for column '{column}' with strategy '{strategy}': {e}", exc_info=True)

    return df

# --- REFACTORED FUNCTIONS --- #

def handle_outliers(df: pd.DataFrame, columns: List[str], strategy: str = 'clip', iqr_multiplier: float = 1.5, std_dev_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Handles outliers in specified numerical columns using IQR clipping or Std Dev clipping.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of numerical columns to handle outliers in.
        strategy (str): Method ('clip' for IQR, 'clip_std' for Standard Deviation).
        iqr_multiplier (float): Multiplier for IQR range (used if strategy='clip').
        std_dev_multiplier (float): Multiplier for Standard Deviation (used if strategy='clip_std').

    Returns:
        pd.DataFrame: DataFrame with outliers handled (modified in place).
    """
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found for outlier handling. Skipping.")
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.warning(f"Outlier handling requested for non-numeric column '{column}'. Skipping.")
            continue

        try:
            lower_bound, upper_bound = None, None
            original_values = df[column].copy()

            if strategy == 'clip':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR
                method_desc = f"IQR * {iqr_multiplier}"

            elif strategy == 'clip_std':
                mean = df[column].mean()
                std_dev = df[column].std()
                lower_bound = mean - std_dev_multiplier * std_dev
                upper_bound = mean + std_dev_multiplier * std_dev
                method_desc = f"Mean +/- {std_dev_multiplier}*StdDev"

            else:
                logger.warning(f"Invalid outlier handling strategy '{strategy}' for column '{column}'. Skipping.")
                continue

            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound) # Use clip method
            clipped_count = (original_values != df[column]).sum() # Compare before/after clip

            if clipped_count > 0:
                 logger.info(f"Clipped {clipped_count} potential outliers in column '{column}' to range [{lower_bound:.2f}, {upper_bound:.2f}] using {method_desc}.")
            else:
                 logger.info(f"No outliers needed clipping in column '{column}' based on {method_desc}.")

        except Exception as e:
            logger.error(f"Error during outlier handling for column '{column}' with strategy '{strategy}': {e}", exc_info=True)

    return df # Return df for potential chaining


def scale_numerical(df: pd.DataFrame, columns: List[str], strategy: str = 'standard') -> Tuple[pd.DataFrame, Optional[object]]:
    """
    Scales specified numerical columns using StandardScaler or MinMaxScaler.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): List of numerical columns to scale.
        strategy (str): Scaling method ('standard' or 'minmax').

    Returns:
        Tuple[pd.DataFrame, Optional[object]]: DataFrame with scaled columns (modified in place), and the fitted scaler object.
    """
    valid_columns = []
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found for scaling. Skipping.")
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            logger.warning(f"Scaling requested for non-numeric column '{column}'. Skipping.")
            continue
        valid_columns.append(column)

    if not valid_columns:
        logger.warning("No valid numerical columns found to scale.")
        return df, None

    scaler = None
    try:
        # Select only the valid columns to scale
        data_to_scale = df[valid_columns].astype(float) # Ensure float type for scaler

        if strategy == 'standard':
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_to_scale)
            logger.info(f"Applied StandardScaler to columns: {valid_columns}.")
        elif strategy == 'minmax':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_to_scale)
            logger.info(f"Applied MinMaxScaler to columns: {valid_columns}.")
        else:
            logger.warning(f"Invalid scaling strategy '{strategy}' for columns: {valid_columns}. Skipping.")
            return df, None

        # Assign scaled data back to the DataFrame
        df[valid_columns] = scaled_data

    except Exception as e:
        logger.error(f"Error during scaling for columns '{valid_columns}' with strategy '{strategy}': {e}", exc_info=True)
        return df, None # Return original df on error

    return df, scaler # Return modified df and the scaler


def extract_date_features(df: pd.DataFrame, column: str, drop_original: bool = True) -> pd.DataFrame:
    """
    Extracts date components (Year, Month, Day, DayOfWeek, DayOfYear) from a datetime column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): The datetime column to extract features from.
        drop_original (bool): Whether to drop the original datetime column.

    Returns:
        pd.DataFrame: DataFrame with new date features (modified in place).
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for date feature extraction. Skipping.")
        return df

    try:
        # Attempt to convert to datetime if not already, handling potential errors
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            logger.info(f"Attempting to convert column '{column}' to datetime.")
            original_series = df[column].copy() # Keep original in case of failure
            try:
                 # Try inferring format first for speed
                 df[column] = pd.to_datetime(df[column], infer_datetime_format=True, errors='coerce')
            except ValueError: # Fallback if infer fails or mixed formats
                 df[column] = pd.to_datetime(df[column], errors='coerce')


            # Check if conversion failed significantly (more than 10% new NaNs)
            original_nulls = original_series.isnull().sum()
            new_nulls = df[column].isnull().sum()
            if new_nulls > original_nulls + int(len(df) * 0.1):
                 logger.warning(f"Significant errors converting '{column}' to datetime ({new_nulls - original_nulls} new NaNs). Reverting and skipping extraction.")
                 df[column] = original_series # Revert
                 return df
            # Impute any coerced NaTs if needed - ffill/bfill is often reasonable for dates
            if df[column].isnull().any():
                 df[column].fillna(method='ffill', inplace=True)
                 df[column].fillna(method='bfill', inplace=True) # Handle potential leading NaNs
                 logger.info(f"Imputed NaT values in '{column}' after conversion using ffill/bfill.")

        # Check again if column is now datetime after conversion attempt
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            logger.warning(f"Column '{column}' could not be converted to datetime. Skipping feature extraction.")
            return df

        logger.info(f"Extracting date features from column '{column}'.")
        dt_col = df[column].dt

        df[f'{column}_Year'] = dt_col.year
        df[f'{column}_Month'] = dt_col.month
        df[f'{column}_Day'] = dt_col.day
        df[f'{column}_DayOfWeek'] = dt_col.dayofweek
        df[f'{column}_DayOfYear'] = dt_col.dayofyear
        # df[f'{column}_WeekOfYear'] = dt_col.isocalendar().week # isocalendar() returns DataFrame, handle carefully
        # df[f'{column}_Hour'] = dt_col.hour # If time component is relevant

        if drop_original:
            df.drop(column, axis=1, inplace=True)
            logger.info(f"Original datetime column '{column}' dropped.")

    except Exception as e:
        logger.error(f"Error extracting date features from column '{column}': {e}", exc_info=True)

    return df

# --- NEW FUNCTIONS FROM EXAMPLES --- #

def convert_to_numeric(df: pd.DataFrame, column: str, errors: str = 'coerce') -> pd.DataFrame:
    """
    Converts a specified column to a numeric data type.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): The column to convert.
        errors (str): How to handle parsing errors ('raise', 'coerce', 'ignore').
                      'coerce' will replace unparseable values with NaN.

    Returns:
        pd.DataFrame: DataFrame with the column converted (modified in place).
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for numeric conversion. Skipping.")
        return df

    if pd.api.types.is_numeric_dtype(df[column]):
        logger.info(f"Column '{column}' is already numeric. Skipping conversion.")
        return df

    try:
        original_nulls = df[column].isnull().sum()
        df[column] = pd.to_numeric(df[column], errors=errors)
        new_nulls = df[column].isnull().sum()
        if errors == 'coerce' and new_nulls > original_nulls:
            logger.warning(f"Coerced {new_nulls - original_nulls} values to NaN in column '{column}' during numeric conversion.")
        logger.info(f"Converted column '{column}' to numeric (dtype: {df[column].dtype}).")
    except Exception as e:
        logger.error(f"Error converting column '{column}' to numeric: {e}", exc_info=True)

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with duplicate rows removed.
    """
    try:
        initial_rows = len(df)
        df.drop_duplicates(inplace=True)
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} duplicate rows.")
        else:
            logger.info("No duplicate rows found.")
    except Exception as e:
        logger.error(f"Error removing duplicate rows: {e}", exc_info=True)
    return df

def strip_whitespace(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Strips leading/trailing whitespace from string values in a specified column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): The object/string column to strip whitespace from.

    Returns:
        pd.DataFrame: DataFrame with whitespace stripped (modified in place).
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for whitespace stripping. Skipping.")
        return df

    if not pd.api.types.is_object_dtype(df[column]):
         # Only attempt on object columns, maybe add category?
         logger.warning(f"Whitespace stripping requested for non-object column '{column}'. Skipping.")
         return df

    try:
        # Check for actual changes to log meaningfully
        original_series = df[column].copy()
        df[column] = df[column].str.strip()
        changes = (original_series != df[column]).sum()
        if changes > 0:
             logger.info(f"Stripped whitespace from {changes} values in column '{column}'.")
        else:
             logger.debug(f"No leading/trailing whitespace found in column '{column}'.") # Use debug if no change
    except Exception as e:
        logger.error(f"Error stripping whitespace from column '{column}': {e}", exc_info=True)

    return df

def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes columns that have only one unique value (constants).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with constant columns removed.
    """
    constant_cols = []
    try:
        for col in df.columns:
            if df[col].nunique(dropna=False) == 1: # dropna=False includes NaN as a potential unique value
                constant_cols.append(col)

        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)
            logger.info(f"Removed constant columns: {constant_cols}")
        else:
            logger.info("No constant columns found.")
    except Exception as e:
        logger.error(f"Error removing constant columns: {e}", exc_info=True)
    return df

def handle_high_cardinality(df: pd.DataFrame, column: str, threshold: float = 0.05, replace_with: str = 'Other') -> pd.DataFrame:
    """
    Handles high cardinality in a categorical column by replacing infrequent values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): The categorical column to process.
        threshold (float): Frequency threshold (percentage of total rows).
                           Categories below this threshold are replaced.
        replace_with (str): The value to replace infrequent categories with.

    Returns:
        pd.DataFrame: DataFrame with high cardinality handled (modified in place).
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for high cardinality handling. Skipping.")
        return df

    if not pd.api.types.is_object_dtype(df[column]) and not pd.api.types.is_categorical_dtype(df[column]):
         logger.warning(f"High cardinality handling requested for non-categorical/object column '{column}'. Skipping.")
         return df

    try:
        # Calculate the frequency threshold in terms of counts
        count_threshold = int(threshold * len(df))
        logger.info(f"Handling high cardinality in '{column}'. Threshold count: {count_threshold} ({threshold*100:.1f}%).")

        # Get value counts
        value_counts = df[column].value_counts()

        # Identify infrequent values
        infrequent_values = value_counts[value_counts < count_threshold].index.tolist()

        if infrequent_values:
            # Replace infrequent values
            df[column] = df[column].apply(lambda x: replace_with if x in infrequent_values else x)
            logger.info(f"Replaced {len(infrequent_values)} infrequent categories in '{column}' with '{replace_with}'.")
        else:
            logger.info(f"No infrequent categories found in '{column}' below the threshold.")

    except Exception as e:
        logger.error(f"Error handling high cardinality in column '{column}': {e}", exc_info=True)

    return df


def encode_label(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Optional[LabelEncoder]]:
    """
    Encodes a target or binary categorical column using LabelEncoder.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): The column to encode.

    Returns:
        Tuple[pd.DataFrame, Optional[LabelEncoder]]: DataFrame with the column label encoded (modified in place),
                                                      and the fitted LabelEncoder object.
    """
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for label encoding. Skipping.")
        return df, None

    encoder = None
    try:
        encoder = LabelEncoder()
        original_dtype = df[column].dtype
        df[column] = encoder.fit_transform(df[column])
        logger.info(f"Applied LabelEncoder to column '{column}'. Original dtype: {original_dtype}, New dtype: {df[column].dtype}.")
        logger.info(f"LabelEncoder classes for '{column}': {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")

    except Exception as e:
        logger.error(f"Error during label encoding for column '{column}': {e}", exc_info=True)
        return df, None # Return original df on error

    return df, encoder

# --- FUNCTION TO PROVIDE DESCRIPTIONS --- #

def get_available_functions_description() -> str:
    """Generates a description string of available data preparation functions for the LLM Planner."""
    descriptions = []
    # Get all functions defined in this module
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            # Exclude this helper function itself
            if name == 'get_available_functions_description':
                continue
            # Try to get signature and docstring
            try:
                sig = inspect.signature(obj)
                doc = inspect.getdoc(obj)
                # Basic formatting: function name, parameters, first line of docstring
                first_line_doc = doc.split('\n')[0] if doc else "No description available."
                descriptions.append(f"- `{name}{sig}`: {first_line_doc}")
            except Exception as e:
                logger.warning(f"Could not inspect function {name}: {e}")
                descriptions.append(f"- `{name}`: Error retrieving details.")

    if not descriptions:
        return "No data preparation functions found."

    return "\nAvailable Data Preparation Functions:\n" + "\n".join(descriptions) 
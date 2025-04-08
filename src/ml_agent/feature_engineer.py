# src/ml_agent/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.exceptions import NotFittedError
import warnings

def encode_categorical(df: pd.DataFrame, column: str, strategy: str = 'one-hot', drop_original: bool = True, drop_first: bool = False, **kwargs) -> tuple[pd.DataFrame, object | None]:
    """Encodes a categorical column using the specified strategy."""
    df = df.copy()
    encoder = None
    original_dtype = df[column].dtype

    if strategy == 'one-hot':
        # Ensure column is treated as categorical for OHE
        df[column] = df[column].astype('category')
        # Handle potential unknown categories during transform by ignoring them
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first' if drop_first else None)
        encoded_data = encoder.fit_transform(df[[column]])
        feature_names = encoder.get_feature_names_out([column])
        encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)
        df = pd.concat([df, encoded_df], axis=1)
        if drop_original:
            df = df.drop(columns=[column])

    elif strategy == 'label':
        encoder = LabelEncoder()
        # Handle potential NaNs - LabelEncoder doesn't like them directly
        mask = df[column].notna()
        df.loc[mask, column + '_encoded'] = encoder.fit_transform(df.loc[mask, column])
        df[column + '_encoded'] = df[column + '_encoded'].astype('float') # Keep as float to represent potential NaNs
        if drop_original:
            df = df.drop(columns=[column])
        else:
             # Restore original dtype if not dropping, but keep encoded separate
             df[column] = df[column].astype(original_dtype)


    # TODO: Add other strategies like 'target', 'frequency', 'binary' etc.
    elif strategy == 'frequency':
         # Calculate frequency map
        freq_map = df[column].value_counts(normalize=True).to_dict()
        # Map frequencies
        df[column + '_freq_encoded'] = df[column].map(freq_map).fillna(0) # Fill NaN freq with 0
        encoder = freq_map # Store map as encoder for potential inverse (though less common)
        if drop_original:
            df = df.drop(columns=[column])

    else:
        warnings.warn(f"Encoding strategy '{strategy}' not implemented. Skipping column '{column}'.")
        return df, None

    print(f"Encoded column '{column}' using strategy '{strategy}'. Dropped original: {drop_original}")
    return df, encoder

def scale_numerical(df: pd.DataFrame, columns: list[str], strategy: str = 'standard', **kwargs) -> tuple[pd.DataFrame, dict]:
    """Scales numerical columns using the specified strategy."""
    df = df.copy()
    scalers = {}
    if not isinstance(columns, list):
        columns = [columns]

    valid_columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_columns:
         warnings.warn(f"No valid numerical columns found in the list provided for scaling: {columns}. Skipping.")
         return df, scalers

    if strategy == 'standard':
        scaler_instance = StandardScaler(**kwargs)
    elif strategy == 'minmax':
        scaler_instance = MinMaxScaler(**kwargs)
    elif strategy == 'robust':
        scaler_instance = RobustScaler(**kwargs)
    else:
        warnings.warn(f"Scaling strategy '{strategy}' not implemented. Skipping scaling for columns: {valid_columns}.")
        return df, scalers

    try:
        df[valid_columns] = scaler_instance.fit_transform(df[valid_columns])
        scalers[strategy] = scaler_instance # Store one scaler for all columns processed together
        print(f"Scaled columns {valid_columns} using strategy '{strategy}'.")
    except Exception as e:
        warnings.warn(f"Could not scale columns {valid_columns} using strategy '{strategy}': {e}")


    return df, scalers


def handle_outliers(df: pd.DataFrame, columns: list[str], strategy: str = 'clip_iqr', iqr_multiplier: float = 1.5, std_dev_multiplier: float = 3.0, **kwargs) -> pd.DataFrame:
    """Handles outliers in numerical columns using the specified strategy."""
    df = df.copy()
    if not isinstance(columns, list):
        columns = [columns]

    for column in columns:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            warnings.warn(f"Column '{column}' not found or not numeric. Skipping outlier handling.")
            continue

        data_col = df[column]
        lower_bound = None
        upper_bound = None

        if strategy == 'clip_iqr':
            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            print(f"Handled outliers in '{column}' using 'clip_iqr' (Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]).")

        elif strategy == 'clip_std':
            mean = data_col.mean()
            std = data_col.std()
            lower_bound = mean - std_dev_multiplier * std
            upper_bound = mean + std_dev_multiplier * std
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            print(f"Handled outliers in '{column}' using 'clip_std' (Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]).")

        elif strategy == 'remove':
             Q1 = data_col.quantile(0.25)
             Q3 = data_col.quantile(0.75)
             IQR = Q3 - Q1
             lower_bound = Q1 - iqr_multiplier * IQR
             upper_bound = Q3 + iqr_multiplier * IQR
             original_rows = len(df)
             df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
             removed_rows = original_rows - len(df)
             print(f"Removed {removed_rows} rows based on outliers in '{column}' using 'remove_iqr' (Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]).")
             # Note: Removing rows can affect index and other columns, use cautiously.

        # TODO: Add 'winsorize' etc.
        else:
            warnings.warn(f"Outlier handling strategy '{strategy}' not implemented for column '{column}'. Skipping.")

    return df


def handle_high_cardinality(df: pd.DataFrame, column: str, threshold: float = 0.95, method: str = 'frequency', replace_with: str = 'Other', top_n: int = 20, **kwargs) -> tuple[pd.DataFrame, object | None]:
    """Handles high cardinality categorical columns."""
    df = df.copy()
    mapping = None

    if column not in df.columns or pd.api.types.is_numeric_dtype(df[column]):
         warnings.warn(f"Column '{column}' not found or is numeric. Skipping high cardinality handling.")
         return df, mapping

    cardinality = df[column].nunique()
    print(f"Column '{column}' has cardinality {cardinality}.")

    # Determine if handling is needed based on some logic (e.g., > N unique values)
    # For now, we assume the planner decided it's needed.

    if method == 'frequency':
        counts = df[column].value_counts(normalize=True)
        cumulative_freq = counts.cumsum()
        # Categories to keep: either cover 'threshold' frequency or top_n, whichever is smaller set but covers some minimum
        categories_to_keep = cumulative_freq[cumulative_freq <= threshold].index.tolist()
        if len(categories_to_keep) < min(top_n, cardinality): # Ensure we keep at least some if threshold is too strict
             categories_to_keep = df[column].value_counts().nlargest(min(top_n, cardinality)).index.tolist()

        if len(categories_to_keep) < cardinality:
            # Create mapping: keep top categories, map others to 'replace_with'
            mapping = {cat: cat for cat in categories_to_keep}
            # Apply mapping: replace categories not in the keep list
            df[column + '_handled'] = df[column].apply(lambda x: mapping.get(x, replace_with))
            df = df.drop(columns=[column]) # Replace original column
            print(f"Handled high cardinality in '{column}' using 'frequency'. Kept {len(categories_to_keep)} categories, replaced others with '{replace_with}'.")
        else:
             print(f"Cardinality of '{column}' ({cardinality}) is low or threshold keeps all. No changes made.")
             df[column + '_handled'] = df[column] # Still create new column for consistency maybe? Or just return df
             df = df.drop(columns=[column])
             mapping = {cat: cat for cat in df[column + '_handled'].unique()} # All categories kept

    # TODO: Add other methods like 'grouping' (needs target), 'target_encoding'
    else:
        warnings.warn(f"High cardinality handling method '{strategy}' not implemented for column '{column}'. Skipping.")

    return df, mapping

def create_interaction_features(df: pd.DataFrame, columns1: list[str], columns2: list[str] | None = None, interaction_type: str = 'multiply', **kwargs) -> pd.DataFrame:
    """Creates interaction features between specified columns."""
    df = df.copy()

    if not isinstance(columns1, list): columns1 = [columns1]
    if columns2 and not isinstance(columns2, list): columns2 = [columns2]

    valid_cols1 = [col for col in columns1 if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    # If columns2 is provided, interact columns1 with columns2
    if columns2:
        valid_cols2 = [col for col in columns2 if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if not valid_cols1 or not valid_cols2:
             warnings.warn("Interaction requires valid numeric columns from both lists. Skipping.")
             return df
        for col1 in valid_cols1:
            for col2 in valid_cols2:
                 if col1 == col2: continue # Avoid self-interaction if lists overlap
                 new_col_name = f"{col1}_{interaction_type}_{col2}"
                 try:
                     if interaction_type == 'multiply':
                         df[new_col_name] = df[col1] * df[col2]
                     elif interaction_type == 'divide':
                          # Avoid division by zero, replace inf with NaN or large number?
                          df[new_col_name] = (df[col1] / df[col2].replace(0, np.nan)).fillna(0) # Example handling
                     # Add more interaction types ('add', 'subtract', 'polynomial'?)
                     else:
                         warnings.warn(f"Interaction type '{interaction_type}' not recognized. Skipping {col1} and {col2}.")
                         continue
                     print(f"Created interaction feature: '{new_col_name}'")
                 except Exception as e:
                     warnings.warn(f"Could not create interaction '{new_col_name}': {e}")

    # If columns2 is None, create pairwise interactions within columns1
    elif len(valid_cols1) > 1:
         from itertools import combinations
         for col1, col2 in combinations(valid_cols1, 2):
              new_col_name = f"{col1}_{interaction_type}_{col2}"
              try:
                  if interaction_type == 'multiply':
                      df[new_col_name] = df[col1] * df[col2]
                  elif interaction_type == 'divide':
                       df[f"{col1}_div_{col2}"] = (df[col1] / df[col2].replace(0, np.nan)).fillna(0)
                       df[f"{col2}_div_{col1}"] = (df[col2] / df[col1].replace(0, np.nan)).fillna(0)
                       new_col_name = f"{col1}_div_{col2} and vice versa" # Adjust naming
                  # Add more interaction types
                  else:
                      warnings.warn(f"Interaction type '{interaction_type}' not recognized. Skipping {col1} and {col2}.")
                      continue
                  print(f"Created interaction feature: '{new_col_name}'")
              except Exception as e:
                  warnings.warn(f"Could not create interaction '{new_col_name}': {e}")
    else:
         warnings.warn("Interaction requires at least two valid numeric columns. Skipping.")


    return df

# Potential main function for testing or direct call if needed
# if __name__ == '__main__':
#     # Example Usage
#     data = {'cat': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'D', 'D', 'D', 'D'],
#             'cat_high': [f'ID_{i}' for i in range(11)],
#             'num1': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10, 11],
#             'num2': [10, 20, 15, 25, 30, 35, 40, -50, 50, 55, 60]}
#     test_df = pd.DataFrame(data)
#
#     # Encoding
#     test_df, _ = encode_categorical(test_df, 'cat', strategy='one-hot')
#     test_df, _ = encode_categorical(test_df, 'cat', strategy='label', drop_original=False) # Keep original too
#     test_df, _ = encode_categorical(test_df, 'cat', strategy='frequency')
#     print("\nAfter Encoding:\n", test_df.head())
#
#     # Outliers
#     test_df_outlier = handle_outliers(test_df.copy(), ['num1', 'num2'], strategy='clip_iqr')
#     print("\nAfter Outlier Handling (IQR):\n", test_df_outlier[['num1', 'num2']].describe())
#     test_df_outlier_std = handle_outliers(test_df.copy(), ['num1', 'num2'], strategy='clip_std')
#     print("\nAfter Outlier Handling (STD):\n", test_df_outlier_std[['num1', 'num2']].describe())
#
#
#     # High Cardinality
#     test_df, _ = handle_high_cardinality(test_df, 'cat_high', threshold=0.8, top_n=3)
#     print("\nAfter High Cardinality Handling:\n", test_df.head())
#
#     # Scaling
#     test_df, _ = scale_numerical(test_df, ['num1', 'num2'], strategy='standard')
#     print("\nAfter Scaling:\n", test_df[['num1', 'num2']].head())
#
#     # Interaction
#     test_df = create_interaction_features(test_df, ['num1', 'num2'])
#     print("\nAfter Interaction:\n", test_df.head()) 
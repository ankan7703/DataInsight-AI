
import pandas as pd


# ---------------------------------------
# 1. Validate Dataset
# ---------------------------------------

def validate_dataframe(df):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    return True


# ---------------------------------------
# 2. Drop Rows with Missing Values
# ---------------------------------------

def drop_missing_rows(df, columns=None):

    validate_dataframe(df)

    if columns is not None:

        invalid_columns = [
            col for col in columns
            if col not in df.columns
        ]

        if invalid_columns:
            raise ValueError(
                f"Columns not found: {invalid_columns}"
            )

        cleaned_df = df.dropna(subset=columns)

    else:

        cleaned_df = df.dropna()

    return cleaned_df.copy()


# ---------------------------------------
# 3. Mean Imputation
# ---------------------------------------

def fill_missing_with_mean(df, columns):

    validate_dataframe(df)

    cleaned_df = df.copy()

    for column in columns:

        if column not in cleaned_df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if not pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):
            raise TypeError(
                f"Mean imputation requires numeric column: {column}"
            )

        if cleaned_df[column].notna().sum() == 0:
            raise ValueError(
                f"Column '{column}' contains no valid values."
            )

        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].mean()
        )

    return cleaned_df


# ---------------------------------------
# 4. Median Imputation
# ---------------------------------------

def fill_missing_with_median(df, columns):

    validate_dataframe(df)

    cleaned_df = df.copy()

    for column in columns:

        if column not in cleaned_df.columns:
            raise ValueError(f"Column '{column}' not found.")

        if not pd.api.types.is_numeric_dtype(
            cleaned_df[column]
        ):
            raise TypeError(
                f"Median imputation requires numeric column: {column}"
            )

        if cleaned_df[column].notna().sum() == 0:
            raise ValueError(
                f"Column '{column}' contains no valid values."
            )

        cleaned_df[column] = cleaned_df[column].fillna(
            cleaned_df[column].median()
        )

    return cleaned_df


# ---------------------------------------
# 5. Mode Imputation
# ---------------------------------------

def fill_missing_with_mode(df, columns):

    validate_dataframe(df)

    cleaned_df = df.copy()

    for column in columns:

        if column not in cleaned_df.columns:
            raise ValueError(f"Column '{column}' not found.")

        modes = cleaned_df[column].mode(dropna=True)

        if modes.empty:
            raise ValueError(
                f"Column '{column}' contains no valid values."
            )

        cleaned_df[column] = cleaned_df[column].fillna(
            modes.iloc[0]
        )

    return cleaned_df


# ---------------------------------------
# 6. Missing Value Cleaning Summary
# ---------------------------------------

def get_cleaning_summary(original_df, cleaned_df):

    validate_dataframe(original_df)
    validate_dataframe(cleaned_df)

    original_missing = int(
        original_df.isnull().sum().sum()
    )

    remaining_missing = int(
        cleaned_df.isnull().sum().sum()
    )

    return {
        "original_rows": len(original_df),
        "cleaned_rows": len(cleaned_df),
        "original_missing_values": original_missing,
        "remaining_missing_values": remaining_missing,
        "missing_values_resolved": (
            original_missing - remaining_missing
        ),
        "rows_removed": (
            len(original_df) - len(cleaned_df)
        )
    }

import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """Load a CSV or Excel file into a DataFrame."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()
    try:
        if extension == ".csv":
            df = pd.read_csv(file_path)
        elif extension in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Supported tabular formats are CSV and Excel.")
    except pd.errors.EmptyDataError as error:
        raise ValueError("The uploaded file is empty.") from error

    if df.empty:
        raise ValueError("The uploaded dataset contains no rows.")
    if len(df.columns) == 0:
        raise ValueError("The uploaded dataset contains no columns.")
    if not df.columns.is_unique:
        raise ValueError("The dataset contains duplicate column names. Rename them before analysis.")
    return df


def _column_profile(df, column):
    series = df[column]
    dtype = str(series.dtype)
    non_null = series.dropna()
    profile = {
        "name": str(column),
        "dtype": dtype,
        "non_null_count": int(series.notna().sum()),
        "missing_count": int(series.isna().sum()),
        "missing_percentage": round(float(series.isna().mean() * 100), 2),
        "unique_count": int(series.nunique(dropna=True)),
        "examples": [str(value)[:120] for value in non_null.head(5).tolist()],
    }

    if pd.api.types.is_numeric_dtype(series):
        profile["inferred_type"] = "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        profile["inferred_type"] = "datetime"
    elif pd.api.types.is_bool_dtype(series):
        profile["inferred_type"] = "boolean"
    else:
        profile["inferred_type"] = (
            "categorical" if series.nunique(dropna=True) <= min(50, max(10, len(df) // 5))
            else "text_or_identifier"
        )
    return profile


def get_dataset_overview(df):
    """Return a schema-neutral overview of the supplied table."""
    return {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
        "data_types": {str(k): str(v) for k, v in df.dtypes.items()},
        "column_profiles": [_column_profile(df, column) for column in df.columns],
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
        "preview": df.head(5).astype(object).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
    }


def get_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / max(len(df), 1)) * 100
    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
    })
    return report[report["missing_count"] > 0].sort_values(
        by="missing_count", ascending=False
    )


def get_duplicate_records(df):
    duplicate_count = int(df.duplicated().sum())
    duplicate_rows = df[df.duplicated(keep=False)]
    return {
        "total_duplicate_rows": duplicate_count,
        "duplicate_records": duplicate_rows.astype(object).where(
            pd.notna(duplicate_rows), None
        ).to_dict(orient="records"),
    }


def get_statistical_summary(df):
    """Summarize available data types without failing on empty type groups."""
    numerical_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(include=["object", "category", "bool", "str"])
    datetime_df = df.select_dtypes(include=["datetime", "datetimetz"])

    numerical_summary = (
        numerical_df.describe().round(2)
        if not numerical_df.empty
        else pd.DataFrame()
    )
    categorical_summary = (
        categorical_df.describe()
        if not categorical_df.empty
        else pd.DataFrame()
    )
    datetime_summary = (
        datetime_df.describe()
        if not datetime_df.empty
        else pd.DataFrame()
    )
    return {
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,
        "datetime_summary": datetime_summary,
    }


def get_outlier_summary(df, exclude_columns=None):
    """IQR outlier screening; caller may explicitly exclude unsuitable fields."""
    numerical_df = df.select_dtypes(include="number").drop(
        columns=exclude_columns or [], errors="ignore"
    )
    report = {}
    for column in numerical_df.columns:
        values = numerical_df[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = values[(values < lower) | (values > upper)]
        report[str(column)] = {
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(len(outliers)),
            "outlier_values": outliers.head(100).tolist(),
        }
    return report


def generate_data_profile(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    return {
        "dataset_overview": get_dataset_overview(df),
        "missing_values": get_missing_values(df),
        "duplicate_records": get_duplicate_records(df),
        "statistical_summary": get_statistical_summary(df),
        "outlier_summary": get_outlier_summary(df),
    }

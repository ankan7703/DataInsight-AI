
import pandas as pd
from pathlib import Path


def load_dataset(file_path):
    """
    Load a CSV or Excel dataset.

    Parameters:
        file_path (str): Path to the dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    try:
        if extension == ".csv":
            df = pd.read_csv(file_path)

        elif extension in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)

        else:
            raise ValueError(
                "Unsupported file format. Please upload CSV or Excel files."
            )

    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV/Excel file is empty.")

    if df.empty:
        raise ValueError("The uploaded dataset contains no data.")

    return df


def get_dataset_overview(df):
    """
    Generate a basic overview of the dataset.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Dataset overview.
    """

    overview = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "memory_usage_kb": round(
            df.memory_usage(deep=True).sum() / 1024, 2
        ),
        "preview": df.head(5).to_dict(orient="records")
    }

    return overview

def get_missing_values(df):

    missing_count = df.isnull().sum()

    missing_percentage = (
        missing_count / len(df)
    ) * 100

    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    missing_report = missing_report[
        missing_report["missing_count"] > 0
    ]

    return missing_report.sort_values(
        by="missing_count",
        ascending=False
    )


def get_duplicate_records(df):

    duplicate_count = df.duplicated().sum()

    duplicate_rows = df[df.duplicated(keep=False)]

    duplicate_report = {
        "total_duplicate_rows": int(duplicate_count),
        "duplicate_records": duplicate_rows.to_dict(
            orient="records"
        )
    }

    return duplicate_report


def get_statistical_summary(df):

    numerical_df = df.select_dtypes(include="number")

    categorical_df = df.select_dtypes(
        include=["object", "category", "bool"]
    )

    numerical_summary = numerical_df.describe().round(2)

    categorical_summary = categorical_df.describe()

    return {
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary
    }


def get_outlier_summary(df, exclude_columns=None):

    if exclude_columns is None:
        exclude_columns = []

    numerical_df = df.select_dtypes(include="number")

    numerical_df = numerical_df.drop(
        columns=exclude_columns,
        errors="ignore"
    )

    outlier_report = {}

    for column in numerical_df.columns:

        Q1 = numerical_df[column].quantile(0.25)
        Q3 = numerical_df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)

        outliers = numerical_df[
            (numerical_df[column] < lower_bound) |
            (numerical_df[column] > upper_bound)
        ][column]

        outlier_report[column] = {
            "lower_bound": float(round(lower_bound, 2)),
            "upper_bound": float(round(upper_bound, 2)),
            "outlier_count": int(len(outliers)),
            "outlier_values": outliers.tolist()
        }

    return outlier_report


def generate_data_profile(df):

    overview = get_dataset_overview(df)

    missing_values = get_missing_values(df)

    duplicate_records = get_duplicate_records(df)

    statistical_summary = get_statistical_summary(df)

    outlier_summary = get_outlier_summary(
        df,
        exclude_columns=["customer_id"]
    )

    complete_profile = {
        "dataset_overview": overview,
        "missing_values": missing_values,
        "duplicate_records": duplicate_records,
        "statistical_summary": statistical_summary,
        "outlier_summary": outlier_summary
    }

    return complete_profile

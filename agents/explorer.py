import pandas as pd

from tools.data_profiler import generate_data_profile


def explore_dataset(df):
    """Build a schema-neutral profile for a tabular dataset."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    profile = generate_data_profile(df)
    missing_values = (
        profile["missing_values"]
        .rename_axis("column")
        .reset_index()
        .to_dict(orient="records")
    )
    statistical_summary = {
        key: value.to_dict(orient="index")
        for key, value in profile["statistical_summary"].items()
    }
    return {
        "agent": "Explorer Agent",
        "status": "success",
        "dataset_overview": profile["dataset_overview"],
        "column_profiles": profile["dataset_overview"]["column_profiles"],
        "missing_values": missing_values,
        "duplicate_records": profile["duplicate_records"],
        "statistical_summary": statistical_summary,
        "outlier_summary": profile["outlier_summary"],
    }


import pandas as pd

from tools.data_profiler import generate_data_profile


# FUNCTION 1: EXPLORE DATASET

def explore_dataset(df):
    """
    Analyze a dataset and generate a structured
    exploration report.

    Parameters:
        df (pd.DataFrame): Input dataset.

    Returns:
        dict: Structured dataset exploration report.
    """

    # Validate the input
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    # Generate the complete data profile
    profile = generate_data_profile(df)

    # Convert missing-value DataFrame into records
    missing_values = (
        profile["missing_values"]
        .rename_axis("column")
        .reset_index()
        .to_dict(orient="records")
    )

    # Convert statistical DataFrames into dictionaries
    statistical_summary = {
        key: value.to_dict(orient="index")
        for key, value in profile["statistical_summary"].items()
    }

    # Prepare the final exploration report
    exploration_report = {

        "agent": "Explorer Agent",

        "status": "success",

        "dataset_overview": profile["dataset_overview"],

        "missing_values": missing_values,

        "duplicate_records": profile["duplicate_records"],

        "statistical_summary": statistical_summary,

        "outlier_summary": profile["outlier_summary"]
    }

    return exploration_report

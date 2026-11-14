import pandas as pd
import pytest

from agents.analyst import execute_analysis_plan
from agents.explorer import explore_dataset


@pytest.fixture
def unrelated_dataset():
    return pd.DataFrame({
        "Department Name": ["Physics", "Chemistry", "Physics", "Math"],
        "Final Score": [91, 84, 78, 96],
        "Enrollment Date": pd.to_datetime([
            "2025-01-01", "2025-01-03", "2025-01-04", "2025-01-05"
        ]),
        "Scholarship": [True, False, True, True],
    })


def test_explorer_profiles_arbitrary_columns(unrelated_dataset):
    report = explore_dataset(unrelated_dataset)
    assert report["dataset_overview"]["total_rows"] == 4
    names = [item["name"] for item in report["column_profiles"]]
    assert names == ["Department Name", "Final Score", "Enrollment Date", "Scholarship"]


def test_group_by_uses_dynamic_column_names(unrelated_dataset):
    plan = {
        "tasks": [{
            "operation": "group_by",
            "group_by": ["Department Name"],
            "metric": "Final Score",
            "aggregation": "mean",
            "limit": 10,
        }]
    }
    result = execute_analysis_plan(unrelated_dataset, plan)
    assert result["status"] == "success"
    rows = result["results"][0]["result"]
    assert rows[0]["Department Name"] == "Math"
    assert rows[0]["result"] == 96.0


def test_unknown_column_fails_safely(unrelated_dataset):
    plan = {
        "tasks": [{
            "operation": "value_counts",
            "columns": ["not_a_real_column"],
            "limit": 10,
        }]
    }
    result = execute_analysis_plan(unrelated_dataset, plan)
    assert result["status"] == "failed"
    assert "Unknown column" in result["errors"][0]["error"]


def test_value_counts_and_aggregate(unrelated_dataset):
    plan = {
        "tasks": [
            {"operation": "value_counts", "columns": ["Department Name"], "limit": 10},
            {"operation": "aggregate", "metric": "Final Score", "aggregation": "mean"},
        ]
    }
    result = execute_analysis_plan(unrelated_dataset, plan)
    assert result["status"] == "success"
    assert result["results"][1]["result"]["value"] == 87.25

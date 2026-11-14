import pandas as pd

from agents.planner import create_rule_based_analysis_plan
from agents.analyst import execute_analysis_plan
from agents.insight_writer import generate_direct_answer


def test_product_highest_price_without_gemini():
    df = pd.DataFrame({
        "Product_Name": ["Notebook", "Pen", "Bag"],
        "Category": ["Stationery", "Stationery", "Accessories"],
        "Price": [45, 10, 850],
    })
    profile = {
        "dataset_overview": {
            "column_names": list(df.columns),
            "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        },
        "column_profiles": [
            {"name": "Product_Name", "inferred_type": "categorical"},
            {"name": "Category", "inferred_type": "categorical"},
            {"name": "Price", "inferred_type": "numeric"},
        ],
    }

    plan = create_rule_based_analysis_plan(
        "product which has highest price",
        profile,
    )
    result = execute_analysis_plan(df, plan)
    answer = generate_direct_answer(
        "product which has highest price",
        result,
    )

    assert result["status"] == "success"
    assert result["results"][0]["result"][0]["Product_Name"] == "Bag"
    assert "Bag" in answer
    assert "850" in answer


def test_lowest_metric_is_sorted_ascending():
    df = pd.DataFrame({
        "Product_Name": ["Notebook", "Pen", "Bag"],
        "Price": [45, 10, 850],
    })
    profile = {
        "dataset_overview": {
            "column_names": list(df.columns),
            "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        },
        "column_profiles": [
            {"name": "Product_Name", "inferred_type": "categorical"},
            {"name": "Price", "inferred_type": "numeric"},
        ],
    }

    plan = create_rule_based_analysis_plan("product with lowest price", profile)
    result = execute_analysis_plan(df, plan)

    assert result["results"][0]["result"][0]["Product_Name"] == "Pen"

import pandas as pd

from agents.planner import create_analysis_plan
from agents.analyst import execute_analysis_plan
from agents.insight_writer import generate_direct_answer


def make_profile(df):
    return {
        "dataset_overview": {
            "column_names": list(df.columns),
            "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        },
        "column_profiles": [
            {
                "name": column,
                "inferred_type": "numeric" if pd.api.types.is_numeric_dtype(df[column]) else "categorical",
            }
            for column in df.columns
        ],
    }


def test_lowest_product_price_uses_fast_path(monkeypatch):
    df = pd.DataFrame({"Product_Name": ["Laptop", "Mouse", "Keyboard"], "Price": [1000, 20, 50]})
    monkeypatch.setattr("agents.planner.get_llm", lambda: (_ for _ in ()).throw(AssertionError("Gemini called")))
    plan = create_analysis_plan("Which product has the lowest price?", make_profile(df))
    result = execute_analysis_plan(df, plan)
    answer = generate_direct_answer("Which product has the lowest price?", result)
    assert plan["planner_mode"] == "fast_rule_based"
    assert result["results"][0]["result"][0]["Product_Name"] == "Mouse"
    assert "Mouse" in answer and "20" in answer


def test_highest_average_by_category_uses_fast_path(monkeypatch):
    df = pd.DataFrame({"Category": [" A ", "A", "B", "B"], "Price": [10, 30, 20, 40]})
    monkeypatch.setattr("agents.planner.get_llm", lambda: (_ for _ in ()).throw(AssertionError("Gemini called")))
    question = "Which category has the highest average price?"
    plan = create_analysis_plan(question, make_profile(df))
    result = execute_analysis_plan(df, plan)
    answer = generate_direct_answer(question, result)
    assert plan["planner_mode"] == "fast_rule_based"
    assert result["results"][0]["result"][0]["Category"] == "B"
    assert result["results"][0]["result"][0]["result"] == 30
    assert "highest average" in answer.lower()


def test_total_quantity_uses_fast_path(monkeypatch):
    df = pd.DataFrame({"Quantity_Available": [2, 5, 3]})
    monkeypatch.setattr("agents.planner.get_llm", lambda: (_ for _ in ()).throw(AssertionError("Gemini called")))
    plan = create_analysis_plan("What is the total quantity available?", make_profile(df))
    result = execute_analysis_plan(df, plan)
    assert plan["planner_mode"] == "fast_rule_based"
    assert result["results"][0]["result"]["value"] == 10

def test_count_rows_matching_categorical_value():
    df = pd.DataFrame({
        "Current Degree Branch": ["IT", " IT ", "CSE", "Information Technology", None],
    })
    profile = make_profile(df)
    question = "Count the number of rows where Current Degree Branch is IT."
    plan = create_analysis_plan(question, profile)
    result = execute_analysis_plan(df, plan)
    answer = generate_direct_answer(question, result)

    assert plan["tasks"][0]["operation"] == "filter_count"
    assert result["status"] == "success"
    assert result["results"][0]["result"]["count"] == 2
    assert "2" in answer


def test_count_each_value_in_branch_column():
    df = pd.DataFrame({
        "Current Degree Branch": ["IT", " IT ", "CSE", "CSE", "Information Technology"],
    })
    profile = make_profile(df)
    question = "How many students belong to each branch?"
    plan = create_analysis_plan(question, profile)
    result = execute_analysis_plan(df, plan)
    answer = generate_direct_answer(question, result)

    assert plan["tasks"][0]["operation"] == "value_counts"
    counts = result["results"][0]["result"]["Current Degree Branch"]
    assert counts == [
        {"value": "CSE", "count": 2},
        {"value": "IT", "count": 2},
        {"value": "Information Technology", "count": 1},
    ]
    assert "CSE" in answer and "Information Technology" in answer

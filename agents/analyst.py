
import pandas as pd


def calculate_purchase_statistics(df):
    """Calculate descriptive statistics for purchase amounts."""

    if "purchase_amount" not in df.columns:
        raise ValueError("Column 'purchase_amount' not found.")

    purchases = df["purchase_amount"].dropna()

    if purchases.empty:
        raise ValueError("No valid purchase amount data available.")

    return {
        "total_purchase_amount": float(purchases.sum()),
        "average_purchase_amount": float(purchases.mean()),
        "median_purchase_amount": float(purchases.median()),
        "minimum_purchase_amount": float(purchases.min()),
        "maximum_purchase_amount": float(purchases.max()),
        "number_of_transactions": int(purchases.count())
    }


def group_by_category(df):
    """Calculate average purchase amount for each product category."""

    required_columns = ["product_category", "purchase_amount"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column '{column}' not found.")

    result = (
        df.groupby("product_category")["purchase_amount"]
        .agg(["count", "mean", "sum"])
        .reset_index()
    )

    result = result.rename(columns={
        "count": "transaction_count",
        "mean": "average_purchase_amount",
        "sum": "total_purchase_amount"
    })

    result = result.sort_values(
        by="average_purchase_amount",
        ascending=False
    )

    return result.round(2).to_dict(orient="records")


def analyze_age_distribution(df):
    """Analyze customer age distribution."""

    if "age" not in df.columns:
        raise ValueError("Column 'age' not found.")

    ages = df["age"].dropna()

    if ages.empty:
        raise ValueError("No valid age data available.")

    age_groups = pd.cut(
        ages,
        bins=[0, 25, 35, 45, 100],
        labels=["18-25", "26-35", "36-45", "46+"],
        include_lowest=True
    )

    return {
        "average_age": float(ages.mean()),
        "median_age": float(ages.median()),
        "minimum_age": int(ages.min()),
        "maximum_age": int(ages.max()),
        "age_group_distribution": {
            str(key): int(value)
            for key, value in age_groups.value_counts().items()
        }
    }


def group_by_city(df):
    """Count customers or records for each city."""

    if "city" not in df.columns:
        raise ValueError("Column 'city' not found.")

    result = (
        df["city"]
        .value_counts()
        .reset_index()
    )

    result.columns = ["city", "record_count"]

    return result.to_dict(orient="records")


def general_exploration(df):
    """Generate a basic numerical summary of the dataset."""

    numeric_summary = df.describe(
        include="number"
    ).round(2)

    return numeric_summary.to_dict()



def execute_analysis_plan(df, analysis_plan):
    """
    Execute predefined operations from the Planner Agent.

    Successful tasks are preserved even if other tasks fail.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not isinstance(analysis_plan, dict):
        raise TypeError("Analysis plan must be a dictionary.")

    tasks = analysis_plan.get("tasks", [])

    if not isinstance(tasks, list):
        raise ValueError("Analysis plan tasks must be a list.")

    operation_handlers = {
        "calculate_purchase_statistics": calculate_purchase_statistics,
        "group_by_category": group_by_category,
        "analyze_age_distribution": analyze_age_distribution,
        "group_by_city": group_by_city,
        "general_exploration": general_exploration
    }

    results = []
    errors = []

    # Handle empty analysis plans
    if not tasks:
        errors.append({
            "operation": None,
            "error": "No analysis tasks were provided by the Planner Agent."
        })

    for index, task in enumerate(tasks):

        # Validate task structure
        if not isinstance(task, dict):
            errors.append({
                "task_index": index,
                "operation": None,
                "error": "Invalid task format. Expected a dictionary."
            })
            continue

        operation = task.get("operation")

        # Validate requested operation
        if operation not in operation_handlers:
            errors.append({
                "task_index": index,
                "operation": operation,
                "error": "Unsupported analysis operation."
            })
            continue

        try:
            output = operation_handlers[operation](df)

            results.append({
                "operation": operation,
                "status": "success",
                "result": output
            })

        except Exception as error:
            errors.append({
                "task_index": index,
                "operation": operation,
                "error": str(error)
            })

    # Determine overall execution status
    if not results:
        status = "failed"

    elif errors:
        status = "completed_with_errors"

    else:
        status = "success"

    return {
        "agent": "Analyst Agent",
        "status": status,
        "total_tasks": len(tasks),
        "total_executed": len(results),
        "total_errors": len(errors),
        "results": results,
        "errors": errors
    }

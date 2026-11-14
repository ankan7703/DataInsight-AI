"""Generic, validated Pandas operations for arbitrary tabular data."""

import pandas as pd


SUPPORTED_AGGREGATIONS = {
    "sum": "sum", "mean": "mean", "median": "median",
    "min": "min", "max": "max", "count": "count",
    "nunique": "nunique", "std": "std",
}


def _require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {missing}")


def _json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _records(frame):
    return [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def dataset_overview(df, task):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns],
        "data_types": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def column_summary(df, task):
    columns = task.get("columns", [])
    _require_columns(df, columns)
    output = {}
    for column in columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe().to_dict()
            output[column] = {str(k): _json_safe(v) for k, v in desc.items()}
        else:
            output[column] = {
                "non_null_count": int(series.notna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "most_common": [
                    {"value": _json_safe(k), "count": int(v)}
                    for k, v in series.value_counts(dropna=False).head(10).items()
                ],
            }
    return output


def value_counts(df, task):
    columns = task.get("columns", [])
    _require_columns(df, columns)
    result = {}
    for column in columns:
        counts = df[column].value_counts(dropna=False).head(task.get("limit", 10))
        result[column] = [
            {"value": _json_safe(index), "count": int(count)}
            for index, count in counts.items()
        ]
    return result


def aggregate(df, task):
    metric = task.get("metric")
    _require_columns(df, [metric])
    method = task.get("aggregation")
    if method not in SUPPORTED_AGGREGATIONS:
        raise ValueError("Unsupported aggregation.")
    series = df[metric]
    if method not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{method} requires a numeric column: {metric}")
    value = getattr(series, method)()
    return {"column": metric, "aggregation": method, "value": _json_safe(value)}


def group_by(df, task):
    groups = task.get("group_by", [])
    metric = task.get("metric")
    _require_columns(df, groups + [metric])
    method = task.get("aggregation")
    if method not in SUPPORTED_AGGREGATIONS:
        raise ValueError("Unsupported aggregation.")
    if method not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"{method} requires a numeric metric: {metric}")
    grouped = df.groupby(groups, dropna=False)[metric].agg(method).reset_index(name="result")
    grouped = grouped.sort_values("result", ascending=False, na_position="last")
    return _records(grouped.head(task.get("limit", 10)))


def correlation(df, task):
    columns = task.get("columns", [])
    _require_columns(df, columns)
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Correlation requires numeric columns: {non_numeric}")
    return df[columns].corr().round(4).to_dict()


def distribution(df, task):
    columns = task.get("columns", [])
    _require_columns(df, columns)
    result = {}
    for column in columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Distribution requires a numeric column: {column}")
        series = df[column].dropna()
        if series.empty:
            result[column] = {"count": 0, "bins": []}
            continue
        counts, edges = __import__("numpy").histogram(series, bins="auto")
        result[column] = {
            "count": int(series.size),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "bins": [
                {"from": float(edges[i]), "to": float(edges[i + 1]), "count": int(counts[i])}
                for i in range(len(counts))
            ],
        }
    return result


def top_n(df, task):
    metric = task.get("metric")
    _require_columns(df, [metric])
    method = task.get("aggregation")
    if method not in SUPPORTED_AGGREGATIONS:
        raise ValueError("Unsupported aggregation.")
    if method not in {"count", "nunique"} and not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(f"{method} requires a numeric column: {metric}")
    ascending = task.get("description") == "minimum"
    frame = df.sort_values(metric, ascending=ascending, na_position="last").head(task.get("limit", 10))
    return _records(frame)


HANDLERS = {
    "dataset_overview": dataset_overview,
    "column_summary": column_summary,
    "value_counts": value_counts,
    "aggregate": aggregate,
    "group_by": group_by,
    "correlation": correlation,
    "distribution": distribution,
    "top_n": top_n,
}


def execute_analysis_plan(df, analysis_plan):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if not isinstance(analysis_plan, dict):
        raise TypeError("Analysis plan must be a dictionary.")

    tasks = analysis_plan.get("tasks", [])
    results, errors = [], []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append({"task_index": index, "error": "Invalid task format."})
            continue
        operation = task.get("operation")
        handler = HANDLERS.get(operation)
        if handler is None:
            errors.append({"task_index": index, "operation": operation, "error": "Unsupported operation."})
            continue
        try:
            results.append({
                "operation": operation,
                "parameters": {k: v for k, v in task.items() if k != "operation"},
                "status": "success",
                "result": handler(df, task),
            })
        except Exception as error:
            errors.append({"task_index": index, "operation": operation, "error": str(error)})

    status = "failed" if not results else ("completed_with_errors" if errors else "success")
    return {
        "agent": "Analyst Agent",
        "status": status,
        "total_tasks": len(tasks),
        "total_executed": len(results),
        "total_errors": len(errors),
        "results": results,
        "errors": errors,
    }

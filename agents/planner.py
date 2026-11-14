"""Schema-aware planning with a deterministic fallback when Gemini is unavailable."""

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from utils.llm import get_llm


Operation = Literal[
    "dataset_overview", "column_summary", "value_counts", "aggregate",
    "group_by", "correlation", "distribution", "top_n",
]


class AnalysisTask(BaseModel):
    operation: Operation
    columns: list[str] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    metric: Optional[str] = None
    aggregation: Optional[Literal["sum", "mean", "median", "min", "max", "count", "nunique", "std"]] = None
    limit: int = Field(default=10, ge=1, le=100)
    description: str = ""


class AnalysisPlan(BaseModel):
    identified_columns: list[str] = Field(default_factory=list)
    tasks: list[AnalysisTask] = Field(min_length=1, max_length=12)
    visualizations: list[Literal["bar_chart", "histogram", "boxplot", "scatter_plot", "correlation_heatmap"]] = Field(default_factory=list)


def _schema_context(report: dict) -> str:
    overview = report.get("dataset_overview", {})
    profiles = report.get("column_profiles", [])
    return str({
        "rows": overview.get("total_rows"),
        "columns": profiles or [
            {"name": name, "dtype": dtype}
            for name, dtype in overview.get("data_types", {}).items()
        ],
        "sample_values": overview.get("preview", [])[:3],
    })


def validate_analysis_plan(plan: AnalysisPlan, available_columns: list[str]) -> dict:
    valid = set(available_columns)
    tasks, used = [], []

    for task in plan.tasks:
        columns = list(dict.fromkeys(task.columns))
        groups = list(dict.fromkeys(task.group_by))
        referenced = columns + groups + ([task.metric] if task.metric else [])
        if any(column not in valid for column in referenced):
            continue
        if task.operation in {"column_summary", "value_counts", "distribution"} and not columns:
            continue
        if task.operation in {"aggregate", "top_n"} and not task.metric:
            continue
        if task.operation == "group_by" and (not groups or not task.metric):
            continue
        if task.operation == "correlation" and len(columns) < 2:
            continue
        if task.operation in {"aggregate", "group_by"} and task.aggregation is None:
            continue
        tasks.append(task.model_dump())
        used.extend(referenced)

    if not tasks:
        raise ValueError("No valid analysis operation could be created for the supplied columns.")

    return {
        "agent": "Planner Agent",
        "status": "success",
        "planner_mode": "schema_aware_gemini",
        "identified_columns": list(dict.fromkeys(used)),
        "tasks": tasks,
        "visualizations": list(dict.fromkeys(plan.visualizations)),
        "total_tasks": len(tasks),
    }


def _normalise(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _find_column(columns, words, numeric_only=False, data_types=None):
    candidates = []
    for column in columns:
        normal = _normalise(column)
        tokens = set(normal.split())
        if any(word in tokens for word in words):
            if numeric_only and data_types and not str(data_types.get(column, "")).startswith(("int", "float")):
                continue
            candidates.append(column)
    return candidates[0] if candidates else None


def create_rule_based_analysis_plan(user_request: str, exploration_report: dict) -> dict:
    """Create a safe plan for common questions without making an LLM call."""
    overview = exploration_report.get("dataset_overview", {})
    columns = overview.get("column_names", [])
    data_types = overview.get("data_types", {})
    profiles = exploration_report.get("column_profiles", [])
    if not columns:
        raise ValueError("No columns were found in the uploaded dataset.")

    request = _normalise(user_request)
    numeric_columns = [
        col for col in columns
        if str(data_types.get(col, "")).startswith(("int", "float"))
    ]
    categorical_columns = [
        col for col in columns
        if col not in numeric_columns
        and (not profiles or next((p.get("inferred_type") for p in profiles if p.get("name") == col), "") in {"categorical", "text_or_identifier"})
    ]

    direction = "max" if re.search(r"\b(highest|maximum|max|largest|most expensive|top|highest priced)\b", request) else (
        "min" if re.search(r"\b(lowest|minimum|min|smallest|cheapest)\b", request) else None
    )
    metric = _find_column(columns, {"price", "amount", "cost", "salary", "revenue", "sales", "profit", "score", "value", "quantity", "stock", "inventory", "units", "rating", "marks", "tenure", "age"}, True, data_types)

    entity = None
    if re.search(r"\b(product|item|name|category|department|region|city|country|customer|employee)\b", request):
        entity = _find_column(
            categorical_columns,
            set(request.split()) & {"product", "item", "name", "category", "department", "region", "city", "country", "customer", "employee"},
        )
        if entity is None:
            # Prefer a descriptive name/category-like column over arbitrary text.
            entity = _find_column(columns, {"product", "item", "name", "category", "department", "region", "city", "country", "customer", "employee"})

    tasks = []
    asks_for_average = bool(re.search(r"\b(average|mean)\b", request))
    if direction and metric and entity and asks_for_average:
        tasks.append(AnalysisTask(
            operation="group_by",
            group_by=[entity],
            metric=metric,
            aggregation="mean",
            limit=1,
            description="lowest_average" if direction == "min" else "highest_average",
        ))
    elif direction and metric:
        if entity:
            tasks.append(AnalysisTask(
                operation="group_by",
                group_by=[entity],
                metric=metric,
                aggregation=direction,
                limit=10,
                description="minimum" if direction == "min" else f"{direction} {metric} by {entity}",
            ))
        else:
            tasks.append(AnalysisTask(
                operation="top_n",
                metric=metric,
                limit=1,
                description=f"Rows with {direction} {metric}",
            ))
            # top_n is descending by design; handle minimum with ascending flag in analyst later.
            if direction == "min":
                tasks[-1].description = "minimum"
    elif re.search(r"\b(count|how many|frequency|distribution)\b", request) and entity:
        tasks.append(AnalysisTask(operation="value_counts", columns=[entity], limit=20))
    elif re.search(r"\b(average|mean|sum|total|median)\b", request) and metric:
        aggregation = "mean" if re.search(r"\b(average|mean)\b", request) else (
            "median" if "median" in request else "sum"
        )
        tasks.append(AnalysisTask(operation="aggregate", metric=metric, aggregation=aggregation))
    else:
        tasks.append(AnalysisTask(operation="dataset_overview"))

    return {
        "agent": "Planner Agent",
        "status": "success",
        "planner_mode": "rule_based_fallback",
        "fallback_reason": "Gemini Planner was unavailable or its quota was exhausted.",
        "identified_columns": list(dict.fromkeys(
            [task.metric for task in tasks if task.metric]
            + [column for task in tasks for column in task.columns + task.group_by]
        )),
        "tasks": [task.model_dump() for task in tasks],
        "visualizations": [],
        "total_tasks": len(tasks),
    }


def create_analysis_plan(user_request: str, exploration_report: dict) -> dict:
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("User request cannot be empty.")
    if not isinstance(exploration_report, dict):
        raise TypeError("Exploration report must be a dictionary.")

    columns = exploration_report.get("dataset_overview", {}).get("column_names", [])
    if not columns:
        raise ValueError("No columns were found in the uploaded dataset.")

    # Fast-path common requests when the schema provides a clear metric.
    normalised_request = _normalise(user_request)
    simple_patterns = [
        r"\b(highest|maximum|max|largest|most expensive|highest priced|lowest|minimum|min|smallest|cheapest)\b",
        r"\b(average|mean|sum|total|median)\b",
        r"\b(count|how many|frequency)\b",
    ]
    is_simple = any(re.search(pattern, normalised_request) for pattern in simple_patterns)
    metric_words = {
        "price", "amount", "cost", "salary", "revenue", "sales", "profit",
        "score", "value", "quantity", "stock", "inventory", "units",
        "rating", "marks", "tenure", "age",
    }
    has_metric = _find_column(
        columns,
        metric_words,
        True,
        exploration_report.get("dataset_overview", {}).get("data_types", {}),
    )
    has_entity = _find_column(
        columns,
        {"product", "item", "name", "category", "department", "region",
         "city", "country", "customer", "employee"},
    )

    if is_simple and (
        has_metric
        or (has_entity and re.search(r"\b(count|how many|frequency)\b", normalised_request))
    ):
        fast_plan = create_rule_based_analysis_plan(user_request, exploration_report)
        fast_plan["planner_mode"] = "fast_rule_based"
        fast_plan.pop("fallback_reason", None)
        return fast_plan

    try:
        parser = PydanticOutputParser(pydantic_object=AnalysisPlan)
        prompt = PromptTemplate(
            template="""Plan the user's data question using only the supplied schema and sample values.
The dataset may belong to any domain. Never assume fixed column names.

QUESTION:
{user_request}

DATASET:
{schema_context}

Allowed operations: dataset_overview, column_summary, value_counts, aggregate, group_by, correlation, distribution, top_n.
Use exact column names. For group_by provide group_by, metric and aggregation. For highest/lowest questions, identify the correct numeric metric and relevant entity column. Do not invent results or columns. Do not treat identifiers as measurements. Keep the plan focused.

{format_instructions}
""",
            input_variables=["user_request", "schema_context"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        response = get_llm().invoke(prompt.format(
            user_request=user_request,
            schema_context=_schema_context(exploration_report),
        ))
        content = response.content
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("AI Planner returned an empty response.")
        structured = parser.parse(content)
        return validate_analysis_plan(structured, columns)

    except Exception as error:
        fallback = create_rule_based_analysis_plan(user_request, exploration_report)
        fallback["fallback_reason"] = str(error)
        return fallback

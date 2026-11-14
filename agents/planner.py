"""Schema-aware analysis planning for arbitrary tabular datasets."""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from utils.llm import get_llm


Operation = Literal[
    "dataset_overview",
    "column_summary",
    "value_counts",
    "aggregate",
    "group_by",
    "correlation",
    "distribution",
    "top_n",
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
    tasks = []
    used = []

    for task in plan.tasks:
        columns = list(dict.fromkeys(task.columns))
        groups = list(dict.fromkeys(task.group_by))
        referenced = columns + groups + ([task.metric] if task.metric else [])
        if any(column not in valid for column in referenced):
            continue
        if task.operation in {"column_summary", "value_counts", "distribution"} and not columns:
            continue
        if task.operation in {"aggregate", "top_n"} and (not task.metric or task.metric not in valid):
            continue
        if task.operation == "group_by" and (not groups or not task.metric):
            continue
        if task.operation == "correlation" and len(columns) < 2:
            continue
        if task.aggregation is None and task.operation in {"aggregate", "group_by"}:
            continue
        tasks.append(task.model_dump())
        used.extend(referenced)

    if not tasks:
        raise ValueError(
            "The planner did not produce a valid operation for the available columns. "
            "Try asking about a specific column or a general dataset overview."
        )

    return {
        "agent": "Planner Agent",
        "status": "success",
        "planner_mode": "schema_aware_gemini",
        "identified_columns": list(dict.fromkeys(used)),
        "tasks": tasks,
        "visualizations": list(dict.fromkeys(plan.visualizations)),
        "total_tasks": len(tasks),
    }


def create_analysis_plan(user_request: str, exploration_report: dict) -> dict:
    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("User request cannot be empty.")
    if not isinstance(exploration_report, dict):
        raise TypeError("Exploration report must be a dictionary.")

    overview = exploration_report.get("dataset_overview", {})
    available_columns = overview.get("column_names", [])
    if not available_columns:
        raise ValueError("No columns were found in the uploaded dataset.")

    parser = PydanticOutputParser(pydantic_object=AnalysisPlan)
    prompt = PromptTemplate(
        template="""You are the schema-aware planning component of a general-purpose data analysis platform.

Create a plan for the user's exact question using ONLY the supplied dataset schema and sample values.
The dataset may belong to any domain. Never assume sales, customers, age, gender, or any particular column name.

USER QUESTION:
{user_request}

DATASET CONTEXT:
{schema_context}

ALLOWED OPERATIONS:
- dataset_overview: general dataset-level facts.
- column_summary: descriptive statistics for named columns.
- value_counts: frequency counts for categorical/discrete columns.
- aggregate: one numeric column aggregated using sum/mean/median/min/max/count/nunique/std.
- group_by: aggregate a numeric metric for one or more grouping columns.
- correlation: relationship between at least two numeric columns.
- distribution: distribution of one numeric column.
- top_n: rank rows/groups by a metric; use group_by first when ranking groups.

RULES:
- Use exact column names from the supplied schema.
- Pick only operations needed to answer the question.
- For a broad request such as "analyse this dataset", include overview and useful column summaries/value counts for suitable columns.
- Do not invent columns, data, results, code, or SQL.
- For grouping questions, use group_by with group_by and metric.
- For "how many records per category", use value_counts on the relevant column.
- Do not treat an identifier as a meaningful measurement merely because it is numeric.
- Keep the plan focused; maximum 8 tasks.
- Suggest visualizations only when they directly help answer the question.

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
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("AI Planner returned an empty response.")

    structured = parser.parse(content)
    return validate_analysis_plan(structured, available_columns)

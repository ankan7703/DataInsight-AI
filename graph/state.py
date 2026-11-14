
from typing import TypedDict, Any
import pandas as pd


class DataInsightState(TypedDict):

    # User input
    df: pd.DataFrame
    user_request: str
    include_detailed_report: bool

    # Explorer Agent output
    exploration_report: dict[str, Any]

    # Planner Agent output
    analysis_plan: dict[str, Any]

    # Analyst Agent output
    analysis_results: dict[str, Any]

    # Quick Answer
    direct_answer: str

    # Insight Writer Agent output
    insight_report: dict[str, Any]

    # Workflow error tracking
    errors: list[str]

    # Workflow status
    status: str
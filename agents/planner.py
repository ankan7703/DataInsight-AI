
from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from utils.llm import get_llm


# ---------------------------------------
# 1. Define the allowed analysis operations
# ---------------------------------------

AllowedOperation = Literal[
    "calculate_purchase_statistics",
    "group_by_category",
    "analyze_age_distribution",
    "group_by_city",
    "general_exploration"
]


# ---------------------------------------
# 2. Define the structured output schema
# ---------------------------------------

class AnalysisTask(BaseModel):
    operation: AllowedOperation = Field(
        description="One supported analysis operation."
    )


class AnalysisPlan(BaseModel):
    identified_columns: list[str] = Field(
        description="Dataset columns relevant to the request."
    )

    tasks: list[AnalysisTask] = Field(
        description="The analysis operations required."
    )

    visualizations: list[str] = Field(
        description="Suggested visualization types."
    )


# ---------------------------------------
# 3. Rule-based fallback Planner
# ---------------------------------------


def create_rule_based_analysis_plan(
    user_request,
    exploration_report
):
    """Create an analysis plan without using an LLM."""

    columns = exploration_report[
        "dataset_overview"
    ]["column_names"]

    request = user_request.lower()

    tasks = []
    identified_columns = []
    visualizations = []

    # 1. Product category analysis
    if any(word in request for word in [
        "category", "categories",
        "product", "products"
    ]):

        if all(column in columns for column in [
            "product_category",
            "purchase_amount"
        ]):

            tasks.append({
                "operation": "group_by_category"
            })

            identified_columns.extend([
                "product_category",
                "purchase_amount"
            ])

            visualizations.append("bar_chart")

    # 2. Overall purchase statistics
    elif any(word in request for word in [
        "purchase", "purchasing",
        "sales", "revenue",
        "spending", "transaction"
    ]):

        if "purchase_amount" in columns:

            tasks.append({
                "operation": "calculate_purchase_statistics"
            })

            identified_columns.append("purchase_amount")

    # 3. Age analysis
    elif any(word in request for word in [
        "age", "customer age",
        "age group"
    ]):

        if "age" in columns:

            tasks.append({
                "operation": "analyze_age_distribution"
            })

            identified_columns.append("age")
            visualizations.append("histogram")

    # 4. City analysis
    elif any(word in request for word in [
        "city", "cities",
        "location", "region"
    ]):

        if "city" in columns:

            tasks.append({
                "operation": "group_by_city"
            })

            identified_columns.append("city")
            visualizations.append("bar_chart")

    # Return None when the question is not recognized.
    # This allows Gemini to handle complex questions.

    if not tasks:
        return None

    return {
        "agent": "Planner Agent",
        "status": "success",
        "planner_mode": "rule_based",
        "user_request": user_request,
        "identified_columns": list(
            dict.fromkeys(identified_columns)
        ),
        "tasks": tasks,
        "visualizations": list(
            dict.fromkeys(visualizations)
        ),
        "total_tasks": len(tasks)
    }


# ---------------------------------------
# 4. Validate the Gemini-generated plan
# ---------------------------------------

def validate_analysis_plan(plan, available_columns):
    """Reject unsupported operations and invalid columns."""

    required_columns = {
        "calculate_purchase_statistics": [
            "purchase_amount"
        ],
        "group_by_category": [
            "product_category",
            "purchase_amount"
        ],
        "analyze_age_distribution": [
            "age"
        ],
        "group_by_city": [
            "city"
        ],
        "general_exploration": []
    }

    validated_tasks = []
    identified_columns = []

    for task in plan.tasks:

        operation = task.operation

        required = required_columns[operation]

        # Skip operations if required columns are missing
        if not all(
            column in available_columns
            for column in required
        ):
            continue

        validated_tasks.append({
            "operation": operation
        })

        identified_columns.extend(required)

    # Remove duplicate tasks
    unique_tasks = []
    seen_operations = set()

    for task in validated_tasks:

        operation = task["operation"]

        if operation not in seen_operations:
            unique_tasks.append(task)
            seen_operations.add(operation)

    # Use general exploration if no valid task remains
    if not unique_tasks:
        unique_tasks = [{
            "operation": "general_exploration"
        }]

    valid_visualizations = {
        "bar_chart",
        "histogram",
        "boxplot",
        "scatter_plot",
        "correlation_heatmap"
    }

    visualizations = [
        visualization
        for visualization in plan.visualizations
        if visualization in valid_visualizations
    ]

    return {
        "agent": "Planner Agent",
        "status": "success",
        "planner_mode": "gemini",
        "identified_columns": list(
            dict.fromkeys(identified_columns)
        ),
        "tasks": unique_tasks,
        "visualizations": list(
            dict.fromkeys(visualizations)
        ),
        "total_tasks": len(unique_tasks)
    }


# ---------------------------------------
# 5. Gemini-powered Planner
# ---------------------------------------

def create_analysis_plan(
    user_request,
    exploration_report
):
    """Generate a structured analysis plan using Gemini."""

    if not isinstance(user_request, str) or not user_request.strip():
        raise ValueError("User request cannot be empty.")

    if not isinstance(exploration_report, dict):
        raise TypeError("Exploration report must be a dictionary.")

    available_columns = exploration_report[
        "dataset_overview"
    ]["column_names"]

    # Try rule-based planning first
    fast_plan = create_rule_based_analysis_plan(
        user_request,
        exploration_report
    )

    if fast_plan is not None:

        print(
            "Using fast rule-based Planner. "
            "Gemini API call skipped."
        )

        return fast_plan

    print(
        "Question requires advanced understanding. "
        "Using Gemini Planner..."
    )

    parser = PydanticOutputParser(
        pydantic_object=AnalysisPlan
    )

    prompt = PromptTemplate(
        template="""
You are the Planner Agent of DataInsight AI.

Your job is to understand the user's data analysis request
and create a structured analysis plan.

USER REQUEST:
{user_request}

AVAILABLE DATASET COLUMNS:
{available_columns}

SUPPORTED OPERATIONS:

1. calculate_purchase_statistics
   Use for total, average, median, minimum or maximum
   purchase amount.

2. group_by_category
   Use to compare purchase amounts across product categories.

3. analyze_age_distribution
   Use to analyze customer ages and age groups.

4. group_by_city
   Use to compare the number of records across cities.

5. general_exploration
   Use for general dataset analysis.

RULES:

- Select only operations directly necessary to answer the user's question.
- Do not select additional operations merely because their columns are available.
- If group_by_category answers the question, do not add
  calculate_purchase_statistics unless overall purchase statistics
  are explicitly requested.
- Never invent dataset columns.
- Only select operations whose required columns exist.
- Select multiple operations only when the question requires them.
- Suggest only supported visualization types.
- Do not generate Python code or SQL queries.
- Do not invent numerical findings.
- Return the result in the requested structured format.

{format_instructions}
""",
        input_variables=[
            "user_request",
            "available_columns"
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    try:

        llm = get_llm()

        formatted_prompt = prompt.format(
            user_request=user_request,
            available_columns=", ".join(available_columns)
        )

        response = llm.invoke(formatted_prompt)

        # Extract text from the model response
        if isinstance(response.content, str):
            response_text = response.content
        else:
            response_text = "".join(
                block.get("text", "")
                for block in response.content
                if isinstance(block, dict)
            )

        # Parse Gemini response into Pydantic model
        structured_plan = parser.parse(response_text)

        # Validate operations and columns
        return validate_analysis_plan(
            structured_plan,
            available_columns
        )

    except Exception as error:

        print(
            f"Gemini Planner failed: {error}"
        )

        print(
            "Switching to rule-based Planner..."
        )

        fallback_plan = create_rule_based_analysis_plan(
            user_request,
            exploration_report
        )

        fallback_plan["fallback_reason"] = str(error)

        return fallback_plan

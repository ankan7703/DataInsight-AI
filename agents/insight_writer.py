"""Evidence-grounded answer and insight generation for any dataset."""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from utils.llm import get_llm


class InsightReport(BaseModel):
    business_insights: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    data_limitations: list[str] = Field(default_factory=list)


def extract_response_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    raise ValueError("Unexpected AI response format.")


def _validate_results(analysis_results):
    if not isinstance(analysis_results, dict):
        raise TypeError("Analysis results must be a dictionary.")
    if analysis_results.get("status") not in {"success", "completed_with_errors"}:
        raise ValueError("There are no successful analysis results to interpret.")
    return analysis_results


def _format_computed_answer(user_request, analysis_results):
    """Produce a transparent answer from deterministic Pandas results."""
    results = _validate_results(analysis_results)
    successful = results.get("results", [])
    if not successful:
        return "The analysis did not produce a usable result."

    for item in successful:
        operation = item.get("operation")
        data = item.get("result")

        if operation == "group_by" and isinstance(data, list) and data:
            first = data[0]
            group_fields = [key for key in first if key != "result"]
            if group_fields:
                group_label = ", ".join(f"{key}: {first[key]}" for key in group_fields)
                metric = item.get("parameters", {}).get("metric", "the selected metric")
                aggregation = item.get("parameters", {}).get("aggregation", "value")
                description = item.get("parameters", {}).get("description")
                if description in {"highest_average", "lowest_average"}:
                    direction = "highest" if description == "highest_average" else "lowest"
                    return (
                        f"**{group_label}** has the {direction} average "
                        f"**{metric}**: **{first['result']}**."
                    )
                return (
                    f"Based on the dataset, **{group_label}** has the "
                    f"{aggregation} {metric}: **{first['result']}**."
                )

        if operation == "top_n" and isinstance(data, list) and data:
            first = data[0]
            params = item.get("parameters", {})
            metric = params.get("metric", "the selected metric")
            direction = "lowest" if params.get("description") == "minimum" else "highest"
            return (
                f"The record with the {direction} **{metric}** is: "
                + ", ".join(f"{key}: {value}" for key, value in first.items())
                + "."
            )

        if operation == "aggregate" and isinstance(data, dict):
            return (
                f"The {data.get('aggregation')} of **{data.get('column')}** "
                f"is **{data.get('value')}**."
            )

        if operation == "value_counts" and isinstance(data, dict):
            column, counts = next(iter(data.items()))
            if counts:
                first = counts[0]
                return (
                    f"The most frequent value in **{column}** is "
                    f"**{first['value']}**, appearing **{first['count']}** time(s)."
                )

        if operation == "dataset_overview" and isinstance(data, dict):
            return (
                f"The dataset contains **{data.get('rows')} rows** and "
                f"**{data.get('columns')} columns**."
            )

    return (
        "The computed analysis ran successfully, but the available result "
        "does not directly answer the question. Try a more specific question."
    )


def generate_direct_answer(user_request, analysis_results):
    """Answer simple computed questions locally; use Gemini only for richer interpretation."""
    results = _validate_results(analysis_results)

    # These operations already contain the exact computed answer. A second LLM
    # request adds latency without improving factual accuracy.
    deterministic_operations = {"group_by", "top_n", "aggregate", "value_counts", "dataset_overview"}
    successful_operations = {
        item.get("operation") for item in results.get("results", [])
    }
    if successful_operations and successful_operations.issubset(deterministic_operations):
        return _format_computed_answer(user_request, results)

    try:
        prompt = PromptTemplate(
            template="""Answer the user's exact question using ONLY the computed analysis results.
Do not invent values, facts, causes, or recommendations. Clearly state when the results do not answer the question.
Distinguish counts, sums, averages and other statistics. Never claim correlation proves causation.

QUESTION:
{question}

COMPUTED RESULTS:
{results}

Return only the answer.
""",
            input_variables=["question", "results"],
        )
        response = get_llm().invoke(prompt.format(
            question=user_request,
            results=str(results),
        ))
        answer = extract_response_text(response).strip()
        if answer:
            return answer
        raise ValueError("AI returned an empty answer.")
    except Exception:
        return _format_computed_answer(user_request, results)


def generate_rule_based_insights(analysis_results):
    """Safe fallback when the language model is unavailable."""
    results = _validate_results(analysis_results)
    insights = []
    for item in results.get("results", []):
        if item.get("operation") == "dataset_overview":
            data = item.get("result", {})
            insights.append(
                f"The dataset contains {data.get('rows', 0)} rows "
                f"and {data.get('columns', 0)} columns."
            )
            missing = data.get("missing_values", {})
            missing_total = sum(missing.values())
            insights.append(f"It contains {missing_total} missing cell(s).")
    errors = results.get("errors", [])
    limitations = [
        f"{len(errors)} planned analysis operation(s) could not be completed."
    ] if errors else [
        "Automated narrative interpretation was unavailable; review the computed results directly."
    ]
    return {
        "agent": "Insight Writer Agent",
        "status": "success",
        "writer_mode": "rule_based_fallback",
        "insights": insights,
        "recommendations": [],
        "data_limitations": limitations,
    }


def generate_insights(analysis_results):
    """Generate domain-neutral insights from verified analysis output."""
    results = _validate_results(analysis_results)
    parser = PydanticOutputParser(pydantic_object=InsightReport)
    prompt = PromptTemplate(
        template="""You are the insight-writing component of a general-purpose data intelligence platform.

Interpret only the computed analysis results. The data may belong to any domain.
Do not automatically frame findings as business insights.

COMPUTED RESULTS:
{results}

INSTRUCTIONS:
- State the most meaningful findings supported by the results.
- Make recommendations only when the evidence supports an actionable recommendation.
- Identify sample-size, missing-data, extraction, and analysis limitations when visible.
- Do not invent numbers, facts, causal explanations, or domain context.
- Distinguish association from causation.
- If the results are insufficient, say so.
- Use neutral language appropriate to the dataset's domain.

{format_instructions}
""",
        input_variables=["results"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    try:
        response = get_llm().invoke(prompt.format(results=str(results)))
        report = parser.parse(extract_response_text(response))
        return {
            "agent": "Insight Writer Agent",
            "status": "success",
            "writer_mode": "gemini",
            "insights": report.business_insights,
            "recommendations": report.recommendations,
            "data_limitations": report.data_limitations,
        }
    except Exception as error:
        fallback = generate_rule_based_insights(results)
        fallback["fallback_reason"] = str(error)
        return fallback

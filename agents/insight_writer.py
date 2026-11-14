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


def generate_direct_answer(user_request, analysis_results):
    """Answer the exact question using computed results, never raw guesses."""
    results = _validate_results(analysis_results)
    prompt = PromptTemplate(
        template="""You are the answer-writing component of a general-purpose data intelligence platform.

Answer the user's exact question using ONLY the computed analysis results below.
The dataset may come from any domain. Do not assume a business context.

QUESTION:
{question}

COMPUTED RESULTS:
{results}

RULES:
- Do not invent values, records, explanations, causes, or recommendations.
- Clearly state when the computed results do not answer the question.
- Distinguish counts, sums, averages, and other statistics correctly.
- Mention relevant failed operations if they affect the answer.
- Keep the answer direct, readable, and appropriately detailed.
- Never claim correlation proves causation.
Return only the answer.
""",
        input_variables=["question", "results"],
    )
    response = get_llm().invoke(prompt.format(
        question=user_request,
        results=str(results),
    ))
    answer = extract_response_text(response).strip()
    if not answer:
        raise ValueError("AI returned an empty answer.")
    return answer


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

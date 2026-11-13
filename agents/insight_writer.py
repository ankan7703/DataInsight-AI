
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from utils.llm import get_llm


# ---------------------------------------
# 1. Define structured output
# ---------------------------------------

class InsightReport(BaseModel):

    business_insights: list[str] = Field(
        description="Important findings from the analysis."
    )

    recommendations: list[str] = Field(
        description="Practical business recommendations."
    )

    data_limitations: list[str] = Field(
        description="Limitations affecting interpretation."
    )


# ---------------------------------------
# 2. Rule-based fallback
# ---------------------------------------

def generate_rule_based_insights(analysis_results):

    insights = []
    recommendations = []

    for item in analysis_results.get("results", []):

        operation = item.get("operation")
        data = item.get("result")

        if operation == "group_by_category" and data:

            highest = max(
                data,
                key=lambda x: x["average_purchase_amount"]
            )

            lowest = min(
                data,
                key=lambda x: x["average_purchase_amount"]
            )

            insights.append(
                f"{highest['product_category']} has the highest "
                f"average purchase amount at "
                f"₹{highest['average_purchase_amount']:,.2f}."
            )

            insights.append(
                f"{lowest['product_category']} has the lowest "
                f"average purchase amount at "
                f"₹{lowest['average_purchase_amount']:,.2f}."
            )

            recommendations.append(
                f"Investigate the factors contributing to the "
                f"higher average purchase amount in "
                f"{highest['product_category']}."
            )

        elif operation == "calculate_purchase_statistics":

            average = data.get("average_purchase_amount")
            total = data.get("total_purchase_amount")

            insights.append(
                f"The average transaction value is "
                f"₹{average:,.2f}, with total purchases "
                f"of ₹{total:,.2f}."
            )

        elif operation == "analyze_age_distribution":

            insights.append(
                f"The average customer age is "
                f"{data['average_age']:.1f} years."
            )

    errors = analysis_results.get("errors", [])

    if errors:
        insights.append(
            f"{len(errors)} analysis task(s) failed. "
            "Interpret the available results cautiously."
        )

    return {
        "agent": "Insight Writer Agent",
        "status": "success",
        "writer_mode": "rule_based_fallback",
        "insights": insights,
        "recommendations": recommendations,
        "data_limitations": []
    }


# ---------------------------------------
# 3. Gemini-powered Insight Writer
# ---------------------------------------

def generate_insights(analysis_results):

    if not isinstance(analysis_results, dict):
        raise TypeError("Analysis results must be a dictionary.")

    if analysis_results.get("status") not in [
        "success",
        "completed_with_errors"
    ]:
        raise ValueError("Invalid analysis results.")

    parser = PydanticOutputParser(
        pydantic_object=InsightReport
    )

    prompt = PromptTemplate(
        template="""
You are the Insight Writer Agent of DataInsight AI.

Your responsibility is to interpret numerical analysis
results and produce accurate business insights.

ANALYSIS RESULTS:
{analysis_results}

INSTRUCTIONS:

1. Identify the most important findings.
2. Explain what the findings mean in business terms.
3. Generate practical and relevant recommendations.
4. Identify limitations in the available data.
5. Never invent numerical values.
6. Do not claim causation from correlation or descriptive data.
7. Do not make unsupported business predictions.
8. Consider the number of observations and data quality.
9. If the analysis contains errors, mention their impact.
10. Keep insights clear, concise and professional.

Return your answer using the required structured format.

{format_instructions}
""",
        input_variables=["analysis_results"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )

    try:

        llm = get_llm()

        formatted_prompt = prompt.format(
            analysis_results=str(analysis_results)
        )

        response = llm.invoke(formatted_prompt)

        if isinstance(response.content, str):
            response_text = response.content

        else:
            response_text = "".join(
                block.get("text", "")
                for block in response.content
                if isinstance(block, dict)
            )

        structured_report = parser.parse(response_text)

        return {
            "agent": "Insight Writer Agent",
            "status": "success",
            "writer_mode": "gemini",
            "insights": structured_report.business_insights,
            "recommendations": structured_report.recommendations,
            "data_limitations": structured_report.data_limitations
        }

    except Exception as error:

        print(
            f"Gemini Insight Writer failed: {error}"
        )

        print(
            "Switching to rule-based Insight Writer..."
        )

        fallback_report = generate_rule_based_insights(
            analysis_results
        )

        fallback_report["fallback_reason"] = str(error)

        return fallback_report

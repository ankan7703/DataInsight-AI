
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
# 2. Helper: Extract Gemini response
# ---------------------------------------

def extract_response_text(response):

    if isinstance(response.content, str):
        return response.content

    if isinstance(response.content, list):

        return "".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
        )

    raise ValueError("Unexpected Gemini response format.")


# ---------------------------------------
# 3. Rule-based Quick Answer Fallback
# ---------------------------------------


def generate_rule_based_answer(
    user_request,
    analysis_results
):

    question = user_request.lower()
    answers = []

    for item in analysis_results.get("results", []):

        operation = item.get("operation")
        data = item.get("result")

        # ---------------------------------------
        # 1. Product Category Analysis
        # ---------------------------------------

        if operation == "group_by_category" and data:

            # Highest total purchase amount
            if any(word in question for word in [
                "highest", "maximum", "most", "top"
            ]) and any(word in question for word in [
                "total", "revenue", "purchase"
            ]):

                highest = max(
                    data,
                    key=lambda x: x.get(
                        "total_purchase_amount", float("-inf")
                    )
                )

                answers.append(
                    f"{highest['product_category']} has the "
                    f"highest total purchase amount, at "
                    f"₹{highest['total_purchase_amount']:,.2f}."
                )

            # Lowest total purchase amount
            elif any(word in question for word in [
                "lowest", "minimum", "least"
            ]) and any(word in question for word in [
                "total", "revenue", "purchase"
            ]):

                lowest = min(
                    data,
                    key=lambda x: x.get(
                        "total_purchase_amount", float("inf")
                    )
                )

                answers.append(
                    f"{lowest['product_category']} has the "
                    f"lowest total purchase amount, at "
                    f"₹{lowest['total_purchase_amount']:,.2f}."
                )

            # Average purchase amount for each category
            if "average" in question:

                averages = [
                    f"{row['product_category']}: "
                    f"₹{row['average_purchase_amount']:,.2f}"
                    for row in data
                ]

                answers.append(
                    "Average purchase amount by category: "
                    + "; ".join(averages) + "."
                )

            # Total purchase amount for each category
            if any(phrase in question for phrase in [
                "total purchase amount for each",
                "total amount for each",
                "total purchase amount by category",
                "total sales by category",
                "total revenue by category"
            ]):

                totals = [
                    f"{row['product_category']}: "
                    f"₹{row['total_purchase_amount']:,.2f}"
                    for row in data
                ]

                answers.append(
                    "Total purchase amount by category: "
                    + "; ".join(totals) + "."
                )

        # ---------------------------------------
        # 2. Overall Purchase Statistics
        # ---------------------------------------

        elif operation == "calculate_purchase_statistics" and data:

            total = data.get("total_purchase_amount")
            average = data.get("average_purchase_amount")

            if total is not None and any(word in question for word in [
                "total", "revenue"
            ]):
                answers.append(
                    f"Total purchase amount is ₹{total:,.2f}."
                )

            if average is not None and "average" in question:
                answers.append(
                    f"Average purchase amount is ₹{average:,.2f}."
                )

        # ---------------------------------------
        # 3. Age Distribution
        # ---------------------------------------

        elif operation == "analyze_age_distribution" and data:

            average_age = data.get("average_age")

            if average_age is not None:
                answers.append(
                    f"The average customer age is "
                    f"{average_age:.1f} years."
                )

        # ---------------------------------------
        # 4. City Analysis
        # ---------------------------------------

        elif operation == "group_by_city" and data:

            highest = max(
                data,
                key=lambda x: x.get(
                    "total_purchase_amount", 0
                )
            )

            answers.append(
                f"{highest.get('city', 'This city')} has the "
                f"highest total purchase amount."
            )

    if answers:
        return " ".join(answers)

    return (
        "The analysis was completed, but a concise answer "
        "could not be generated from the available results. "
        "Please refer to the analysis results."
    )


# ---------------------------------------
# 4. Gemini-powered Quick Answer
# ---------------------------------------


def generate_direct_answer(
    user_request,
    analysis_results
):

    if not isinstance(analysis_results, dict):
        raise TypeError("Analysis results must be a dictionary.")

    if analysis_results.get("status") not in [
        "success",
        "completed_with_errors"
    ]:
        raise ValueError("Invalid analysis results.")

    # Step 1: Try answering using Python first.
    rule_based_answer = generate_rule_based_answer(
        user_request,
        analysis_results
    )

    # Step 2: If Python can answer the question,
    # return immediately without calling Gemini.
    if rule_based_answer:

        print("Using Python Quick Answer. Gemini API call skipped.")

        return rule_based_answer

    # Step 3: Use Gemini for questions that need
    # more advanced natural-language understanding.
    try:

        llm = get_llm()

        prompt = PromptTemplate(
            template="""
You are the Quick Answer Agent of DataInsight AI.

Answer the user's exact question using only the
provided analysis results.

USER QUESTION:
{user_request}

ANALYSIS RESULTS:
{analysis_results}

INSTRUCTIONS:
1. Use simple everyday language.
2. Keep the answer to one or two short sentences.
3. Include relevant numerical values when available.
4. Never invent facts or numbers.
5. If the results cannot answer the question, say so.
6. Do not hide failed analysis tasks.

Return ONLY the direct answer.
""",
            input_variables=[
                "user_request",
                "analysis_results"
            ]
        )

        formatted_prompt = prompt.format(
            user_request=user_request,
            analysis_results=str(analysis_results)
        )

        response = llm.invoke(formatted_prompt)

        answer = extract_response_text(response).strip()

        if not answer:
            raise ValueError("Gemini returned an empty answer.")

        return answer

    except Exception as error:

        print(f"Gemini Quick Answer failed: {error}")

        return (
            "I couldn't generate an advanced answer right now. "
            "Please try again when the AI service is available."
        )

    finally:

        print("Using Gemini Quick Answer.")

# ---------------------------------------
# 5. Rule-based Detailed Report Fallback
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

            if average is not None and total is not None:

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
# 6. Gemini-powered Detailed Insight Writer
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

        response_text = extract_response_text(response)

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

        print(f"Gemini Insight Writer failed: {error}")
        print("Switching to rule-based Insight Writer...")

        fallback_report = generate_rule_based_insights(
            analysis_results
        )

        fallback_report["fallback_reason"] = str(error)

        return fallback_report
    
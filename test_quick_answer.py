
from tools.data_profiler import load_dataset
from agents.analyst import group_by_category
from agents.insight_writer import (
    generate_direct_answer,
    generate_rule_based_answer
)

df = load_dataset("data/sample.csv")

analysis_results = {
    "status": "success",
    "results": [
        {
            "operation": "group_by_category",
            "status": "success",
            "result": group_by_category(df)
        }
    ],
    "errors": []
}

question = (
    "Which product category generates the "
    "highest total purchase amount?"
)

# Test rule-based fallback first
answer = generate_rule_based_answer(
    question,
    analysis_results
)

print("\nQUICK ANSWER:")
print(answer)

assert "Clothing" in answer
assert "25,100" in answer

print("\nQUICK ANSWER TEST PASSED!")

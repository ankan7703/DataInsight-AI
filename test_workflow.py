
from tools.data_profiler import load_dataset
from graph.workflow import run_workflow


# Load dataset
df = load_dataset("data/sample.csv")

# User question
user_request = (
    "Which type of product generates the most money on average?"
)

# Execute complete LangGraph workflow
final_state = run_workflow(df, user_request)


# Display final report
print("\n========== DATAINSIGHT AI REPORT ==========")

print("\nWORKFLOW ERRORS:")
print(final_state["errors"])

print("\nBUSINESS INSIGHTS:")

for index, insight in enumerate(
    final_state["insight_report"]["insights"],
    start=1
):
    print(f"{index}. {insight}")

print("\nRECOMMENDATIONS:")

for index, recommendation in enumerate(
    final_state["insight_report"]["recommendations"],
    start=1
):
    print(f"{index}. {recommendation}")

print("\nLANGGRAPH WORKFLOW EXECUTED SUCCESSFULLY!")

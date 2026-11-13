
from tools.data_profiler import load_dataset
from agents.explorer import explore_dataset
from agents.planner import create_analysis_plan
from agents.analyst import execute_analysis_plan
from agents.insight_writer import generate_insights


# Load dataset
df = load_dataset("data/sample.csv")

# Explorer Agent
exploration_report = explore_dataset(df)

# Planner Agent
user_request = (
    "Which type of product generates the most money on average?"
)

analysis_plan = create_analysis_plan(
    user_request,
    exploration_report
)

# Analyst Agent
analysis_results = execute_analysis_plan(
    df,
    analysis_plan
)

# Gemini Insight Writer
insight_report = generate_insights(
    analysis_results
)


# Display report
print("\n========== INSIGHT WRITER AGENT ==========")

print("WRITER MODE:", insight_report["writer_mode"])

print("\nBUSINESS INSIGHTS:")

for index, insight in enumerate(
    insight_report["insights"], start=1
):
    print(f"{index}. {insight}")

print("\nRECOMMENDATIONS:")

for index, recommendation in enumerate(
    insight_report["recommendations"], start=1
):
    print(f"{index}. {recommendation}")

print("\nDATA LIMITATIONS:")

for index, limitation in enumerate(
    insight_report["data_limitations"], start=1
):
    print(f"{index}. {limitation}")

print("\nINSIGHT WRITER TEST COMPLETED!")

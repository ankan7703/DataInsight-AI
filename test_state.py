
from graph.state import DataInsightState
from tools.data_profiler import load_dataset


df = load_dataset("data/sample.csv")

initial_state: DataInsightState = {
    "df": df,
    "user_request": "Analyze customer purchasing behaviour.",
    "exploration_report": {},
    "analysis_plan": {},
    "analysis_results": {},
    "insight_report": {},
    "errors": []
}

print("STATE INITIALIZED SUCCESSFULLY!")

print("Dataset shape:", initial_state["df"].shape)
print("User request:", initial_state["user_request"])
print("Initial errors:", initial_state["errors"])

print("\nSTATE STRUCTURE TEST PASSED!")

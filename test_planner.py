
from tools.data_profiler import load_dataset
from agents.explorer import explore_dataset
from agents.planner import create_analysis_plan


# Load dataset
df = load_dataset("data/sample.csv")

# Generate Explorer Agent report
exploration_report = explore_dataset(df)

# Define user's analytical question
user_request = (
    "Analyze customer purchasing behaviour "
    "and identify which product category has "
    "the highest average purchase amount."
)

# Run Planner Agent
plan = create_analysis_plan(
    user_request,
    exploration_report
)

# Display compact results
print("AGENT:", plan["agent"])
print("STATUS:", plan["status"])

print("IDENTIFIED COLUMNS:", plan["identified_columns"])

print("TOTAL TASKS:", plan["total_tasks"])

print(
    "PLANNED OPERATIONS:",
    [task["operation"] for task in plan["tasks"]]
)

print(
    "VISUALIZATIONS:",
    [chart["type"] for chart in plan["visualizations"]]
)

print("\nPLANNER AGENT WORKING SUCCESSFULLY!")

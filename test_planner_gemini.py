
from tools.data_profiler import load_dataset
from agents.explorer import explore_dataset
from agents.planner import create_analysis_plan


# Load dataset
df = load_dataset("data/sample.csv")

# Generate dataset profile
exploration_report = explore_dataset(df)


# Test natural-language question
user_request = (
    "Which type of product generates the most money on average?"
)

# Generate analysis plan
analysis_plan = create_analysis_plan(
    user_request,
    exploration_report
)


# Display results
print("\n========== GEMINI PLANNER AGENT ==========")

print("PLANNER MODE:", analysis_plan["planner_mode"])

print("\nUSER REQUEST:")
print(user_request)

print("\nIDENTIFIED COLUMNS:")
print(analysis_plan["identified_columns"])

print("\nPLANNED OPERATIONS:")

for task in analysis_plan["tasks"]:
    print("-", task["operation"])

print("\nVISUALIZATIONS:")
print(analysis_plan["visualizations"])

print("\nTOTAL TASKS:")
print(analysis_plan["total_tasks"])

if analysis_plan["planner_mode"] == "gemini":
    print("\nGEMINI PLANNER WORKING SUCCESSFULLY!")
else:
    print("\nRULE-BASED FALLBACK WAS USED.")

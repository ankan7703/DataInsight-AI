
from tools.data_profiler import load_dataset
from agents.explorer import explore_dataset
from agents.planner import create_analysis_plan
from agents.analyst import execute_analysis_plan


# Step 1: Load dataset
df = load_dataset("data/sample.csv")

# Step 2: Explorer Agent
exploration_report = explore_dataset(df)

# Step 3: Planner Agent
user_request = (
    "Analyze customer purchasing behaviour and identify "
    "which product category has the highest average purchase amount."
)

analysis_plan = create_analysis_plan(
    user_request,
    exploration_report
)

# Step 4: Analyst Agent
analysis_results = execute_analysis_plan(
    df,
    analysis_plan
)

# Step 5: Display results
print("\n========== ANALYST AGENT ==========")

print("STATUS:", analysis_results["status"])
print("TOTAL TASKS:", analysis_results["total_tasks"])
print("EXECUTED:", analysis_results["total_executed"])
print("ERRORS:", analysis_results["total_errors"])

for item in analysis_results["results"]:

    print("\nOPERATION:", item["operation"])
    print("RESULT:")

    print(item["result"])

if analysis_results["errors"]:
    print("\nERROR DETAILS:")
    print(analysis_results["errors"])

print("\nANALYST AGENT TEST COMPLETED!")

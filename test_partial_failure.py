
from tools.data_profiler import load_dataset
from agents.analyst import execute_analysis_plan

df = load_dataset("data/sample.csv")

# One valid task and two invalid tasks
test_plan = {
    "tasks": [
        {"operation": "group_by_category"},
        {"operation": "invalid_operation"},
        "malformed_task"
    ]
}

result = execute_analysis_plan(df, test_plan)

print("\n========== PARTIAL FAILURE TEST ==========")

print("STATUS:", result["status"])
print("TOTAL TASKS:", result["total_tasks"])
print("SUCCESSFUL TASKS:", result["total_executed"])
print("FAILED TASKS:", result["total_errors"])

print("\nSUCCESSFUL RESULTS:")

for item in result["results"]:
    print(item["operation"], ":", item["status"])

print("\nERRORS:")

for error in result["errors"]:
    print(error)

assert result["status"] == "completed_with_errors"
assert result["total_executed"] == 1
assert result["total_errors"] == 2

print("\nPARTIAL FAILURE TEST PASSED!")

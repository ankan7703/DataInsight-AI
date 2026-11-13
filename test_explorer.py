
from tools.data_profiler import load_dataset
from agents.explorer import explore_dataset


# Load the sample dataset
df = load_dataset("data/sample.csv")

# Run the Explorer Agent
report = explore_dataset(df)

# Display a compact verification
print("AGENT:", report["agent"])
print("STATUS:", report["status"])

print(
    "TOTAL ROWS:",
    report["dataset_overview"]["total_rows"]
)

print(
    "MISSING COLUMNS:",
    [item["column"] for item in report["missing_values"]]
)

print(
    "DUPLICATES:",
    report["duplicate_records"]["total_duplicate_rows"]
)

print(
    "STATISTICAL SECTIONS:",
    list(report["statistical_summary"].keys())
)

print("\nEXPLORER AGENT WORKING SUCCESSFULLY!")

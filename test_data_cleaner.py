
from tools.data_profiler import load_dataset
from tools.data_cleaner import (
    drop_missing_rows,
    fill_missing_with_mean,
    fill_missing_with_median,
    fill_missing_with_mode,
    get_cleaning_summary
)


df = load_dataset("data/sample.csv")

print("\n========== DATA CLEANER TEST ==========")

print("Original missing values:", df.isnull().sum().sum())


# Test 1: Mean imputation
mean_df = fill_missing_with_mean(
    df,
    ["annual_income", "purchase_amount"]
)

print("\nMEAN IMPUTATION")
print("Remaining missing:", mean_df.isnull().sum().sum())


# Test 2: Median imputation
median_df = fill_missing_with_median(
    df,
    ["annual_income", "purchase_amount"]
)

print("\nMEDIAN IMPUTATION")
print("Remaining missing:", median_df.isnull().sum().sum())



# Test 3: Mode imputation

# Create a test dataset with a missing categorical value
mode_test_df = df.copy()

mode_test_df.loc[0, "gender"] = None

mode_df = fill_missing_with_mode(
    mode_test_df,
    ["gender", "product_category", "city"]
)

print("\nMODE IMPUTATION")

print(
    "Missing gender values:",
    mode_df["gender"].isnull().sum()
)

print(
    "Remaining numeric missing values:",
    mode_df[["annual_income", "purchase_amount"]]
    .isnull().sum().sum()
)


# Test 4: Drop rows
drop_df = drop_missing_rows(df)

print("\nDROP MISSING ROWS")
print("Original rows:", len(df))
print("Remaining rows:", len(drop_df))


# Test 5: Cleaning summary
summary = get_cleaning_summary(
    df,
    mean_df
)

print("\nCLEANING SUMMARY")

for key, value in summary.items():
    print(f"{key}: {value}")


# Assertions
assert mean_df.isnull().sum().sum() == 0
assert median_df.isnull().sum().sum() == 0
assert mode_df["gender"].isnull().sum() == 0

assert (
    mode_df[["annual_income", "purchase_amount"]]
    .isnull().sum().sum() == 2
)

assert len(drop_df) == 14

print("\nALL DATA CLEANER TESTS PASSED!")

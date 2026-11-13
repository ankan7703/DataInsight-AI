
from tools.data_profiler import load_dataset
from tools.sql_tools import (
    create_duckdb_connection,
    execute_sql_query
)


# Load the dataset
df = load_dataset("data/sample.csv")

# Create DuckDB connection
connection = create_duckdb_connection(df)

# Query 1: Average purchase amount
query1 = """
SELECT AVG(purchase_amount) AS average_purchase
FROM dataset
"""

print("AVERAGE PURCHASE:")
print(execute_sql_query(connection, query1))


# Query 2: Customers by city
query2 = """
SELECT city, COUNT(*) AS total_customers
FROM dataset
GROUP BY city
ORDER BY total_customers DESC
"""

print("\nCUSTOMERS BY CITY:")
print(execute_sql_query(connection, query2))


# Query 3: Average purchase by category
query3 = """
SELECT product_category,
       ROUND(AVG(purchase_amount), 2) AS avg_purchase
FROM dataset
GROUP BY product_category
ORDER BY avg_purchase DESC
"""

print("\nAVERAGE PURCHASE BY CATEGORY:")
print(execute_sql_query(connection, query3))


# Close the database connection
connection.close()

print("\nSQL ENGINE WORKING SUCCESSFULLY!")

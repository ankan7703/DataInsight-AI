
import duckdb
import pandas as pd
import re


# FUNCTION 1: CREATE DUCKDB CONNECTION

def create_duckdb_connection(df, table_name="dataset"):

    # Validate table name
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError("Invalid table name.")

    # in-memory DuckDB connection
    connection = duckdb.connect(database=":memory:")

    # Register Pandas DataFrame as a SQL table
    connection.register(table_name, df)

    return connection


# FUNCTION 2: EXECUTE SQL QUERY

def execute_sql_query(connection, query):

    # Remove leading and trailing whitespace
    query = query.strip()

    # Allow only SELECT or WITH queries
    if not re.match(r"^(SELECT|WITH)\b", query, re.IGNORECASE):
        raise ValueError(
            "Only SELECT and WITH queries are allowed."
        )

    # Reject multiple SQL statements
    query_without_trailing_semicolon = query.rstrip(";")

    if ";" in query_without_trailing_semicolon:
        raise ValueError("Multiple SQL statements are not allowed.")

    try:
        # Execute query and convert results to DataFrame
        result = connection.execute(query).fetchdf()

        return result

    except Exception as error:
        raise RuntimeError(
            f"SQL execution failed: {error}"
        ) from error

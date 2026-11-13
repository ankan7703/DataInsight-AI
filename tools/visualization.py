
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# FUNCTION 1: HISTOGRAM

def plot_histogram(df, column, bins=10):

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError("Histogram requires a numerical column.")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=bins,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    fig.tight_layout()

    return fig


# FUNCTION 2: BOX PLOT

def plot_boxplot(df, column):

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError("Box plot requires a numerical column.")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x=column,
        ax=ax
    )

    ax.set_title(f"Box Plot of {column}")
    ax.set_xlabel(column)

    fig.tight_layout()

    return fig


# FUNCTION 3: CORRELATION HEATMAP

def plot_correlation_heatmap(df):

    numerical_df = df.select_dtypes(include="number")

    if numerical_df.shape[1] < 2:
        raise ValueError(
            "At least two numerical columns are required."
        )

    correlation_matrix = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    fig.tight_layout()

    return fig


# FUNCTION 4: CATEGORICAL BAR CHART

def plot_categorical_bar(df, column):

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")

    if pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(
            "Bar chart requires a categorical column."
        )

    category_counts = df[column].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x=category_counts.index,
        y=category_counts.values,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()

    return fig


# FUNCTION 5: INTERACTIVE SCATTER PLOT

def plot_scatter_plotly(
    df,
    x_column,
    y_column,
    color_column=None
):

    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found.")

    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found.")

    if not pd.api.types.is_numeric_dtype(df[x_column]):
        raise ValueError("X-axis column must be numerical.")

    if not pd.api.types.is_numeric_dtype(df[y_column]):
        raise ValueError("Y-axis column must be numerical.")

    if color_column is not None:
        if color_column not in df.columns:
            raise ValueError(
                f"Column '{color_column}' not found."
            )

    fig = px.scatter(
        df,
        x=x_column,
        y=y_column,
        color=color_column,
        title=f"{y_column} vs {x_column}",
        hover_data=df.columns.tolist()
    )

    return fig

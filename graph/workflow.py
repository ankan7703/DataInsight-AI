import time
from functools import wraps
from langgraph.graph import StateGraph, START, END

from graph.state import DataInsightState

from agents.explorer import explore_dataset
from agents.planner import create_analysis_plan
from agents.analyst import execute_analysis_plan
from agents.insight_writer import (
    generate_direct_answer,
    generate_insights
)

# ---------------------------------------
# Execution Time Logger
# ---------------------------------------

def log_execution_time(func):

    @wraps(func)
    def wrapper(state):

        start_time = time.perf_counter()

        try:
            return func(state)

        finally:
            elapsed_time = time.perf_counter() - start_time

            print(
                f"[TIMING] {func.__name__}: "
                f"{elapsed_time:.2f} seconds"
            )

    return wrapper


# ---------------------------------------
# 1. Explorer Node
# ---------------------------------------

@log_execution_time
def explorer_node(state: DataInsightState):

    print("Running Explorer Agent...")

    try:
        report = explore_dataset(state["df"])

        return {
            "exploration_report": report
        }

    except Exception as error:

        return {
            "errors": state.get("errors", [])
            + [f"Explorer Agent: {str(error)}"]
        }


# ---------------------------------------
# 2. Planner Node
# ---------------------------------------

@log_execution_time
def planner_node(state: DataInsightState):

    print("Running Planner Agent...")

    try:
        plan = create_analysis_plan(
            state["user_request"],
            state["exploration_report"]
        )

        return {
            "analysis_plan": plan
        }

    except Exception as error:

        return {
            "errors": state.get("errors", [])
            + [f"Planner Agent: {str(error)}"]
        }


# ---------------------------------------
# 3. Analyst Node
# ---------------------------------------

@log_execution_time
def analyst_node(state: DataInsightState):

    print("Running Analyst Agent...")

    try:
        results = execute_analysis_plan(
            state["df"],
            state["analysis_plan"]
        )

        return {
            "analysis_results": results
        }

    except Exception as error:

        return {
            "errors": state.get("errors", [])
            + [f"Analyst Agent: {str(error)}"]
        }


# ---------------------------------------
# 4. Quick Answer Node
# ---------------------------------------

@log_execution_time
def quick_answer_node(state: DataInsightState):

    print("Generating Quick Answer...")

    try:
        answer = generate_direct_answer(
            state["user_request"],
            state["analysis_results"]
        )

        return {
            "direct_answer": answer
        }

    except Exception as error:

        return {
            "errors": state.get("errors", [])
            + [f"Quick Answer: {str(error)}"]
        }


# ---------------------------------------
# 5. Detailed Insight Writer Node
# ---------------------------------------

@log_execution_time
def insight_writer_node(state: DataInsightState):

    print("Running Detailed Insight Writer...")

    try:
        report = generate_insights(
            state["analysis_results"]
        )

        return {
            "insight_report": report
        }

    except Exception as error:

        return {
            "errors": state.get("errors", [])
            + [f"Insight Writer Agent: {str(error)}"]
        }


# ---------------------------------------
# 6. Conditional Routing
# ---------------------------------------

def route_after_explorer(state: DataInsightState):

    if state.get("errors") or not state.get("exploration_report"):
        print("Explorer failed. Stopping workflow.")
        return "stop"

    return "continue"


def route_after_planner(state: DataInsightState):

    if state.get("errors") or not state.get("analysis_plan"):
        print("Planner failed. Stopping workflow.")
        return "stop"

    return "continue"


def route_after_analyst(state: DataInsightState):

    analysis_results = state.get("analysis_results", {})

    if state.get("errors"):
        print("Analyst Agent encountered an error. Stopping.")
        return "stop"

    if not analysis_results:
        print("No analysis results available. Stopping.")
        return "stop"

    if analysis_results.get("status") == "failed":
        print("All analysis tasks failed. Stopping workflow.")
        return "stop"

    return "continue"


def route_after_quick_answer(state: DataInsightState):

    if state.get("errors") or not state.get("direct_answer"):
        print("Quick Answer generation failed. Stopping workflow.")
        return "stop"

    if state.get("include_detailed_report", False):
        print("Detailed report requested.")
        return "detailed_report"

    print("Quick Answer generated. Skipping detailed report.")
    return "finish"


# ---------------------------------------
# 7. Build LangGraph Workflow
# ---------------------------------------

def build_workflow():

    workflow = StateGraph(DataInsightState)

    # Register nodes
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("quick_answer", quick_answer_node)
    workflow.add_node("insight_writer", insight_writer_node)

    # Explorer
    workflow.add_edge(START, "explorer")

    workflow.add_conditional_edges(
        "explorer",
        route_after_explorer,
        {
            "continue": "planner",
            "stop": END
        }
    )

    # Planner
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "continue": "analyst",
            "stop": END
        }
    )

    # Analyst
    workflow.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "continue": "quick_answer",
            "stop": END
        }
    )

    # Quick Answer
    workflow.add_conditional_edges(
        "quick_answer",
        route_after_quick_answer,
        {
            "detailed_report": "insight_writer",
            "finish": END,
            "stop": END
        }
    )

    # Detailed report
    workflow.add_edge("insight_writer", END)

    return workflow.compile()


# ---------------------------------------
# 8. Execute Workflow
# ---------------------------------------

def run_workflow(
    df,
    user_request,
    include_detailed_report=False
):

    app = build_workflow()

    initial_state = {
        "df": df,
        "user_request": user_request,
        "include_detailed_report": include_detailed_report,
        "exploration_report": {},
        "analysis_plan": {},
        "analysis_results": {},
        "direct_answer": "",
        "insight_report": {},
        "errors": []
    }

    start_time = time.perf_counter()

    result = app.invoke(initial_state)

    total_time = time.perf_counter() - start_time

    print("\n========== WORKFLOW TIMING ==========")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("=====================================\n")

    return result
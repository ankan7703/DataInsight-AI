
from langgraph.graph import StateGraph, START, END

from graph.state import DataInsightState

from agents.explorer import explore_dataset
from agents.planner import create_analysis_plan
from agents.analyst import execute_analysis_plan
from agents.insight_writer import generate_insights


# ---------------------------------------
# 1. Explorer Node
# ---------------------------------------

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
# 4. Insight Writer Node
# ---------------------------------------

def insight_writer_node(state: DataInsightState):

    print("Running Insight Writer Agent...")

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
# 5. Conditional Routing
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
        print("Analyst Agent encountered a workflow error. Stopping.")
        return "stop"

    if not analysis_results:
        print("No analysis results available. Stopping.")
        return "stop"

    if analysis_results.get("status") == "failed":
        print("All analysis tasks failed. Stopping workflow.")
        return "stop"

    return "continue"


# ---------------------------------------
# 6. Build LangGraph Workflow
# ---------------------------------------

def build_workflow():

    workflow = StateGraph(DataInsightState)

    # Register nodes
    workflow.add_node("explorer", explorer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("insight_writer", insight_writer_node)

    # Explorer routing
    workflow.add_edge(START, "explorer")

    workflow.add_conditional_edges(
        "explorer",
        route_after_explorer,
        {
            "continue": "planner",
            "stop": END
        }
    )

    # Planner routing
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "continue": "analyst",
            "stop": END
        }
    )

    # Analyst routing
    workflow.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "continue": "insight_writer",
            "stop": END
        }
    )

    # Final node
    workflow.add_edge("insight_writer", END)

    return workflow.compile()


# ---------------------------------------
# 7. Execute Workflow
# ---------------------------------------

def run_workflow(df, user_request):

    app = build_workflow()

    initial_state = {
        "df": df,
        "user_request": user_request,
        "exploration_report": {},
        "analysis_plan": {},
        "analysis_results": {},
        "insight_report": {},
        "errors": []
    }

    return app.invoke(initial_state)

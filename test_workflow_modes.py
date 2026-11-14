
from graph.workflow import route_after_quick_answer


# Simulate a successful Quick Answer
base_state = {
    "errors": [],
    "direct_answer": "Clothing generates the highest total purchase amount.",
    "include_detailed_report": False
}


# Test 1: Detailed report disabled
result = route_after_quick_answer(base_state)

print("\nTEST 1: Quick Answer Mode")
print("Routing result:", result)

assert result == "finish"


# Test 2: Detailed report enabled
detailed_state = {
    **base_state,
    "include_detailed_report": True
}

result = route_after_quick_answer(detailed_state)

print("\nTEST 2: Detailed Report Mode")
print("Routing result:", result)

assert result == "detailed_report"


# Test 3: Quick Answer generation failed
failed_state = {
    **base_state,
    "direct_answer": "",
    "include_detailed_report": True
}

result = route_after_quick_answer(failed_state)

print("\nTEST 3: Quick Answer Failure")
print("Routing result:", result)

assert result == "stop"


print("\nALL WORKFLOW ROUTING TESTS PASSED!")

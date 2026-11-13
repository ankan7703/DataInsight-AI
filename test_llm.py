
from utils.llm import get_llm

llm = get_llm()

print("Gemini initialized successfully.")

response = llm.invoke(
    "Explain the role of an AI data analyst in two sentences."
)

print("\nGEMINI RESPONSE:")

if isinstance(response.content, list):
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(block["text"])
else:
    print(response.content)

print("\nLANGCHAIN GEMINI TEST PASSED!")

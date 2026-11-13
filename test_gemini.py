
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env")

print("1. API key loaded successfully.", flush=True)

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-4-hydrogen",
    google_api_key=api_key
)

print("2. Gemini model initialized.", flush=True)

# Send test request
print("3. Sending request to Gemini...", flush=True)

try:
    response = llm.invoke(
        "Explain data analysis in exactly two sentences."
    )

    print("4. Response received!", flush=True)
    print("\nGEMINI RESPONSE:")

    for block in response.content:
        if block.get("type") == "text":
            print(block["text"])

    print("\nGEMINI API CONNECTION SUCCESSFUL!")

except Exception as error:
    print("\nGEMINI REQUEST FAILED:")
    print(type(error).__name__)
    print(str(error))

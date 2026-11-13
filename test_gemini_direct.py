
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# 1. Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")

print("API key loaded.", flush=True)


# 2. Initialize Gemini client
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=30000)
)

print("Gemini client initialized.", flush=True)


# 3. Retry Gemini request
try:

    for attempt in range(1, 4):

        print(f"\nAttempt {attempt}/3...", flush=True)

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=(
                    "Explain data analysis in exactly "
                    "two sentences."
                )
            )

            print("\nGEMINI RESPONSE:")
            print(response.text)

            print("\nGEMINI API CONNECTION SUCCESSFUL!")
            break

        except Exception as error:

            print(
                type(error).__name__,
                str(error),
                flush=True
            )

            if attempt < 3:

                wait_time = attempt * 10

                print(
                    f"Retrying in {wait_time} seconds...",
                    flush=True
                )

                time.sleep(wait_time)

            else:

                print(
                    "\nAll attempts failed. "
                    "Gemini may be temporarily unavailable."
                )

finally:
    client.close()

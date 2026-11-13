
from langchain_google_genai import ChatGoogleGenerativeAI

from config import GOOGLE_API_KEY, GEMINI_MODEL


def get_llm():
    """Initialize and return the Gemini language model."""

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Check your .env file."
        )

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

    return llm

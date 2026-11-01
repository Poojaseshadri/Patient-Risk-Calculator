import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GENAI_API_KEY")

client = genai.Client(api_key=api_key) if api_key else None


def chat_bot(prompt):
    if client is None:
        return "Gemini API key is missing. Please configure GENAI_API_KEY."

    try:
        response = client.models.generate_content(
           model="gemini-3.8-flash",
            contents=prompt
        )
        return response.text or "No response generated."

    except Exception as e:
        return f"Gemini API error: {str(e)}"
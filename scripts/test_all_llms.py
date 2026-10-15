import os

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
import anthropic


load_dotenv()


QUESTION = "What is the purpose of bail in criminal proceedings?"


# ============================================================
# GEMINI
# ============================================================

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "ERROR: GEMINI_API_KEY not found"

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=QUESTION
        )

        return response.text

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ============================================================
# OPENAI
# ============================================================

def test_openai():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return "ERROR: OPENAI_API_KEY not found"

    try:
        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model="gpt-5-mini",
            input=QUESTION
        )

        return response.output_text

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ============================================================
# ANTHROPIC
# ============================================================

def test_anthropic():
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY not found"

    try:
        client = anthropic.Anthropic(api_key=api_key)

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": QUESTION
                }
            ]
        )

        return response.content[0].text

    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("MULTI-PROVIDER LLM API TEST")
print("=" * 70)

print("\nQuestion:")
print(QUESTION)


print("\n" + "=" * 70)
print("1. GEMINI")
print("=" * 70)

gemini_result = test_gemini()
print(gemini_result)


print("\n" + "=" * 70)
print("2. OPENAI")
print("=" * 70)

openai_result = test_openai()
print(openai_result)


print("\n" + "=" * 70)
print("3. ANTHROPIC")
print("=" * 70)

anthropic_result = test_anthropic()
print(anthropic_result)


print("\n" + "=" * 70)
print("TEST COMPLETED")
print("=" * 70)
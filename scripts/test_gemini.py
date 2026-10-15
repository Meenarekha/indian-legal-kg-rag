import os
import time

from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found.\n"
        "Make sure your .env file contains:\n"
        "GEMINI_API_KEY=your_key"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(api_key=api_key)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-3.8-flash"

QUESTION = """
What is the purpose of bail in criminal proceedings?

Give a concise answer in 3 to 5 sentences.
Do not provide legal advice.
"""


# ============================================================
# GEMINI REQUEST WITH RETRY
# ============================================================

def generate_with_retry(max_retries=3, base_delay=5):

    for attempt in range(1, max_retries + 1):

        try:

            print(f"\nAttempt {attempt}/{max_retries}...")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=QUESTION
            )

            return response.text

        except Exception as e:

            error_text = str(e)

            print("\nRequest failed:")
            print(error_text)

            # Retry temporary server/rate-limit errors
            if "503" in error_text or "UNAVAILABLE" in error_text:
                if attempt < max_retries:

                    delay = base_delay * (2 ** (attempt - 1))

                    print(
                        f"\nGemini is temporarily unavailable."
                        f"\nRetrying in {delay} seconds..."
                    )

                    time.sleep(delay)

                else:
                    raise RuntimeError(
                        "Gemini remained unavailable after "
                        f"{max_retries} attempts."
                    )

            elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                if attempt < max_retries:

                    delay = base_delay * (2 ** (attempt - 1))

                    print(
                        f"\nGemini rate limit encountered."
                        f"\nRetrying in {delay} seconds..."
                    )

                    time.sleep(delay)

                else:
                    raise RuntimeError(
                        "Gemini rate limit persisted after "
                        f"{max_retries} attempts."
                    )

            else:
                # Authentication, invalid request, etc.
                # should not be repeatedly retried.
                raise


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GEMINI API CONNECTION TEST")
    print("=" * 70)

    print(f"\nModel: {MODEL_NAME}")

    print("\nQuestion:")
    print(QUESTION.strip())

    try:

        answer = generate_with_retry()

        print("\n" + "=" * 70)
        print("SUCCESS")
        print("=" * 70)

        print("\nGemini response:")
        print(answer)

        print("\n" + "=" * 70)
        print("Gemini API connection is working.")
        print("=" * 70)

    except Exception as e:

        print("\n" + "=" * 70)
        print("FAILED")
        print("=" * 70)

        print(f"\nError type: {type(e).__name__}")
        print(f"Error: {e}")

        print("\nThe Gemini API test was not successful.")


if __name__ == "__main__":
    main()
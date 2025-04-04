import os
from openai import OpenAI, APIConnectionError, AuthenticationError
from dotenv import load_dotenv

def validate_llm_connection():
    """
    Validates the connection to the OpenAI API endpoint.

    Loads API base URL and key from environment variables (expected in .env file)
    and attempts a simple chat completion request.
    """
    print("Attempting to validate LLM connection...")

    # Load environment variables from .env file
    load_dotenv()
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_base:
        print("Error: OPENAI_API_BASE not found in environment variables.")
        print("Please ensure it's set in your .env file.")
        return False
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please ensure it's set in your .env file.")
        return False

    print(f"Using API Base: {api_base}") # Be cautious about logging keys

    try:
        # Initialize the OpenAI client
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=10.0, # Add a timeout
        )

        # Attempt a simple API call (adjust model if needed)
        print("Sending a test request to the LLM...")
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Or your default available model
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=5
        )

        print("LLM connection successful!")
        print(f"Received response: {response.choices[0].message.content}")
        return True

    except AuthenticationError:
        print("Error: Authentication failed. Check your API key.")
        return False
    except APIConnectionError as e:
        print(f"Error: Could not connect to API endpoint at {api_base}.")
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    if validate_llm_connection():
        print("\nLLM Validation Passed.")
    else:
        print("\nLLM Validation Failed.") 
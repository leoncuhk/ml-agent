# src/ml_agent/llm_interface.py
import os
import logging
from openai import OpenAI, APIConnectionError, AuthenticationError
from dotenv import load_dotenv

logger = logging.getLogger('ml_agent') # Use the agent's logger

# --- LLM Client Setup ---
def initialize_llm_client():
    """Initializes and returns the OpenAI client using .env variables."""
    load_dotenv()
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_base or not api_key:
        logger.error("OpenAI API Base URL or Key not found in .env file.")
        raise ValueError("Missing OpenAI credentials in .env file")

    try:
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=20.0,
        )
        client.models.list() # Test connection
        logger.info(f"OpenAI client initialized successfully. Connected to: {api_base}")
        return client
    except AuthenticationError:
        logger.error("OpenAI Authentication failed. Check API key.")
        raise
    except APIConnectionError as e:
        logger.error(f"Failed to connect to OpenAI API at {api_base}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI client initialization: {e}")
        raise 
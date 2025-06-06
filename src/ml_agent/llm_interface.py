# src/ml_agent/llm_interface.py
import os
import logging
from openai import OpenAI # Use official OpenAI library
from dotenv import load_dotenv

logger = logging.getLogger('ml_agent') # Use the agent's logger

# --- LLM Client Setup ---
def initialize_llm_client():
    """Initializes and returns the OpenAI client."""
    load_dotenv()
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini") # Allow model override

    if not api_base or not api_key:
        logger.error("OpenAI API Base URL or Key not found in .env file.")
        raise ValueError("Missing OpenAI credentials in .env file")

    try:
        # Use the official OpenAI client
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=20.0,
        )
        # Test connection by creating a simple completion
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}]
        )
        logger.info(f"OpenAI client initialized successfully for model '{model_name}'.")
        return client
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM client initialization: {e}")
        raise 
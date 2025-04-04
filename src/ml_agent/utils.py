import os
import logging
import re

# --- Logging Setup ---
def setup_logging(log_path="logs/", log_level=logging.INFO):
    """Sets up logging configuration."""
    os.makedirs(log_path, exist_ok=True)
    log_filename = os.path.join(log_path, 'ml_agent.log')
    # Get root logger to configure handlers
    logger = logging.getLogger() # Get root logger
    logger.setLevel(log_level) # Set level on root logger

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_filename)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream Handler (Console)
        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s') # Simpler format for console
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
    return logging.getLogger('ml_agent') # Return specific logger for the agent module


# --- Parameter Parsing ---
def parse_llm_params(recommendations: str, logger: logging.Logger) -> dict:
    """Parses the LLM recommendation string to extract potential H2O parameters."""
    parsed_params = {}
    patterns = {
        'max_runtime_secs': r"max_runtime_secs\s*[:=]\s*(\d+)",
        'max_models': r"max_models\s*[:=]\s*(\d+)",
        'exclude_algos': r"exclude_algos\s*[:=]\s*(\[.+?\]|\".+?\"|'.+?')", 
        'sort_metric': r"sort_metric\s*[:=]\s*([\'\"]?\w+[\'\"]?)",
        'balance_classes': r"balance_classes\s*[:=]\s*(True|False)",
        'nfolds': r"nfolds\s*[:=]\s*(\d+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, recommendations, re.IGNORECASE)
        if match:
            value_str = match.group(1).strip()
            try:
                if key in ['max_runtime_secs', 'max_models', 'nfolds']:
                    parsed_params[key] = int(value_str)
                elif key == 'exclude_algos':
                    algos_str = value_str.strip('[] "\'')
                    parsed_params[key] = [algo.strip().strip('"\'') for algo in algos_str.split(',')]
                elif key == 'balance_classes':
                        parsed_params[key] = value_str.lower() == 'true'
                elif key == 'sort_metric':
                        parsed_params[key] = value_str.strip('\'"')
                else:
                    parsed_params[key] = value_str
                logger.info(f"Parsed LLM suggestion: {key}={parsed_params[key]}")
            except Exception as e:
                logger.warning(f"Could not parse value for {key} from LLM suggestion '{value_str}': {e}")
    return parsed_params 
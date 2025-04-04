import pandas as pd
import h2o
import os
import time
import random
from datetime import datetime
import json
from openai import OpenAI, APIConnectionError, AuthenticationError
import logging
from dotenv import load_dotenv
import io
import re

# Imports from other modules within this package
from .llm_interface import initialize_llm_client
from .h2o_executor import run_h2o_automl
from .utils import setup_logging, parse_llm_params

# LLM调用函数，参考yunwu.py
def chat_with_llm(prompt, max_retries=3):
    """
    与LLM模型对话，包含重试机制
    :param prompt: 用户输入
    :param max_retries: 最大重试次数
    :return: 模型回复
    """
    # 创建客户端
    client = OpenAI(
        # 可以根据需要替换为其他API端点和密钥
        base_url="https://yunwu.ai/v1",
        api_key="sk-kZpsgjS8XplmWbO0VO4RBPBujvHpl30erAXestY8CmbLygel"
    )

    for attempt in range(max_retries):
        try:
            # 添加随机延迟，避免并发问题
            if attempt > 0:
                delay = random.uniform(1, 3)
                print(f"等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
            
            print(f"正在发送请求... (尝试 {attempt + 1}/{max_retries})")
            
            response = client.chat.completions.create(
                # model="deepseek-v3",
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.7,
                max_tokens=2000
            )
            
            # 检查是否有错误信息
            if hasattr(response, 'error'):
                error_msg = response.error.get('message', '未知错误')
                if error_msg == 'concurrency exceeded':
                    print(f"并发超限，将重试...")
                    continue
                return f"API错误: {error_msg}"
            
            # 尝试获取响应内容
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
            
            print(f"响应格式异常: {response}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"发生错误: {str(e)}")
                continue
            return f"所有重试都失败了: {str(e)}"
    
    return "达到最大重试次数，仍未获得有效响应"

# Initialize logger for this module
# Note: setup_logging configures the root logger, getLogger gets a specific one
logger = logging.getLogger('ml_agent')

# --- Standalone H2O ML Agent Class ---
class H2OMLAgent:
    """
    An agent that uses an LLM to recommend H2O AutoML steps and generates
    parameters for execution.
    """
    def __init__(self, log=True, log_path="logs/", model_directory="models/"):
        """
        Initializes the H2OMLAgent.

        Args:
            log (bool): Whether to enable logging.
            log_path (str): Path to the directory for saving logs.
            model_directory (str): Path to the directory for saving models.
        """
        self.log = log
        self.log_path = log_path
        self.model_directory = model_directory
        # Ensure directories exist
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_directory, exist_ok=True)

        # Setup logging using the utility function
        # We get the specific logger instance, but setup_logging configures the root handlers
        self.logger = setup_logging(log_path=self.log_path)
        if not self.log:
            self.logger.disabled = True
            # Also disable handlers for root logger if log=False entirely?
            # for handler in logging.getLogger().handlers[:]:
            #     logging.getLogger().removeHandler(handler)
            # logging.getLogger().addHandler(logging.NullHandler())

        self.logger.info("Initializing H2OMLAgent...")

        # Initialize LLM client using the interface function
        try:
            self.llm_client = initialize_llm_client()
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise

        # Initialize results dictionary
        self.results = {}
        self.logger.info("H2OMLAgent initialized.")

    # --- Data Description ---
    def _generate_data_description(self, df: pd.DataFrame, max_unique_values_to_list=10) -> str:
        self.logger.info(f"Generating data description for DataFrame with shape {df.shape}...")
        description = f"### Data Summary\n"
        description += f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        description += f"### Column Information (Types, Non-Null Counts):\n```\n{info_str}```\n"
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            description += f"### Missing Value Counts:\n"
            description += missing_values.to_string() + "\n\n"
        else:
            description += f"- No missing values detected.\n\n"
        description += f"### Unique Value Analysis:\n"
        for col in df.columns:
            num_unique = df[col].nunique()
            description += f"- Column '{col}': {num_unique} unique values.\n"
            if num_unique <= max_unique_values_to_list and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
                 unique_vals = df[col].unique()
                 description += f"    Unique values: {list(unique_vals)}\n"
        try:
            numerical_stats = df.describe().to_string()
            description += f"\n### Numerical Column Statistics:\n```\n{numerical_stats}```\n"
        except Exception as e:
             self.logger.warning(f"Could not generate numerical statistics (likely no numeric columns): {e}")
             description += f"\n- No numerical columns found for descriptive statistics.\n"
        self.logger.debug(f"Generated Data Description:\n{description}")
        return description

    # --- LLM Recommendations ---
    def _get_llm_recommendations(self, data_description: str, user_instructions: str, target_variable: str) -> str:
        self.logger.info("Getting LLM recommendations...")
        prompt = f"""
        As an expert Data Scientist specializing in H2O AutoML, your task is to analyze the provided data summary and user instructions to recommend the best way to approach this machine learning problem using H2O AutoML.
        **Your goal is NOT to generate Python code, but to provide:**
        1.  A brief analysis of the data and the problem type (e.g., binary classification, regression, multiclass classification).
        2.  Recommendations for key H2O AutoML parameters based on the data and instructions (e.g., suggestions for `max_runtime_secs`, `max_models`, potential `exclude_algos`, `sort_metric`, important preprocessing considerations like handling imbalanced data if apparent).
        3.  Any specific advice or warnings based on the data description (e.g., handling of high cardinality features, potential data leakage if identifiable).
        **Input Data Summary:**
        {data_description}
        **Target Variable:**
        `{target_variable}`
        **User Instructions:**
        {user_instructions}
        **Output:**
        Please provide your analysis and recommendations in a clear, concise format. Focus on actionable advice for configuring and running H2O AutoML effectively for this specific scenario.
        """
        self.logger.debug(f"Sending the following prompt to LLM:\n{prompt}")
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert Data Scientist specializing in H2O AutoML configuration."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=500
            )
            recommendations = response.choices[0].message.content
            self.logger.info("Successfully received recommendations from LLM.")
            self.logger.debug(f"LLM Raw Response:\n{recommendations}")
            return recommendations
        except Exception as e:
            self.logger.error(f"Error calling LLM for recommendations: {e}")
            return "Error: Could not get recommendations from LLM."

    # --- Parameter Generation ---
    def _generate_h2o_params(self, llm_recommendations: str, target_variable: str, user_preferences: dict) -> dict:
        self.logger.info("Generating H2O AutoML parameters...")
        default_params = {'seed': 1234}
        self.logger.debug(f"Default H2O parameters: {default_params}")
        
        # Use the parsing function from utils
        llm_suggested_params = parse_llm_params(llm_recommendations, self.logger)
        self.logger.debug(f"LLM suggested parameters: {llm_suggested_params}")
        
        final_params = {**default_params, **llm_suggested_params}
        self.logger.debug(f"User provided preferences: {user_preferences}")
        valid_h2o_args = [
            'max_runtime_secs', 'max_models', 'stopping_metric', 'sort_metric',
            'exclude_algos', 'include_algos', 'nfolds', 'balance_classes',
            'max_after_balance_size', 'stopping_rounds', 'stopping_tolerance',
            'seed', 'project_name'
        ]
        for key, value in user_preferences.items():
            if key in valid_h2o_args:
                if value is not None:
                     self.logger.info(f"Applying user preference: {key}={value} (overriding previous value)")
                     final_params[key] = value
                else:
                    self.logger.debug(f"Ignoring user preference {key}=None")
            else:
                 self.logger.warning(f"Ignoring unknown user preference: '{key}'. Not a recognized H2O AutoML argument.")
        if 'y' in final_params: del final_params['y']
        if 'target' in final_params: del final_params['target']
        if 'target_variable' in final_params: del final_params['target_variable']
        self.logger.info(f"Final H2O AutoML parameters generated: {final_params}")
        return final_params

    # --- H2O Execution ---
    def _run_h2o_automl(self, data_raw: pd.DataFrame, target_variable: str, h2o_params: dict):
        self.logger.info("--- Starting H2O AutoML Execution ---")
        results = {"leaderboard": None, "best_model_id": None, "best_model_path": None, "error": None}
        try:
            self.logger.info("Initializing H2O cluster...")
            h2o.init()
            self.logger.info("H2O initialized.")
            self.logger.info("Converting data to H2O Frame...")
            h2o_df = h2o.H2OFrame(data_raw)
            self.logger.info("H2O Frame created.")
            y = target_variable
            x = h2o_df.columns
            x.remove(y)
            self.logger.info(f"Target variable (y): {y}")
            self.logger.info(f"Predictor variables (x): {x}")
            
            # Determine problem type more robustly
            problem_type = None
            target_col = h2o_df[y]
            num_unique = target_col.nunique()
            is_numeric = target_col.isnumeric()[0]
            is_factor = target_col.isfactor()[0]
            # Explicit check for small number of unique integers, likely classification
            is_int = target_col.types[y] == 'int'
            
            self.logger.info(f"Target '{y}' info: unique={num_unique}, numeric={is_numeric}, factor={is_factor}, type={target_col.types[y]}")
            
            # Heuristic: If low unique count and not float, treat as classification
            if num_unique <= 10 and target_col.types[y] != 'real': 
                problem_type = "classification"
                if not is_factor:
                    self.logger.info(f"Target '{y}' has {num_unique} unique values (and is not float). Converting to factor for classification.")
                    h2o_df[y] = target_col.asfactor()
                else:
                     self.logger.info(f"Target '{y}' is already a factor with {num_unique} levels.")
            elif is_numeric: # If it's numeric (and not caught by above), assume regression
                problem_type = "regression"
                self.logger.info(f"Target '{y}' is numeric. Treating as regression.")
            else: # Fallback if unclear (e.g., string IDs), might need user clarification or error
                 self.logger.warning(f"Could not reliably determine problem type for target '{y}'. AutoML might fail.")
                 # Optionally, default to classification if factor
                 if is_factor:
                      problem_type = "classification"
                      self.logger.warning(f"Target '{y}' is factor, defaulting to classification.")
                      
            # Override sort_metric if needed based on problem type
            if problem_type == "regression" and h2o_params.get('sort_metric') == 'AUC':
                self.logger.warning("Sort metric AUC is invalid for regression. Removing it to use H2O default.")
                h2o_params.pop('sort_metric', None)
            elif problem_type == "classification" and h2o_params.get('sort_metric') == 'RMSE': # Example
                 self.logger.warning("Sort metric RMSE is unusual for classification. Keeping user preference but check results.")

            self.logger.info(f"Starting H2O AutoML for inferred problem type: {problem_type}")
            self.logger.info(f"Using H2O AutoML parameters: {h2o_params}")

            aml = h2o.automl.H2OAutoML(**h2o_params)
            aml.train(x=x, y=y, training_frame=h2o_df)
            self.logger.info("H2O AutoML training complete.")
            self.logger.info("Fetching AutoML leaderboard...")
            lb = aml.leaderboard
            results["leaderboard"] = lb.as_data_frame()
            self.logger.debug(f"Leaderboard:\n{results['leaderboard']}")
            if aml.leader:
                results["best_model_id"] = aml.leader.model_id
                self.logger.info(f"Best model ID: {results['best_model_id']}")
                try:
                    model_path = h2o.save_model(model=aml.leader, path=self.model_directory, force=True)
                    results["best_model_path"] = model_path
                    self.logger.info(f"Best model saved to: {model_path}")
                except Exception as save_e:
                     self.logger.error(f"Failed to save the best model: {save_e}")
                     results["error"] = f"Training complete but failed to save model: {save_e}"
            else:
                self.logger.warning("H2O AutoML finished, but no leader model found.")
                results["error"] = "AutoML finished without a leader model."
        except Exception as e:
            self.logger.error(f"An error occurred during H2O AutoML execution: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            results["error"] = str(e)
        self.logger.info("--- H2O AutoML Execution Finished ---")
        return results

    # --- Agent Invocation ---
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str, target_variable: str, **kwargs):
        self.logger.info("--- Starting ML Agent Invocation ---")
        self.logger.info(f"User Instructions: {user_instructions}")
        self.logger.info(f"Target Variable: {target_variable}")
        self.logger.info(f"Additional Parameters: {kwargs}")
        self.results = {} # Reset results
        
        try:
            # 1. Generate Data Description
            data_desc = self._generate_data_description(data_raw)
            self.results["data_description"] = data_desc
            self.logger.info("\nData Description:")
            self.logger.info(data_desc)
            
            # 2. Get LLM Recommendations
            llm_recs = self._get_llm_recommendations(data_desc, user_instructions, target_variable)
            self.results["llm_recommendations"] = llm_recs
            self.logger.info("\nLLM Recommendations:")
            self.logger.info(llm_recs)
            
            # 3. Generate H2O AutoML Parameters
            h2o_params = self._generate_h2o_params(llm_recs, target_variable, kwargs)
            self.results["h2o_parameters"] = h2o_params
            self.logger.info("\nGenerated H2O AutoML Parameters:")
            self.logger.info(h2o_params)
            
            # 4. Execute H2O AutoML using the executor function
            self.logger.info("--- Executing H2O AutoML --- ")
            # Pass necessary args like model_directory
            h2o_results = run_h2o_automl(data_raw, target_variable, self.model_directory, h2o_params)
            
            # 5. Update Results Store
            self.results.update(h2o_results) # Directly update from the results dict
            
            if self.results.get("error"): 
                self.logger.error(f"H2O execution failed: {self.results['error']}")
            else:
                 self.logger.info("H2O AutoML execution finished.") # Don't assume success, check error
                 if not self.results.get("best_model_id") and not self.results.get("error"):
                      self.logger.warning("H2O AutoML finished without error, but no leader model found.")
                 elif self.results.get("best_model_id"):
                      self.logger.info("H2O AutoML execution successful (leader found).")
                      
        except Exception as e:
             self.logger.error(f"An unexpected error occurred during agent invocation: {e}", exc_info=True)
             self.results["error"] = f"Agent invocation failed: {e}"
             
        self.logger.info("--- ML Agent Invocation Finished ---")
        return self.results

    # --- Result Reporting ---
    def get_leaderboard(self) -> pd.DataFrame | None:
        """Returns the H2O AutoML leaderboard as a Pandas DataFrame."""
        if hasattr(self, 'results') and self.results and 'leaderboard' in self.results:
            lb = self.results['leaderboard']
            if isinstance(lb, pd.DataFrame):
                return lb
            else:
                self.logger.warning("Stored leaderboard is not a Pandas DataFrame.")
                return None
        else:
            self.logger.warning("Leaderboard not available. Run invoke_agent first or check for errors.")
            return None

    def get_workflow_summary(self, markdown=False) -> str:
        """Generates a summary of the agent's workflow and results."""
        if not hasattr(self, 'results') or not self.results:
            return "Workflow has not been run yet or no results available."

        summary = "### ML Agent Workflow Summary\n\n"
        
        # Safely extract parts of the data description
        data_desc = self.results.get('data_description', '')
        desc_lines = data_desc.split('\n')
        shape_line = next((line for line in desc_lines if 'Shape:' in line), '- Shape: N/A')
        summary += "**1. Data Analysis:**\n"
        summary += f"> {shape_line.strip()}\n\n"

        summary += "**2. LLM Recommendations:**\n"
        recs = self.results.get('llm_recommendations', 'N/A')
        # Truncate long recommendations for summary
        recs_summary = (recs[:200] + '...') if len(recs) > 200 else recs 
        if markdown:
            summary += f"> ```\n> {recs_summary}\n> ```\n\n" # Show truncated version in summary
        else:
             summary += f"> {recs_summary}\n\n"

        summary += "**3. H2O AutoML Configuration:**\n"
        params_str = json.dumps(self.results.get('h2o_parameters', {}), indent=2)
        if markdown:
            summary += f"> ```json\n> {params_str}\n> ```\n\n"
        else:
            summary += f"> {params_str}\n\n"

        summary += "**4. H2O AutoML Execution:**\n"
        if self.results.get('error'):
            summary += f"> Status: **Failed**\n> Error: {self.results['error']}\n\n"
        elif self.results.get('best_model_id'):
            summary += f"> Status: **Completed Successfully**\n"
            summary += f"> Best Model ID: {self.results['best_model_id']}\n"
            summary += f"> Best Model Path: {self.results['best_model_path']}\n"
            leaderboard = self.get_leaderboard()
            if leaderboard is not None and not leaderboard.empty:
                # Try to dynamically find metric columns (often first few after model_id)
                metric_col = leaderboard.columns[1] if len(leaderboard.columns) > 1 else 'Metric'
                summary += f"> Leader Model Performance ({metric_col}): {leaderboard.iloc[0, 1]:.4f}\n"
            summary += "\n"
        elif 'leaderboard' in self.results and self.results['leaderboard'] is not None:
             summary += "> Status: Completed, but no leader model found (leaderboard might be empty or only contain failed models).\n\n"
        else:
            summary += "> Status: Unknown (Likely failed before generating leaderboard).\n\n"

        return summary

# --- Example Usage (REMOVE THIS - Moved to run_agent.py) ---
# if __name__ == "__main__":
#    ... (old test code removed) 
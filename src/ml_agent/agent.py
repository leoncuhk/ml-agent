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
from . import data_preparer # Import the new module

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
    def _generate_data_description(self, df: pd.DataFrame, max_unique_values_to_list=10, high_cardinality_threshold=50, iqr_multiplier=1.5) -> str:
        self.logger.info(f"Generating data description for DataFrame with shape {df.shape}...")
        description = f"### Data Summary\\n"
        description += f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns\\n\\n"

        # Basic Info
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        description += f"### Column Information (Types, Non-Null Counts):\\n```\\n{info_str}```\\n"

        # Missing Values
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            description += f"### Missing Value Counts:\\n```\\n{missing_values.to_string()}```\\n\\n"
        else:
            description += f"- No missing values detected.\\n\\n"

        description += f"### Feature Analysis:\\n"
        high_card_features = []
        potential_outlier_cols = []
        numeric_cols = df.select_dtypes(include=['number']).columns

        # Unique Value Analysis & High Cardinality Check
        description += f"#### Unique Values & Cardinality:\\n"
        for col in df.columns:
            num_unique = df[col].nunique()
            is_high_card = num_unique > high_cardinality_threshold
            description += f"- **\'{col}\'**: {num_unique} unique values."
            if is_high_card:
                description += f" (High Cardinality - Threshold > {high_cardinality_threshold})"
                high_card_features.append(col)
            description += "\\n"

            # List unique values only if low cardinality and categorical/object
            if num_unique <= max_unique_values_to_list and (df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col].dtype)):
                try:
                    unique_vals = df[col].unique()
                    # Limit displayed values if accidentally large list comes through
                    display_vals = list(unique_vals[:max_unique_values_to_list])
                    if len(unique_vals) > max_unique_values_to_list:
                         display_vals.append("...")
                    description += f"    - *Unique values*: {display_vals}\\n"
                except Exception as e:
                    self.logger.warning(f"Could not display unique values for column '{col}': {e}")
        if high_card_features:
            description += f"\\n*High Cardinality Columns Detected*: {high_card_features}\\n"


        # Numerical Analysis: Statistics, Skewness, Outliers
        if not numeric_cols.empty:
            description += f"\\n#### Numerical Feature Analysis:\\n"
            try:
                numerical_stats = df[numeric_cols].describe().to_string()
                description += f"##### Statistics:\\n```\\n{numerical_stats}```\\n"
            except Exception as e:
                self.logger.warning(f"Could not generate numerical statistics: {e}")

            description += f"##### Skewness:\\n"
            try:
                 skewness = df[numeric_cols].skew()
                 description += f"```\\n{skewness.to_string()}```\\n"
                 highly_skewed = skewness[abs(skewness) > 1].index.tolist()
                 if highly_skewed:
                     description += f"*Highly Skewed Columns (|skewness| > 1)*: {highly_skewed}\\n"
            except Exception as e:
                self.logger.warning(f"Could not calculate skewness: {e}")


            description += f"\\n##### Potential Outliers (Using IQR * {iqr_multiplier}):\\n"
            try:
                Q1 = df[numeric_cols].quantile(0.25)
                Q3 = df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - iqr_multiplier * IQR
                upper_bound = Q3 + iqr_multiplier * IQR

                outlier_counts = {}
                for col in numeric_cols:
                     outliers = df[(df[col] < lower_bound[col]) | (df[col] > upper_bound[col])]
                     count = outliers.shape[0]
                     if count > 0:
                         outlier_counts[col] = count
                         potential_outlier_cols.append(col)

                if outlier_counts:
                    description += f"```\\n"
                    for col, count in outlier_counts.items():
                         description += f"- {col}: {count} potential outliers\\n"
                    description += f"```\\n"
                    description += f"*Columns with Potential Outliers*: {potential_outlier_cols}\\n"
                else:
                    description += "- No potential outliers detected based on IQR method.\\n"
            except Exception as e:
                self.logger.warning(f"Could not perform IQR outlier analysis: {e}")
        else:
             description += f"\\n- No numerical columns found for detailed analysis (Statistics, Skewness, Outliers).\\n"

        self.logger.debug(f"Generated Data Description:\\n{description}")
        return description

    # --- LLM Data Preparation Planning --- NEW METHOD ---
    def _get_llm_data_prep_plan(self, data_description: str, target_variable: str) -> list | None:
        """
        Prompts the LLM to generate a structured data preparation plan (list of steps).
        """
        self.logger.info("Getting LLM data preparation plan...")
        prompt = f"""
        As an expert Data Scientist, analyze the provided comprehensive data summary and target variable (`{target_variable}`) to devise a data preparation plan.
        Your goal is to suggest specific steps to clean and prepare the data for modeling using the available operations: "impute_missing", "encode_categorical", "handle_outliers", "scale_numerical", "extract_date_features".

        **Consider the following guidelines:**
        - **Imputation:** Use "impute_missing" for columns listed with missing values. Choose a sensible strategy ('mean', 'median', 'mode').
        - **Encoding:** Use "encode_categorical" for 'object' or 'category' type columns (excluding the target variable unless imputation was needed). Prefer 'one-hot'. Consider setting "drop_original": false for high cardinality features if you think the original might still be useful alongside encoding (though usually drop=true is fine).
        - **Outliers:** If the summary indicates '*Columns with Potential Outliers*', use "handle_outliers" with strategy 'clip' for those numerical columns.
        - **Scaling:** If the summary indicates '*Highly Skewed Columns*', consider using "scale_numerical" with strategy 'standard' or 'minmax' for those numerical columns *after* handling outliers. Standard scaling is generally preferred. Apply scaling only to numerical features (not the target or categorical features).
        - **Date Features:** If any column looks like a date/time object (check Column Information), use "extract_date_features".
        - **Target Variable:** Generally avoid modifying the target variable `'{{target_variable}}'`, except for necessary imputation. Do NOT scale or encode the target variable.
        - **Order:** Think about the order. Usually: Impute -> Handle Outliers -> Extract Dates -> Encode Categorical -> Scale Numerical.

        **Output ONLY a JSON list of operations.** Each object in the list should represent one step and conform to the required keys for each operation type:
        - `impute_missing`: `{{ "operation": "impute_missing", "column": "col_name", "strategy": "mean|median|mode" }}`
        - `encode_categorical`: `{{ "operation": "encode_categorical", "column": "col_name", "strategy": "one-hot", "drop_original": true|false }}`
        - `handle_outliers`: `{{ "operation": "handle_outliers", "column": "col_name", "strategy": "clip", "iqr_multiplier": 1.5 }}` (strategy/multiplier are optional, defaults shown)
        - `scale_numerical`: `{{ "operation": "scale_numerical", "column": "col_name", "strategy": "standard|minmax" }}` (strategy optional, default 'standard')
        - `extract_date_features`: `{{ "operation": "extract_date_features", "column": "col_name", "drop_original": true|false }}`

        Example JSON Output:
        [
            {{"operation": "impute_missing", "column": "Age", "strategy": "median"}},
            {{"operation": "handle_outliers", "column": "Salary", "strategy": "clip"}},
            {{"operation": "extract_date_features", "column": "RegistrationDate"}},
            {{"operation": "encode_categorical", "column": "Gender", "strategy": "one-hot"}},
            {{"operation": "scale_numerical", "column": "Salary", "strategy": "standard"}}
        ]

        **Input Data Summary:**
        {data_description}

        **Target Variable:** `{target_variable}`

        **Instructions:**
        Generate the JSON list of preparation steps based *only* on the data summary and the guidelines above. Be concise and focus on necessary steps identified in the summary. Ensure the output is valid JSON. Output *only* the JSON list, without any introductory text or explanations.
        """
        self.logger.debug(f"Sending the following prompt to LLM for data prep plan:\n{prompt}")
        try:
            # Using the existing llm_client setup in __init__
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini", # Or your preferred model
                messages=[
                    {"role": "system", "content": "You are a data preparation planner. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, # Lower temperature for more deterministic, structured output
                max_tokens=1000,
                response_format={{"type": "json_object"}} # Request JSON output if API supports
            )
            raw_response = response.choices[0].message.content
            self.logger.debug(f"LLM Raw Data Prep Plan Response:\n{raw_response}")

            # Clean potential markdown ```json ... ``` tags
            match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_response, re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
            else:
                json_str = raw_response.strip()

            # Attempt to parse the JSON string directly
            prep_plan = json.loads(json_str)

            # Validate basic structure (is it a list?)
            if not isinstance(prep_plan, list):
                 # Sometimes the response might be nested under a key like "steps"
                 if isinstance(prep_plan, dict) and "steps" in prep_plan and isinstance(prep_plan["steps"], list):
                    prep_plan = prep_plan["steps"]
                 else:
                    self.logger.error(f"LLM response for data prep plan is not a list: {prep_plan}")
                    return None

            self.logger.info(f"Successfully received and parsed data preparation plan: {len(prep_plan)} steps.")
            return prep_plan
        except json.JSONDecodeError as json_err:
            self.logger.error(f"Failed to decode JSON response from LLM for data prep plan: {json_err}")
            self.logger.error(f"LLM Raw Response was: {raw_response}")
            return None
        except Exception as e:
            self.logger.error(f"Error calling LLM for data prep plan: {e}")
            return None

    # --- Execute Data Preparation Plan --- NEW METHOD ---
    def _execute_data_prep_plan(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        """
        Executes the data preparation steps defined in the plan.
        """
        if not plan:
            self.logger.info("No data preparation plan provided or plan is empty. Skipping execution.")
            return df

        self.logger.info(f"Executing data preparation plan with {len(plan)} steps...")
        current_df = df.copy() # Work on a copy

        for i, step in enumerate(plan):
            self.logger.info(f"Step {i+1}/{len(plan)}: {step}")
            operation = step.get("operation")
            column = step.get("column")

            if not operation:
                 self.logger.warning(f"Skipping invalid step (missing 'operation'): {step}")
                 continue
            # Column might not be needed for all ops, but most require it
            if not column and operation not in []: # Add ops that don't need columns here
                self.logger.warning(f"Skipping invalid step (missing 'column' for required op '{operation}'): {step}")
                continue

            try:
                if operation == "impute_missing":
                    strategy = step.get("strategy")
                    if not strategy:
                        self.logger.warning(f"Missing 'strategy' for impute_missing on column '{column}'. Skipping.")
                        continue
                    current_df = data_preparer.impute_missing(current_df, column, strategy)

                elif operation == "encode_categorical":
                    strategy = step.get("strategy")
                    if not strategy:
                        self.logger.warning(f"Missing 'strategy' for encode_categorical on column '{column}'. Skipping.")
                        continue
                    drop_original = step.get("drop_original", True)
                    current_df = data_preparer.encode_categorical(current_df, column, strategy, drop_original)

                elif operation == "handle_outliers": # NEW
                    strategy = step.get("strategy", "clip") # Default strategy
                    iqr_multiplier = step.get("iqr_multiplier", 1.5)
                    current_df = data_preparer.handle_outliers(current_df, column, strategy, iqr_multiplier)

                elif operation == "scale_numerical": # NEW
                    strategy = step.get("strategy", "standard") # Default strategy
                    current_df = data_preparer.scale_numerical(current_df, column, strategy)

                elif operation == "extract_date_features": # NEW
                    drop_original = step.get("drop_original", True)
                    current_df = data_preparer.extract_date_features(current_df, column, drop_original)

                else:
                    self.logger.warning(f"Unsupported operation '{operation}' in step: {step}. Skipping.")

            except Exception as e:
                 self.logger.error(f"Error executing step {i+1} ({step}): {e}", exc_info=True)
                 self.logger.warning("Continuing with the next preparation step despite the error.")

        self.logger.info("Finished executing data preparation plan.")
        return current_df

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

    # --- Main Agent Invocation --- MODIFIED ---
    def invoke_agent(self, data_raw: pd.DataFrame, user_instructions: str, target_variable: str, **kwargs):
        """
        Runs the full ML agent workflow: data description, LLM planning for data prep,
        data prep execution, (future: feature eng), LLM config for H2O, H2O execution.
        """
        start_time = time.time()
        self.logger.info("--- Starting ML Agent Invocation ---")
        self.results = {} # Clear previous results

        if not isinstance(data_raw, pd.DataFrame):
            self.logger.error("Input data_raw must be a pandas DataFrame.")
            raise TypeError("Input data_raw must be a pandas DataFrame.")
        if target_variable not in data_raw.columns:
             self.logger.error(f"Target variable '{target_variable}' not found in DataFrame columns.")
             raise ValueError(f"Target variable '{target_variable}' not found in DataFrame columns.")

        self.results['start_time'] = datetime.now().isoformat()
        self.results['user_instructions'] = user_instructions
        self.results['target_variable'] = target_variable
        self.results['initial_data_shape'] = data_raw.shape

        # 1. Generate Initial Data Description
        data_description = self._generate_data_description(data_raw)
        self.results['initial_data_description'] = data_description
        self.logger.info("Generated initial data description.")

        # 2. Get LLM Data Preparation Plan
        prep_plan = self._get_llm_data_prep_plan(data_description, target_variable)
        self.results['llm_data_prep_plan'] = prep_plan if prep_plan else "Failed to get/parse plan"

        # 3. Execute Data Preparation Plan
        if prep_plan:
            prepared_df = self._execute_data_prep_plan(data_raw, prep_plan)
            self.results['prepared_data_shape'] = prepared_df.shape
            self.logger.info(f"Data preparation complete. Shape changed from {data_raw.shape} to {prepared_df.shape}")
        else:
            self.logger.warning("Skipping H2O phase as data preparation failed or produced no plan.")
            prepared_df = data_raw # Use raw data if prep failed
            self.results['prepared_data_shape'] = data_raw.shape


        # --- TODO: Insert Feature Engineering Steps Here ---
        #    - Generate description of *prepared_df*
        #    - Call LLM for feature engineering plan
        #    - Execute feature engineering plan -> final_df = ...
        #    - For now, use prepared_df as final_df
        final_df = prepared_df
        self.results['final_data_shape'] = final_df.shape # Update once FE is added
        self.logger.info("Feature engineering step skipped for now.")
        # --- End TODO ---


        # 4. Get LLM Recommendations for H2O (using FINAL data description)
        #    Requires generating description for final_df first!
        final_data_description = self._generate_data_description(final_df) # Use final data
        self.results['final_data_description'] = final_data_description
        llm_recommendations = self._get_llm_recommendations(final_data_description, user_instructions, target_variable)
        self.results['llm_h2o_recommendations'] = llm_recommendations
        self.logger.info("Generated LLM recommendations for H2O AutoML.")

        # 5. Generate H2O Parameters (using LLM recommendations and user kwargs)
        #    Need to refine how user_preferences are extracted from kwargs
        user_preferences = {k: v for k, v in kwargs.items() if k in ['max_runtime_secs', 'max_models', 'exclude_algos', 'sort_metric', 'nfolds', 'stopping_metric', 'stopping_rounds', 'stopping_tolerance']}
        h2o_params = self._generate_h2o_params(llm_recommendations, target_variable, user_preferences)
        self.results['h2o_parameters_used'] = h2o_params
        self.logger.info(f"Generated H2O parameters: {h2o_params}")

        # 6. Run H2O AutoML (using FINAL data)
        h2o_results = self._run_h2o_automl(final_df, target_variable, h2o_params) # Use final_df
        self.results.update(h2o_results) # Merge H2O results
        self.logger.info("H2O AutoML execution finished.")

        end_time = time.time()
        self.results['total_duration_seconds'] = round(end_time - start_time, 2)
        self.results['end_time'] = datetime.now().isoformat()

        self.logger.info(f"--- ML Agent Invocation Finished (Duration: {self.results['total_duration_seconds']}s) ---")
        # Optionally save results to a log file
        self._save_results_log()

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

    # --- Helper to save results --- NEW ---
    def _save_results_log(self):
        """Saves the results dictionary to a JSON file in the log path."""
        if not self.log:
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_path, f"agent_results_{timestamp}.json")

            # Convert unserializable items (like DataFrames) to strings or summaries
            serializable_results = {}
            for key, value in self.results.items():
                 if isinstance(value, pd.DataFrame):
                     serializable_results[key] = value.to_string() # Or just shape, columns etc.
                 # Add checks for other potential non-serializable types if needed
                 else:
                     serializable_results[key] = value

            with open(log_file, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            self.logger.info(f"Agent results saved to {log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save agent results log: {e}")

# --- Example Usage (REMOVE THIS - Moved to run_agent.py) ---
# if __name__ == "__main__":
#    ... (old test code removed) 
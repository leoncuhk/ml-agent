import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from h2o.model import ModelBase
import os
import time
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
import io
import re
import functools
import shutil
from h2o.exceptions import H2OStartupError

# Imports from other modules within this package
from .llm_interface import initialize_llm_client
from .utils import setup_logging, parse_llm_params
from .data_preparer import DataPreparer

# --- Standalone H2O ML Agent Class ---
class H2OMLAgent:
    """
    An agent that uses an LLM to recommend H2O AutoML steps and generates
    parameters for execution.
    """
    def __init__(self, log=True, log_path="logs/", model_directory="models/"):
        """
        Initializes the H2OMLAgent.
        """
        self.log = log
        self.log_path = log_path
        self.model_directory = model_directory
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.model_directory, exist_ok=True)

        self.logger = setup_logging(log_path=self.log_path)
        if not self.log:
            self.logger.disabled = True

        self.logger.info("Initializing H2OMLAgent...")
        try:
            self.llm = initialize_llm_client()
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM client: {e}")
            raise

        self.results = {}
        self.config = {}
        self.data_preparer = DataPreparer(self.logger)
        self.logger.info("H2OMLAgent initialized.")

    def _check_java_installed(self):
        """Checks if Java is installed and available in the system's PATH."""
        self.logger.info("Checking for Java installation...")
        if shutil.which('java'):
            self.logger.info("Java is found in PATH.")
            return True
        self.logger.error("Java is not found in PATH.")
        return False

    def _generate_data_description(self, df: pd.DataFrame, max_unique_values_to_list=10) -> str:
        self.logger.info(f"Generating data description for DataFrame with shape {df.shape}...")
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        description = (
            f"### Data Summary\n"
            f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            f"### Column Information (Types, Non-Null Counts):\n```\n{info_str}```\n"
        )

        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            description += f"### Missing Value Counts:\n```\n{missing_values.to_string()}```\n\n"
        else:
            description += "- No missing values detected.\n\n"
        
        description += "### Feature Analysis:\n"
        for col in df.columns:
            num_unique = df[col].nunique()
            description += f"- **'{col}'**: {num_unique} unique values.\n"
            if num_unique <= max_unique_values_to_list and not pd.api.types.is_numeric_dtype(df[col].dtype):
                unique_vals = list(df[col].unique())
                description += f"  - *Unique values*: {unique_vals}\n"
        
        self.logger.info("Generated initial data description.")
        return description

    def _get_data_prep_plan(self, data_description, user_prompt):
        self.logger.info("Getting LLM data preparation plan...")
        prompt = f"""
Analyze the following dataset summary and user request to create a data preparation and feature engineering plan.
**User Request:** "{user_prompt}"
**Dataset Summary:**
{data_description}

**Your Task:**
Provide a JSON plan for data preparation and feature engineering. The plan should be a list of steps, where each step is a dictionary with "action" and "parameters".
**Available Actions:**
- "impute": Fill missing values. Parameters: "column" (string), "strategy" (string, e.g., "mean", "median", "most_frequent").
- "scale": Scale numerical features. Parameters: "column" (string), "method" (string, e.g., "standard", "min_max").
- "encode": Convert categorical features to numerical. Parameters: "column" (string), "method" (string, e.g., "one_hot", "label").
- "drop": Remove a column. Parameters: "column" (string).
- "create_interaction": Create a new feature by interacting two existing features. Parameters: "columns" (list of two strings).

**CRITICAL RULE: NEVER, under any circumstances, drop the target variable '{self.config['target_variable']}'.**

**Output ONLY the JSON plan in a single code block.**
"""
        try:
            response = self.llm.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            llm_response_content = response.choices[0].message.content
            plan = self._extract_json_from_response(llm_response_content)
            if plan:
                self.logger.info(f"Received data preparation plan from LLM: {plan}")
            return plan
        except Exception as e:
            self.logger.error(f"Error calling LLM for data prep plan: {e}")
            return None

    def _execute_data_prep_plan(self, df: pd.DataFrame, plan: dict | list) -> pd.DataFrame:
        if not plan:
            self.logger.info("Data preparation plan is empty. Skipping execution.")
            return df

        # Extract the list of steps if the plan is a dictionary
        steps = []
        if isinstance(plan, dict):
            if 'steps' in plan and isinstance(plan.get('steps'), list):
                steps = plan['steps']
                self.logger.info("Extracted 'steps' from dictionary wrapper.")
            elif 'plan' in plan and isinstance(plan.get('plan'), list):
                steps = plan['plan']
                self.logger.info("Extracted 'plan' from dictionary wrapper.")
            elif 'data_preparation_plan' in plan and isinstance(plan.get('data_preparation_plan'), list):
                steps = plan['data_preparation_plan']
                self.logger.info("Extracted 'data_preparation_plan' from dictionary wrapper.")
        elif isinstance(plan, list):
            steps = plan
        
        if not steps:
            self.logger.warning("Data preparation plan is invalid or contains no steps. Skipping execution.")
            return df
            
        self.logger.info(f"Executing data preparation plan with {len(steps)} steps...")
        df_prepared = self.data_preparer.process(df, steps)
        self.logger.info(f"Data preparation plan executed successfully. New data shape: {df_prepared.shape}")
        return df_prepared

    def _get_llm_recommendations(self, data_description, user_prompt):
        self.logger.info("Getting LLM recommendations...")
        prompt = f"""
You are an expert in H2O AutoML. Based on the user's request and a summary of their data, suggest parameters for H2O's AutoML.
The user wants to find the best model for the target variable '{self.config['target_variable']}'.
**User Request:** "{user_prompt}"
**Data Summary:**
{data_description}
**Your Task:**
Provide a list of key-value pair suggestions for the H2O AutoML constructor.
- Suggest a `max_runtime_secs` (e.g., 60, 120, 300).
- Suggest a `max_models` (e.g., 10, 20, 50).
- Suggest `exclude_algos` if certain algorithms are not suitable (e.g., `exclude_algos: ["DeepLearning", "GLM"]`).
- Suggest a `sort_metric` appropriate for the problem (e.g., "AUC" for binary classification, "RMSE" for regression).
- Suggest `balance_classes: true` if the target is binary and might be imbalanced.
**Output ONLY the key-value pairs, one per line.** Example:
max_runtime_secs: 60
max_models: 10
sort_metric: AUC
"""
        try:
            response = self.llm.chat.completions.create(
                model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}]
            )
            llm_response_content = response.choices[0].message.content
            self.logger.info("Successfully received recommendations from LLM.")
            return llm_response_content
        except Exception as e:
            self.logger.error(f"Error calling LLM for recommendations: {e}")
            return None

    def _generate_h2o_params(self, llm_recommendations, user_preferences):
        self.logger.info("Generating H2O AutoML parameters...")
        h2o_params = {'seed': 1234}
        if llm_recommendations:
            parsed_params = parse_llm_params(llm_recommendations, self.logger)
            h2o_params.update(parsed_params)
            self.logger.info(f"Applied LLM suggestions: {parsed_params}")

        for key, value in user_preferences.items():
            if key in h2o_params:
                self.logger.info(f"Applying user preference: {key}={value} (overriding previous value)")
            h2o_params[key] = value

        self.logger.info(f"Final H2O AutoML parameters generated: {h2o_params}")
        return h2o_params
        
    def _run_h2o_automl(self, data_prepared: pd.DataFrame, h2o_params: dict):
        self.logger.info("--- Starting H2O AutoML Execution ---")
        try:
            if not self._check_java_installed():
                error_message = (
                    "Cannot find Java. Please install the latest JRE from\n"
                    "http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#java-requirements"
                )
                raise H2OStartupError(error_message)

            h2o.init(nthreads=-1, max_mem_size="8g")
            self.logger.info("H2O initialized.")
            
            h2o_df = h2o.H2OFrame(data_prepared)
            self.logger.info("H2O Frame created.")

            y = self.config["target_variable"]
            x = [col for col in h2o_df.columns if col != y]
            self.logger.info(f"Target variable (y): {y}")
            self.logger.info(f"Predictor variables (x): {x}")

            target_col = h2o_df[y]
            
            # --- Robust Problem Type Inference ---
            is_factor = target_col.isfactor()[0]
            num_levels = target_col.nlevels()[0] if is_factor else data_prepared[y].nunique()

            self.task = "regression"
            if is_factor:
                self.task = "classification"
                self.logger.info(f"Target '{y}' is a factor with {num_levels} levels. Task set to classification.")
            elif num_levels == 2:
                self.task = "classification"
                self.logger.info(f"Target '{y}' has 2 unique values. Converting to factor for classification.")
                h2o_df[y] = target_col.asfactor()
            elif 2 < num_levels < 20:
                self.task = "classification"
                self.logger.info(f"Target '{y}' has {num_levels} unique values (<20). Converting to factor for classification.")
                h2o_df[y] = target_col.asfactor()

            self.logger.info(f"Starting H2O AutoML for inferred problem type: {self.task}")
            
            if self.task == "classification":
                h2o_params.setdefault('sort_metric', 'AUC')
            
            self.logger.info(f"Using H2O AutoML parameters: {h2o_params}")

            aml = H2OAutoML(**h2o_params)
            aml.train(x=x, y=y, training_frame=h2o_df)
            
            self.logger.info("H2O AutoML training complete.")
            self.results['leaderboard'] = aml.leaderboard.as_data_frame()
            
            if aml.leader and self.results['leaderboard'] is not None and not self.results['leaderboard'].empty:
                self.results['best_model'] = aml.leader
                model_path = h2o.save_model(model=aml.leader, path=self.model_directory, force=True)
                self.results['best_model_path'] = model_path
                self.logger.info(f"Best model saved to: {model_path}")

                # --- Enhanced Model Evaluation ---
                # Log confusion matrix for classification tasks
                if self.task == "classification":
                    cm = aml.leader.confusion_matrix(valid=True)
                    if cm:
                        self.results['confusion_matrix'] = cm.table.as_data_frame().to_dict()
                        self.logger.info(f"Confusion Matrix (Validation):\n{cm.table}")

                # Log feature importance for all model types
                try:
                    fi = aml.leader.varimp(use_pandas=True)
                    if fi is not None:
                        self.results['feature_importance'] = fi.to_dict('records')
                        self.logger.info(f"Feature Importance:\n{fi}")
                except Exception as e:
                    self.logger.warning(f"Could not retrieve feature importance: {e}")
                # --- End of Enhanced Evaluation ---

                self.results['status'] = 'Completed Successfully'
            else:
                self.logger.warning("H2O AutoML finished, but no leader model found.")
                self.results['status'] = 'Failed'
                self.results['error'] = 'AutoML finished without a leader model.'

        except Exception as e:
            self.logger.error(f"An error occurred during H2O AutoML execution: {e}", exc_info=True)
            self.results['status'] = 'Failed'
            self.results['error'] = str(e)
        finally:
            self.logger.info("--- H2O AutoML Execution Finished ---")

    def _extract_json_from_response(self, text: str) -> list | dict | None:
        if not text:
            self.logger.warning("LLM response was empty.")
            return None

        # Attempt 1: Parse the entire string as JSON (for response_format={"type": "json_object"})
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            self.logger.debug("Response is not a raw JSON object. Trying to extract from markdown.")

        # Attempt 2: Extract from a markdown code block (```json ... ```)
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from extracted markdown block: {e}")
                self.logger.debug(f"Content of invalid JSON block: {json_str}")
                return None
        
        self.logger.warning("No valid JSON found in LLM response, either raw or in a markdown block.")
        return None

    def invoke_agent(self, df: pd.DataFrame, user_instructions: str, target_variable: str, **kwargs):
        start_time = time.time()
        self.logger.info("--- Starting ML Agent Invocation ---")
        
        self.config = {
            "target_variable": target_variable,
            "user_instructions": user_instructions,
            "user_preferences": kwargs
        }
        self.results = {'config': self.config}

        initial_data_desc = self._generate_data_description(df)
        data_prep_plan = self._get_data_prep_plan(initial_data_desc, user_instructions)
        self.results['data_prep_plan'] = data_prep_plan
        df_prepared = self._execute_data_prep_plan(df, data_prep_plan)

        final_data_desc = self._generate_data_description(df_prepared)
        llm_recs = self._get_llm_recommendations(final_data_desc, user_instructions)
        self.results['llm_recommendations'] = llm_recs
        
        h2o_params = self._generate_h2o_params(llm_recs, kwargs)
        self.results['h2o_params'] = h2o_params
        self._run_h2o_automl(df_prepared, h2o_params)
        
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f"--- ML Agent Invocation Finished (Duration: {duration:.2f}s) ---")
        self._save_results_log()
        return self.results

    def get_leaderboard(self) -> pd.DataFrame | None:
        lb = self.results.get('leaderboard')
        if isinstance(lb, pd.DataFrame):
            return lb
        self.logger.warning("Stored leaderboard is not a Pandas DataFrame.")
        return None

    def get_results(self) -> dict:
        """Return the results dictionary containing all execution results."""
        return self.results

    def get_workflow_summary(self, markdown=False) -> str:
        summary = "### ML Agent Workflow Summary\n\n"
        
        summary += "**1. Data Preparation Plan (from LLM):**\n"
        plan = self.results.get('data_prep_plan')
        if plan:
            summary += f"```json\n{json.dumps(plan, indent=2)}\n```\n"
        else:
            summary += "> No valid plan was generated or executed.\n"

        summary += "\n**2. H2O AutoML Configuration:**\n"
        params = self.results.get('h2o_params', {})
        summary += f"```json\n{json.dumps(params, indent=2)}\n```\n"
        
        summary += "\n**3. H2O AutoML Execution:**\n"
        status = self.results.get('status', 'N/A')
        summary += f"> Status: **{status}**\n"
        if 'error' in self.results:
            summary += f"> Error: {self.results['error']}\n"
        if self.results.get('best_model'):
            summary += f"> Best Model ID: {self.results['best_model'].model_id}\n"
        if 'best_model_path' in self.results:
            summary += f"> Best Model Path: {self.results.get('best_model_path')}\n"

        return summary

    def _save_results_log(self):
        log_filename = f"agent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_filepath = os.path.join(self.log_path, log_filename)
        
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, (pd.DataFrame, h2o.H2OFrame, ModelBase)):
                serializable_results[key] = str(type(value))
            elif key == 'best_model':
                 serializable_results[key] = value.model_id if value else None
            else:
                serializable_results[key] = value

        try:
            with open(log_filepath, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            self.logger.info(f"Agent results saved to {log_filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results log: {e}") 
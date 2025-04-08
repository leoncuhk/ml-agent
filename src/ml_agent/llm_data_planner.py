# src/ml_agent/llm_planner.py

import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import logging
import re
from inspect import signature, Parameter # To inspect data_preparer functions
import pandas as pd
from typing import Dict, Any, Optional

# Assuming data_preparer is in the same directory level
try:
    from . import data_preparer
except ImportError:
    # Allow running script directly for testing
    import data_preparer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Function to Get Toolkit Info ---
def get_data_preparer_functions_info() -> str:
    """
    Generates a description of available functions in data_preparer.py
    using their docstrings and signatures.
    """
    logger.debug("Generating toolkit description from data_preparer.py")
    functions_info = []
    # Functions to expose to the LLM planner
    # We can be selective here if needed
    available_functions = [
        'impute_missing',
        'convert_to_numeric',
        'remove_duplicates',
        'strip_whitespace',
        'remove_constant_columns',
        'handle_high_cardinality',
        'encode_categorical',
        'encode_label', # Maybe less common for automatic planning unless target is clear
        'handle_outliers',
        'scale_numerical',
        'extract_date_features'
    ]

    for func_name in available_functions:
        if hasattr(data_preparer, func_name):
            func = getattr(data_preparer, func_name)
            if callable(func):
                sig = signature(func)
                docstring = func.__doc__ if func.__doc__ else "No description available."
                # Extract purpose from docstring (simple approach)
                purpose = docstring.split('\n')[1].strip() if docstring else "N/A"

                params_list = []
                for name, param in sig.parameters.items():
                    if name == 'df': continue # Exclude 'df' parameter
                    param_info = f"{name}"
                    if param.default is not Parameter.empty:
                        # Represent default values concisely for the prompt
                        default_val = param.default
                        if isinstance(default_val, str):
                            default_val_str = f"'{default_val}'"
                        elif isinstance(default_val, (list, dict)) and not default_val:
                             default_val_str = str(type(default_val)()) # e.g., '[]' or '{}'
                        elif default_val is None:
                             default_val_str = "None"
                        else:
                             default_val_str = str(default_val)
                        param_info += f" (default: {default_val_str})"
                    params_list.append(param_info)

                functions_info.append(
                    f"- Function: `{func_name}({', '.join(params_list)})`\n"
                    f"  Purpose: {purpose}"
                    # Add more details from docstring if needed, e.g., strategies
                )
        else:
            logger.warning(f"Function '{func_name}' listed in available_functions not found in data_preparer.")

    if not functions_info:
        logger.error("Could not find any specified functions in data_preparer module.")
        return "Error: No data preparation functions found."

    logger.debug(f"Generated toolkit description with {len(functions_info)} functions.")
    return "\n".join(functions_info)


def construct_planner_prompt(analysis_json: dict, toolkit_description: str, user_goal: str = None) -> str:
    """Constructs the prompt for the LLM planning task."""
    logger.debug("Constructing LLM planner prompt.")

    # Convert analysis dict to JSON string for the prompt
    analysis_str = json.dumps(analysis_json, indent=2)

    # Define the desired JSON output structure for the plan
    plan_structure_description = """
    [
      {
        "step": <int>, // Sequential step number (starting from 1)
        "function": "<string: function_name_from_toolkit>",
        "params": {
          "<string: param_name>": "<value>", // Use appropriate JSON types (string, number, boolean, list)
          // e.g., "column": "ColumnName", "strategy": "median", "columns": ["Col1", "Col2"]
        },
        "reasoning": "<string: Brief explanation why this step is needed based on the analysis, referencing specific findings>"
      },
      // ... more steps
    ]
    """

    # Updated prompt with more specific instructions on ordering and logic
    prompt = f"""
You are an expert data scientist acting as a planner for an automated data preprocessing pipeline.

**Input:**

1.  **Data Analysis Report (JSON):** This report details the structure, quality issues, and preliminary suggestions for a dataset.
    ```json
    {analysis_str}
    ```

2.  **User Goal (if provided):** {user_goal if user_goal else "Not specified. General preprocessing for ML."}

3.  **Available Data Preprocessing Tools (Python functions in `data_preparer.py`):** You MUST use ONLY these functions to build the plan. Pay close attention to their parameters and purpose.
    ```
    {toolkit_description}
    ```

**Your Task:**

Based on the Data Analysis Report and the User Goal, generate a step-by-step data preprocessing plan. The plan should address the identified `potential_issues` and follow the `suggested_actions` where appropriate, using the available tools. Ensure the steps are in a **strict logical order** to maximize effectiveness and avoid conflicts.

**Output Format:**

Return **ONLY** a single, valid JSON list object representing the plan. Do not include any introductory text, explanations, markdown formatting, or anything else outside the JSON list itself. Adhere strictly to the following structure for each step in the list:

```json
{plan_structure_description}
```

**Critical Instructions for Plan Logic and Ordering:**

1.  **Initial Cleanup:** Start with steps that remove data entirely:
    *   Call `remove_duplicates()` first if duplicates were identified.
    *   Call `remove_constant_columns()` early if constant columns were identified.
    *   **Include steps to address useless ID/PII columns identified in the analysis (like FullName, StudentID if not target/key). Even if there isn't a specific 'remove_column' function in the toolkit, state the intent by including a step like: `{{"step": X, "function": "COMMENT", "params": {{"column_to_remove": "ColumnName"}}, "reasoning": "Remove PII/ID column as identified in analysis."}}`. The execution phase can handle this comment.**

2.  **Basic Cleaning:** Then perform basic value cleaning:
    *   Call `strip_whitespace()` for columns identified with whitespace issues.

3.  **Type Conversion:** Convert columns to their correct types *before* operations that depend on type:
    *   Call `convert_to_numeric()` for columns identified as numeric but stored as object/string. Specify `errors='coerce'`.

4.  **Date Feature Extraction:** Attempt to extract features from date columns *before* imputation:
    *   Call `extract_date_features()` for columns identified as Datetime. Let the function handle potential parsing errors internally. Set `drop_original=True` if the original date string is no longer needed.

5.  **Imputation:** Handle missing values *after* type conversions and initial cleaning/extraction, as the strategy might depend on the final type or presence of NaNs introduced by previous steps:
    *   Call `impute_missing()` for columns with missing values. Choose the strategy ('mean', 'median', 'mode', or even 'constant' like 'Missing') based on the column type and distribution mentioned in the analysis (e.g., use 'median' for skewed numerical data, 'mode' for categorical).

6.  **Outlier Handling:** Address outliers *after* type conversions and imputation:
    *   Call `handle_outliers()` for numerical columns identified with outliers. Specify the `columns` list and a suitable `strategy` ('clip' or 'clip_std').

7.  **High Cardinality:** Handle high cardinality categorical features:
    *   Call `handle_high_cardinality()` for categorical columns with too many unique values, specifying a `threshold` (e.g., 0.05 for 5%) and `replace_with='Other'`.

8.  **Encoding:** Encode remaining categorical (and potentially boolean) features needed for the model. This should happen *after* cleaning and high cardinality handling:
    *   Call `encode_categorical()` for object/category columns (use `strategy='one-hot'`, `drop_original=True`, consider `drop_first=True`).
    *   Call `encode_label()` if the analysis identified a clear categorical target variable that needs label encoding.

9.  **Scaling:** Scale numerical features *last*, after all cleaning, imputation, extraction, and encoding steps that might affect which columns are numeric:
    *   **IMPORTANT:** Identify *all* columns that are numeric *at this final stage* (excluding IDs, targets unless regression, etc.).
    *   Call `scale_numerical()` specifying the complete list of numerical `columns` to be scaled and a `strategy` (e.g., 'standard').

**Parameter Specification:**

*   Accurately specify the `column` or `columns` parameter for each function based on the analysis report. Use correct JSON types (string, number, boolean, list of strings).
*   Choose appropriate strategies ('mean', 'median', 'mode', 'clip', 'standard', 'one-hot', etc.) based on the analysis and standard practices.

**Reasoning:** Briefly justify *why* each step is necessary, linking it back to the analysis report findings. For the "COMMENT" steps, explain the intent (e.g., "Remove PII column...").

Now, generate the JSON plan based on these refined instructions.
"""
    logger.debug(f"Constructed planner prompt length: {len(prompt)}")
    return prompt


def augment_response_with_chinese_summary(plan_json):
    """
    为生成的数据准备计划添加中文总结
    
    Args:
        plan_json: 原始计划JSON对象
        
    Returns:
        计划JSON对象，添加了中文总结字段
    """
    if not isinstance(plan_json, list):
        logger.warning("План не является списком, невозможно добавить китайское резюме")
        return plan_json
    
    steps_count = len(plan_json)
    
    # 解析步骤类型、目标列等
    operation_types = {}
    processed_columns = set()
    
    for step in plan_json:
        if not isinstance(step, dict):
            continue
            
        function = step.get('function', '')
        if function:
            operation_types[function] = operation_types.get(function, 0) + 1
        
        # 收集处理的列
        for param_name in ['column', 'columns']:
            column_param = step.get('args', {}).get(param_name)
            if isinstance(column_param, str):
                processed_columns.add(column_param)
            elif isinstance(column_param, list):
                processed_columns.update(column_param)
    
    # 构建中文总结
    summary = f"数据准备计划包含 {steps_count} 个步骤，"
    
    # 添加操作类型摘要
    if operation_types:
        op_text = []
        for op_type, count in operation_types.items():
            if op_type == 'impute_missing':
                op_text.append(f"缺失值填充 {count} 次")
            elif op_type == 'convert_to_numeric':
                op_text.append(f"数值类型转换 {count} 次")
            elif op_type == 'remove_duplicates':
                op_text.append("去除重复行")
            elif op_type == 'strip_whitespace':
                op_text.append("去除字符串空格")
            elif op_type == 'handle_outliers':
                op_text.append("处理异常值")
            elif op_type == 'scale_numerical':
                op_text.append("数值特征缩放")
            elif op_type == 'encode_categorical':
                op_text.append("分类特征编码")
            else:
                op_text.append(f"{op_type} {count} 次")
        
        summary += "包括" + "、".join(op_text) + "。"
    
    # 添加处理列数量
    if processed_columns:
        summary += f"总共处理了 {len(processed_columns)} 个特征列。"
    
    # 为第一个步骤添加summary_chinese字段，使其能在workflow日志中显示
    if plan_json and isinstance(plan_json[0], dict):
        plan_json[0]['summary_chinese'] = summary
    
    return plan_json


def generate_plan_with_llm(analysis_report: dict, user_goal: str = None) -> list | None:
    """
    Generates a data preprocessing plan using an LLM based on an analysis report.

    Args:
        analysis_report (dict): The structured analysis report from llm_analyzer.
        user_goal (str, optional): A high-level user goal. Defaults to None.

    Returns:
        list | None: A list of dictionaries representing the plan steps,
                     or None if plan generation fails.
    """
    logger.info("Starting LLM plan generation.")

    # --- Step 1: Load API Key ---
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY not found in .env file. LLM planning cannot proceed.")
        return None

    genai.configure(api_key=api_key)

    # --- Step 2: Get Toolkit Description ---
    toolkit_description = get_data_preparer_functions_info()
    if "Error" in toolkit_description:
         logger.error("Failed to get toolkit description.")
         return None

    # --- Step 3: Construct Prompt ---
    prompt = construct_planner_prompt(analysis_report, toolkit_description, user_goal)
    if not prompt:
        logger.error("Failed to construct planner prompt.")
        return None

    # --- Step 4: Call LLM API ---
    try:
        logger.info("Sending request to LLM API for planning...")
        # Consider using a more capable model if complex planning is needed,
        # but flash should be okay for this structured task.
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        logger.info("Received response from LLM API.")

        # --- Step 5: Parse LLM Response ---
        # Reuse the JSON parsing logic (or similar)
        # Assume the response should be a JSON list directly
        plan = parse_llm_json_response(response.text, expect_type=list)
        if not plan:
             logger.error("Failed to parse JSON list from LLM response.")
             logger.debug(f"Raw LLM response for planning: {response.text}")
             return None

        # Basic validation: check if it's a list of dicts with expected keys
        if isinstance(plan, list) and all(isinstance(item, dict) and 'function' in item and 'params' in item for item in plan):
            logger.info(f"Successfully parsed LLM plan with {len(plan)} steps.")
            
            # 添加中文总结
            plan = augment_response_with_chinese_summary(plan)
            
            return plan
        else:
            logger.error(f"Parsed JSON is not a valid plan structure (list of dicts with 'function' and 'params'). Parsed data: {plan}")
            return None

    except Exception as e:
        logger.error(f"An error occurred during LLM API call or parsing for planning: {e}", exc_info=True)
        return None


def parse_llm_json_response(response_text: str, expect_type: type = dict) -> dict | list | None:
    """
    Attempts to extract and parse a JSON object or list from the LLM's response text.
    (Enhanced version from llm_analyzer to handle lists too)
    """
    logger.debug(f"Attempting to parse JSON {expect_type.__name__} from LLM response.")
    json_str = None

    # Try to find JSON block enclosed in ```json ... ```
    match_block = re.search(r"```json\s*([\[{].*?[\]}])\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if match_block:
        json_str = match_block.group(1)
        logger.debug(f"Found JSON {expect_type.__name__} block within ```json ... ```")
    else:
        # Fallback: try to find JSON starting with { or [ and ending with } or ]
        # Be careful with greedy matching, ensure it's likely the main JSON part
        if expect_type == dict:
            match_direct = re.search(r"(\{.*?\})", response_text, re.DOTALL)
        elif expect_type == list:
             match_direct = re.search(r"(\[.*?\])", response_text, re.DOTALL)
        else:
            match_direct = None # Should not happen with current usage

        if match_direct:
             json_str = match_direct.group(1)
             logger.debug(f"Found JSON {expect_type.__name__} block directly.")
        else:
            # Sometimes the LLM might just return the JSON without markers
            # Try cleaning up the start/end and parsing directly if it looks like JSON
            potential_json = response_text.strip()
            if potential_json.startswith( ('{', '[') ) and potential_json.endswith( ('}', ']') ):
                logger.debug("Attempting to parse stripped response text as JSON.")
                json_str = potential_json
            else:
                 logger.warning(f"Could not find JSON {expect_type.__name__} block in the response.")
                 return None

    if not json_str:
        return None

    try:
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, expect_type):
            logger.debug(f"Successfully parsed JSON as {expect_type.__name__}.")
            return parsed_json
        else:
            logger.warning(f"Parsed JSON is not of expected type {expect_type.__name__}. Type: {type(parsed_json)}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}")
        logger.error(f"Invalid JSON string received: {json_str[:500]}...") # Log beginning of problematic string
        return None


def plan_data_prep(df: pd.DataFrame, analysis_results: Dict[str, Any], language: str = 'en') -> Dict[str, Any]:
    """
    Create a detailed data preparation plan using LLM based on data analysis results.
    
    Args:
        df: DataFrame to be processed
        analysis_results: Results from the data analysis stage
        language: Output language - 'en' for English, 'cn' for Chinese and English
        
    Returns:
        Dict containing the data preparation plan
    """
    if language == 'cn':
        logger.info("开始根据数据分析结果生成数据准备计划...")
    else:
        logger.info("Starting generation of data preparation plan based on analysis results...")
    
    # Extract key information from analysis
    data_overview = analysis_results.get('data_overview', '')
    quality_issues = analysis_results.get('quality_issues', [])
    prep_recommendations = analysis_results.get('preparation_recommendations', [])
    column_analysis = analysis_results.get('column_analysis', [])
    potential_target = analysis_results.get('potential_target_variable')
    potential_id_columns = analysis_results.get('potential_id_columns', [])
    
    # Create simplified column info for LLM
    column_info = []
    for col in column_analysis:
        col_name = col.get('column_name', '')
        data_type = col.get('data_type', '')
        missing_pct = col.get('missing_values', {}).get('percentage', 0)
        distribution = col.get('distribution', 'unknown')
        
        column_info.append({
            'name': col_name,
            'type': data_type,
            'missing_percentage': missing_pct,
            'distribution': distribution
        })
    
    # Create a categorized list of quality issues
    categorized_issues = {}
    for issue in quality_issues:
        issue_type = issue.get('issue_type', '')
        if issue_type not in categorized_issues:
            categorized_issues[issue_type] = []
        categorized_issues[issue_type].append(issue)
    
    if language == 'cn':
        # 中文提示
        prompt = f"""
你是一位专业的数据工程师，负责设计数据准备和清洗流程。请基于以下信息创建详细的数据准备计划：

### 数据集概述
{data_overview}

### 列信息
```
{json.dumps(column_info, indent=2, ensure_ascii=False)}
```

### 数据质量问题
```
{json.dumps(categorized_issues, indent=2, ensure_ascii=False)}
```

### 建议的准备步骤
```
{json.dumps(prep_recommendations, indent=2, ensure_ascii=False)}
```

### 潜在目标变量
{potential_target}

### 潜在ID列
{', '.join(potential_id_columns) if potential_id_columns else "未发现ID列"}

请创建一个详细且可操作的数据准备计划，其中包含以下内容：
1. 一系列具体的数据清洗、转换和准备步骤
2. 每个步骤的理由说明
3. 每个步骤的Python函数调用

请按照以下JSON格式提供计划：
```json
{{
  "data_prep_plan": [
    {{
      "step_number": 1,
      "step_name": "步骤的简洁名称",
      "reasoning": "为什么需要这个步骤",
      "function": "函数名称",
      "params": {{
        "参数1": "值1",
        "参数2": "值2"
      }}
    }},
    ...更多步骤...
  ],
  "expected_outcomes": [
    "预期成果1",
    "预期成果2",
    ...
  ],
  "data_preparation_summary": "英文数据准备计划的总结",
  "data_preparation_summary_chinese": "中文数据准备计划的总结"
}}
```

请确保：
1. 每个步骤都是独立的，且按照逻辑顺序排列
2. 函数名称是从常见数据处理函数中选取的（例如：remove_constant_columns, remove_duplicate_rows, impute_missing_values, convert_to_datetime, convert_to_numeric, encode_categorical, remove_outliers, scale_numeric, etc.）
3. 参数是根据具体情况定义的适当值
4. 计划全面解决所有已识别的数据质量问题
5. 如果建议放弃某些列，请提供明确的理由

请确保JSON格式正确，并提供逻辑清晰、步骤具体的方案。
"""
    else:
        # English prompt
        prompt = f"""
You are a professional data engineer tasked with designing a data preparation and cleaning pipeline. Create a detailed data preparation plan based on the following information:

### Dataset Overview
{data_overview}

### Column Information
```
{json.dumps(column_info, indent=2)}
```

### Data Quality Issues
```
{json.dumps(categorized_issues, indent=2)}
```

### Suggested Preparation Steps
```
{json.dumps(prep_recommendations, indent=2)}
```

### Potential Target Variable
{potential_target}

### Potential ID Columns
{', '.join(potential_id_columns) if potential_id_columns else "No ID columns found"}

Please create a detailed, actionable data preparation plan that includes:
1. A series of specific data cleaning, transformation, and preparation steps
2. Reasoning for each step
3. Python function calls for each step

Provide the plan in the following JSON format:
```json
{{
  "data_prep_plan": [
    {{
      "step_number": 1,
      "step_name": "Concise name for the step",
      "reasoning": "Why this step is needed",
      "function": "function_name",
      "params": {{
        "param1": "value1",
        "param2": "value2"
      }}
    }},
    ...more steps...
  ],
  "expected_outcomes": [
    "Expected outcome 1",
    "Expected outcome 2",
    ...
  ],
  "data_preparation_summary": "Summary of the data preparation plan"
}}
```

Please ensure that:
1. Each step is self-contained and in logical order
2. Function names are from common data processing functions (e.g., remove_constant_columns, remove_duplicate_rows, impute_missing_values, convert_to_datetime, convert_to_numeric, encode_categorical, remove_outliers, scale_numeric, etc.)
3. Parameters are appropriate values defined for the specific context
4. The plan comprehensively addresses all identified data quality issues
5. If dropping columns is suggested, provide clear reasoning

Ensure the JSON is well-formed and the plan is logical with concrete steps.
"""

    try:
        # Call LLM with the prompt
        response = chat_with_llm(prompt)
        
        # Parse the response to extract JSON
        plan_results = _extract_json_from_text(response)
        
        if not plan_results:
            if language == 'cn':
                logger.warning("LLM响应未包含有效的JSON，将使用基础数据准备计划")
            else:
                logger.warning("LLM response didn't contain valid JSON, using basic data preparation plan")
            plan_results = _create_basic_prep_plan(analysis_results, language)
        
        # Validate and fix the plan
        plan_results = _validate_and_fix_prep_plan(plan_results, analysis_results, language)
        
        if language == 'cn':
            logger.info("数据准备计划已生成，包含 %d 个步骤", len(plan_results.get('data_prep_plan', [])))
            if 'data_preparation_summary_chinese' in plan_results:
                logger.info("中文总结: %s", plan_results.get('data_preparation_summary_chinese'))
        else:
            logger.info("Data preparation plan generated with %d steps", len(plan_results.get('data_prep_plan', [])))
        
        return plan_results
        
    except Exception as e:
        if language == 'cn':
            logger.error(f"生成数据准备计划时出错: {str(e)}")
        else:
            logger.error(f"Error generating data preparation plan: {str(e)}")
        return _create_basic_prep_plan(analysis_results, language)


def _create_basic_prep_plan(analysis_results: Dict[str, Any], language: str = 'en') -> Dict[str, Any]:
    """
    Create a basic data preparation plan when LLM fails.
    
    Args:
        analysis_results: Results from the data analysis stage
        language: Output language - 'en' for English, 'cn' for Chinese and English
        
    Returns:
        Dict containing a basic data preparation plan
    """
    if language == 'cn':
        logger.info("创建基本数据准备计划...")
    else:
        logger.info("Creating basic data preparation plan...")
    
    # Extract info from analysis results
    quality_issues = analysis_results.get('quality_issues', [])
    column_analysis = analysis_results.get('column_analysis', [])
    potential_id_columns = analysis_results.get('potential_id_columns', [])
    
    # Initialize basic plan
    data_prep_plan = []
    step_num = 1
    
    # Step 1: Remove constant columns
    data_prep_plan.append({
        "step_number": step_num,
        "step_name": "Remove constant columns" if language == 'en' else "移除常量列",
        "reasoning": "Constant columns don't provide any information for analysis" if language == 'en' else "常量列不提供任何分析信息",
        "function": "remove_constant_columns",
        "params": {}
    })
    step_num += 1
    
    # Step 2: Handle whitespace in string columns
    data_prep_plan.append({
        "step_number": step_num,
        "step_name": "Strip whitespace" if language == 'en' else "去除字符串空白",
        "reasoning": "Whitespace can cause inconsistencies in string data" if language == 'en' else "字符串中的空白可能导致数据不一致",
        "function": "strip_whitespace",
        "params": {}
    })
    step_num += 1
    
    # Step 3: Convert numeric columns with string format
    data_prep_plan.append({
        "step_number": step_num,
        "step_name": "Convert numeric strings" if language == 'en' else "转换数值字符串",
        "reasoning": "Convert string columns with numeric values to proper numeric types" if language == 'en' else "将包含数值的字符串列转换为适当的数值类型",
        "function": "convert_to_numeric",
        "params": {}
    })
    step_num += 1
    
    # Step 4: Convert date columns
    date_columns = []
    for col in column_analysis:
        if col.get('distribution') == 'datetime' or col.get('likely_date_format'):
            date_columns.append(col.get('column_name', ''))
    
    if date_columns:
        data_prep_plan.append({
            "step_number": step_num,
            "step_name": "Convert date columns" if language == 'en' else "转换日期列",
            "reasoning": "Standardize date columns to datetime format" if language == 'en' else "将日期列标准化为datetime格式",
            "function": "convert_to_datetime",
            "params": {
                "columns": date_columns
            }
        })
        step_num += 1
        
        # Step 5: Extract date features if date columns exist
        data_prep_plan.append({
            "step_number": step_num,
            "step_name": "Extract date features" if language == 'en' else "提取日期特征",
            "reasoning": "Create useful features from datetime columns" if language == 'en' else "从日期时间列创建有用特征",
            "function": "extract_date_features",
            "params": {
                "columns": date_columns
            }
        })
        step_num += 1
    
    # Step 6: Handle missing values
    missing_value_columns = []
    for col in column_analysis:
        missing_pct = col.get('missing_values', {}).get('percentage', 0)
        if missing_pct > 0:
            missing_value_columns.append(col.get('column_name', ''))
    
    if missing_value_columns:
        data_prep_plan.append({
            "step_number": step_num,
            "step_name": "Impute missing values" if language == 'en' else "填充缺失值",
            "reasoning": "Fill missing values to prepare data for modeling" if language == 'en' else "填充缺失值以准备建模数据",
            "function": "impute_missing_values",
            "params": {
                "strategy": "auto"
            }
        })
        step_num += 1
    
    # Step 7: Handle potential ID columns
    if potential_id_columns:
        data_prep_plan.append({
            "step_number": step_num,
            "step_name": "Flag ID columns" if language == 'en' else "标记ID列",
            "reasoning": "Mark ID columns so they can be excluded from feature engineering" if language == 'en' else "标记ID列，以便从特征工程中排除",
            "function": "flag_id_columns",
            "params": {
                "columns": potential_id_columns
            }
        })
        step_num += 1
    
    # Create expected outcomes
    expected_outcomes = []
    expected_outcomes.append(
        "Clean dataset with standardized data types" if language == 'en' else "具有标准化数据类型的干净数据集"
    )
    expected_outcomes.append(
        "Properly formatted dates and numeric values" if language == 'en' else "格式正确的日期和数值"
    )
    expected_outcomes.append(
        "Missing values handled appropriately" if language == 'en' else "适当处理的缺失值"
    )
    
    # Create summary
    if language == 'en':
        summary = "Basic data preparation plan to clean and standardize the dataset by handling missing values, "
        summary += "correcting data types, and formatting dates."
    else:
        summary = "通过处理缺失值、更正数据类型和格式化日期来清理和标准化数据集的基本数据准备计划。"
        summary_chinese = "基本数据准备计划，通过处理缺失值、修正数据类型和标准化日期格式来清理数据集。"
    
    # Assemble the final plan
    plan_results = {
        "data_prep_plan": data_prep_plan,
        "expected_outcomes": expected_outcomes,
        "data_preparation_summary": summary
    }
    
    # Add Chinese summary if needed
    if language == 'cn':
        plan_results["data_preparation_summary_chinese"] = summary_chinese
    
    if language == 'cn':
        logger.info("基本数据准备计划已创建，包含 %d 个步骤", len(data_prep_plan))
    else:
        logger.info("Basic data preparation plan created with %d steps", len(data_prep_plan))
    
    return plan_results


def _validate_and_fix_prep_plan(plan_results: Dict[str, Any], analysis_results: Dict[str, Any], language: str = 'en') -> Dict[str, Any]:
    """
    Validate and fix the data preparation plan if needed.
    
    Args:
        plan_results: The LLM-generated data preparation plan
        analysis_results: Results from the data analysis stage
        language: Output language - 'en' for English, 'cn' for Chinese and English
        
    Returns:
        Validated and fixed data preparation plan
    """
    # Check required keys
    required_keys = ["data_prep_plan", "expected_outcomes", "data_preparation_summary"]
    missing_keys = [key for key in required_keys if key not in plan_results]
    
    if missing_keys:
        if language == 'cn':
            logger.warning("数据准备计划缺少以下键: %s", missing_keys)
        else:
            logger.warning("Data preparation plan missing keys: %s", missing_keys)
        
        # Create missing keys with default values
        for key in missing_keys:
            if key == "data_prep_plan":
                plan_results[key] = []
            elif key == "expected_outcomes":
                plan_results[key] = ["Clean and prepared dataset"]
            elif key == "data_preparation_summary":
                plan_results[key] = "Data preparation plan to clean and format the dataset."
    
    # Validate each step in the plan
    valid_steps = []
    for step in plan_results.get("data_prep_plan", []):
        # Check if required step fields are present
        if not all(k in step for k in ["step_number", "step_name", "function", "params"]):
            if language == 'cn':
                logger.warning("跳过无效的步骤: %s", step.get("step_name", "未命名步骤"))
            else:
                logger.warning("Skipping invalid step: %s", step.get("step_name", "Unnamed step"))
            continue
        
        # Add reasoning if missing
        if "reasoning" not in step:
            step["reasoning"] = "Improve data quality" if language == 'en' else "提高数据质量"
        
        # Ensure params is a dictionary
        if not isinstance(step["params"], dict):
            if language == 'cn':
                logger.warning("步骤 '%s' 的参数不是字典，设置为空字典", step.get("step_name", ""))
            else:
                logger.warning("Params for step '%s' is not a dictionary, setting to empty dict", step.get("step_name", ""))
            step["params"] = {}
        
        valid_steps.append(step)
    
    # Update with valid steps
    plan_results["data_prep_plan"] = valid_steps
    
    # Add or fix step numbers to ensure they're sequential
    for i, step in enumerate(valid_steps):
        step["step_number"] = i + 1
    
    # Ensure Chinese summary exists if needed
    if language == 'cn' and "data_preparation_summary_chinese" not in plan_results:
        english_summary = plan_results.get("data_preparation_summary", "数据准备计划")
        plan_results["data_preparation_summary_chinese"] = f"中文数据准备总结: {english_summary}"
    
    return plan_results


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text response"""
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Look for JSON in code blocks or directly in text
    try:
        # Try to find code blocks with JSON
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if match:
            return json.loads(match.group(1))
        
        # Try to find JSON object directly
        match = re.search(r'(\{[\s\S]*\})', text)
        if match:
            return json.loads(match.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return None

# --- Test Block ---
if __name__ == '__main__':
    # Load the example analysis JSON from the previous step's output
    # IMPORTANT: Replace this with the actual JSON output from llm_analyzer
    example_analysis_report = {
        # PASTE YOUR ACTUAL llm_analyzer JSON OUTPUT HERE
      "overall_summary": {
        "num_rows": 7,
        "num_columns": 9,
        "potential_target_variable": "Grade",
        "data_quality_overview": "Dataset contains duplicates, potential PII in FullName, outliers in Score, numeric data as strings with whitespace in Attendance (%), mixed date formats with missing values in LastLogin, missing/empty text in Notes, and a constant column Campus."
      },
      "column_analysis": [
        {
          "column_name": "StudentID",
          "original_dtype": "int64",
          "inferred_semantic_type": "ID",
          "missing_percentage": 0.0,
          "unique_values_summary": {"count": 6,"is_highly_cardinal": False},
          "potential_issues": ["Duplicate IDs indicate duplicate rows"],
          "analysis_summary": "Identifier column for students. Contains duplicates, suggesting duplicate records.",
          "suggested_actions": ["Remove duplicate rows based on all columns or specific key columns"]
        },
        {
          "column_name": "FullName",
          "original_dtype": "object",
          "inferred_semantic_type": "Text",
          "missing_percentage": 0.0,
          "unique_values_summary": {"count": 6, "is_highly_cardinal": False, "top_5_categories (if Categorical/Text)": ["Alice Smith", " Bob Johnson ", "Charlie Brown", "David Williams", "Eve Davis"]},
          "potential_issues": ["Suspected PII", "Leading/trailing whitespace"],
          "analysis_summary": "Student names. Contains potential PII and leading/trailing whitespace.",
          "suggested_actions": ["Strip whitespace", "Review PII (Potentially remove or anonymize)"]
        },
        {
            "column_name": "Score",
            "original_dtype": "int64",
            "inferred_semantic_type": "Numerical",
            "missing_percentage": 0.0,
            "unique_values_summary": {"count": 6,"is_highly_cardinal": False,"numeric_range (if Numerical)": ["60","999"]},
            "potential_issues": ["Potential outliers (value 999)"],
            "analysis_summary": "Numerical score column. Contains a significant outlier.",
            "suggested_actions": ["Handle outliers (clip IQR or clip_std)"]
        },
        {
            "column_name": "Grade",
            "original_dtype": "object",
            "inferred_semantic_type": "Categorical",
            "missing_percentage": 0.0,
            "unique_values_summary": {"count": 4,"is_highly_cardinal": False,"top_5_categories (if Categorical/Text)": ["B","A","C","D"]},
            "potential_issues": [],
            "analysis_summary": "Categorical grade. Seems clean. Likely target variable.",
            "suggested_actions": ["Encode categorical (one-hot or label if target)"]
        },
        {
            "column_name": "Attendance (%)",
            "original_dtype": "object",
            "inferred_semantic_type": "Numerical",
            "missing_percentage": 0.0,
            "unique_values_summary": {"count": 6,"is_highly_cardinal": False},
            "potential_issues": ["Stored as string/object","Contains leading/trailing whitespace"],
            "analysis_summary": "Attendance percentage stored as string with whitespace. Needs conversion to numeric.",
            "suggested_actions": ["Strip whitespace","Convert to numeric (coerce errors)"]
        },
        {
            "column_name": "LastLogin",
            "original_dtype": "object",
            "inferred_semantic_type": "Datetime",
            "missing_percentage": 14.29,
            "unique_values_summary": {"count": 5,"is_highly_cardinal": False},
            "potential_issues": ["Mixed date formats","Missing values"],
            "analysis_summary": "Last login date stored as string with inconsistent formats and missing values.",
            "suggested_actions": ["Standardize date format (needs robust parsing)","Impute missing (e.g., using mode or a fixed date, depending on use case)","Extract date features"]
        },
        {
            "column_name": "Notes",
            "original_dtype": "object",
            "inferred_semantic_type": "Text",
            "missing_percentage": 14.29,
            "unique_values_summary": {"count": 5,"is_highly_cardinal": False},
            "potential_issues": ["Missing values","Contains empty strings"],
            "analysis_summary": "Free text notes. Contains missing values and empty strings.",
            "suggested_actions": ["Impute missing (e.g., with 'None' or empty string)"]
        },
        {
            "column_name": "IsActive",
            "original_dtype": "bool",
            "inferred_semantic_type": "Boolean",
            "missing_percentage": 0.0,
            "unique_values_summary": {"count": 2,"is_highly_cardinal": False},
            "potential_issues": [],
            "analysis_summary": "Boolean flag indicating activity status. Seems clean.",
            "suggested_actions": []
        },
        {
            "column_name": "Campus",
            "original_dtype": "object",
            "inferred_semantic_type": "Constant",
            "missing_percentage": 0.0,
            "unique_values_summary": {"count": 1,"is_highly_cardinal": False,"constant_value (if Constant)": "Main"},
            "potential_issues": ["Constant value (no variance)"],
            "analysis_summary": "Column contains only one value ('Main'). Provides no predictive information.",
            "suggested_actions": ["Remove column"]
        }
      ]
    }


    print("--- Generating Plan from Example Analysis (Improved Prompt) ---")
    # Make sure you have a .env file with GOOGLE_API_KEY=your_key
    plan = generate_plan_with_llm(example_analysis_report, user_goal="Predict Grade based on other features")

    print("\n--- Generated Plan (Improved) ---")
    if plan:
        print(json.dumps(plan, indent=2))
    else:
        print("Plan generation failed.")

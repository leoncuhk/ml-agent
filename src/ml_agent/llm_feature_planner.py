import openai
import os
import json
import logging
from dotenv import load_dotenv
import pandas as pd
import io
import re
from typing import Dict, Any, List, Optional
from .agent import chat_with_llm  # Import the chat_with_llm function
from .llm_interface import initialize_llm_client  # Import the modern OpenAI client initialization

# --- LLM Utility Functions (Ideally move to llm_utils.py later) ---

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize the OpenAI client using the modern approach
try:
    openai_client = initialize_llm_client()
    logger.info("Successfully initialized OpenAI client.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

def call_llm(prompt: str, system_prompt: str = "You are a helpful AI assistant expert in data analysis and machine learning.", max_tokens=2048, temperature=0.1) -> dict:
    """
    Calls the configured LLM API (currently OpenAI compatible) and attempts to parse the JSON response.

    Args:
        prompt (str): The main user prompt.
        system_prompt (str): The system prompt defining the AI's role.
        max_tokens (int): Maximum tokens for the response.
        temperature (float): Sampling temperature.

    Returns:
        dict: A dictionary containing the parsed JSON content or an error message.
              Includes 'raw_response' key with the original text content.
              Includes 'overall_summary_chinese' if found in the parsed JSON.
    """
    response_content = None
    raw_response_text = ""
    parsed_json = {}
    chinese_summary = "N/A" # Default

    try:
        logger.debug(f"System Prompt: {system_prompt}")
        logger.debug(f"User Prompt: \\n{prompt}")

        if not openai_client:
            raise ValueError("OpenAI client is not initialized.")

        # 使用现代OpenAI客户端
        completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo", # Consider making this configurable
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        if completion.choices and len(completion.choices) > 0:
            raw_response_text = completion.choices[0].message.content.strip()
            logger.debug(f"Raw LLM Response:\\n{raw_response_text}")
            try:
                # Try to find JSON within potential markdown fences
                json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response_text, re.DOTALL)
                if json_block_match:
                    json_str = json_block_match.group(1)
                else:
                    # Fallback: find first '{' and last '}'
                    json_start = raw_response_text.find('{')
                    json_end = raw_response_text.rfind('}')
                    if json_start != -1 and json_end != -1:
                        json_str = raw_response_text[json_start:json_end+1]
                    else:
                        raise json.JSONDecodeError("No JSON object found", raw_response_text, 0)

                parsed_json = json.loads(json_str)
                logger.info("Successfully parsed JSON response from LLM.")
                chinese_summary = parsed_json.get('overall_summary_chinese', parsed_json.get('summary_chinese', 'N/A (Not found in JSON)'))

            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to parse LLM response as JSON: {json_e}")
                parsed_json = {"error": f"JSONDecodeError: {json_e}", "raw_response_preview": raw_response_text[:500]}
        else:
            logger.error("LLM API call did not return valid choices or message.")
            parsed_json = {"error": "Invalid response structure from LLM API."}

    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)
        parsed_json = {"error": f"Unexpected error: {e}"}

    if not isinstance(parsed_json, dict):
        parsed_json = {"error": "Parsing resulted in non-dict object", "parsed_result": str(parsed_json)}

    parsed_json['raw_response'] = raw_response_text
    if 'overall_summary_chinese' not in parsed_json and 'summary_chinese' not in parsed_json:
         parsed_json['overall_summary_chinese'] = chinese_summary # Ensure it's always present

    return parsed_json


def extract_code_plan(llm_response: Any) -> list:
    """
    Extracts the plan steps (list of dictionaries) from the LLM response.
    Handles both dictionary and string response formats.
    Returns the list or an empty list if extraction fails or an error occurred.
    """
    # If the response is a string, try to extract JSON
    if isinstance(llm_response, str):
        try:
            # Try to find JSON within potential markdown fences
            json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
            else:
                # Fallback: find first '{' and last '}'
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_str = llm_response[json_start:json_end+1]
                else:
                    logger.error("Could not find JSON object in string response")
                    return []

            llm_response = json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse string response as JSON: {e}")
            return []

    # Now llm_response should be a dictionary (if parsing succeeded)
    if not isinstance(llm_response, dict):
        logger.error(f"Response is not a dictionary after parsing: {type(llm_response)}")
        return []

    if "error" in llm_response:
        logger.error(f"Cannot extract plan due to previous error: {llm_response['error']}")
        return [] # Return empty list on error

    plan = llm_response.get('plan_steps', llm_response.get('plan'))

    if isinstance(plan, list):
        valid_steps = []
        for i, step in enumerate(plan):
            if isinstance(step, dict) and 'module' in step and 'function' in step and 'args' in step:
                valid_steps.append(step)
            else:
                logger.warning(f"Step {i+1} in the plan is invalid or missing keys: {step}. Skipping.")
        if not valid_steps and plan: # Log if we had steps but none were valid
             logger.warning(f"Extracted plan list, but no valid steps found.")
        elif valid_steps:
             logger.info(f"Extracted {len(valid_steps)} valid plan steps.")
        return valid_steps
    else:
        logger.error(f"Could not find a valid 'plan_steps' or 'plan' list in the LLM response.")
        return []

# --- Feature Engineering Planner Logic ---

def generate_fe_plan(analysis_report: Dict[str, Any], 
                    data_so_far: pd.DataFrame,  
                    prep_evaluation: Dict[str, Any] = None,
                    user_instructions: str = None,
                    language: str = 'en') -> List[Dict[str, Any]]:
    """
    Generate a feature engineering plan including transformation and creation steps.

    Args:
        analysis_report: Original data analysis report
        data_so_far: DataFrame after data preparation stage
        prep_evaluation: Data preparation evaluation report
        user_instructions: User instructions
        language: Language for output - 'en' for English only, 'cn' for Chinese and English

    Returns:
        List[Dict]: List of feature engineering steps, each containing module, function, parameters and reasoning
    """
    if language == 'cn':
        logger.info("开始生成特征工程计划...")
    else:
        logger.info("Starting feature engineering plan generation...")
    
    # Extract target variable (if exists)
    target_variable = None
    if analysis_report:
        # Check if analysis_report is a string and try to parse it as JSON
        if isinstance(analysis_report, str):
            try:
                analysis_report = json.loads(analysis_report)
            except json.JSONDecodeError:
                if language == 'cn':
                    logger.warning(f"分析报告是一个字符串且无法解析为JSON")
                else:
                    logger.warning(f"Analysis report is a string and couldn't be parsed as JSON")
                analysis_report = {"overall_summary": analysis_report}
        
        # Now safely access the dictionary
        if isinstance(analysis_report, dict):
            # Try direct access first
            target_variable = analysis_report.get('potential_target_variable')
            # Try nested in overall_summary if it's a dict
            if target_variable is None and isinstance(analysis_report.get('overall_summary'), dict):
                target_variable = analysis_report.get('overall_summary', {}).get('potential_target_variable')
            
        if language == 'cn':
            logger.info(f"从分析报告中提取的目标变量: {target_variable}")
        else:
            logger.info(f"Extracted target variable from analysis: {target_variable}")
    
    # Generate data summary
    df_info = _get_df_summary(data_so_far, target_variable)
    
    # Handle prep_evaluation
    prep_eval_summary = "Data preparation evaluation not provided"
    if prep_evaluation:
        # Check if prep_evaluation is a string and try to parse it as JSON
        if isinstance(prep_evaluation, str):
            try:
                prep_evaluation = json.loads(prep_evaluation)
            except json.JSONDecodeError:
                if language == 'cn':
                    logger.warning(f"数据准备评估是一个字符串且无法解析为JSON")
                else:
                    logger.warning(f"Data prep evaluation is a string and couldn't be parsed as JSON")
                prep_eval_summary = prep_evaluation
        
        # Now safely access the dictionary
        if isinstance(prep_evaluation, dict):
            # Use language-appropriate summary if available
            if language == 'cn' and 'overall_summary_chinese' in prep_evaluation:
                prep_eval_summary = prep_evaluation.get('overall_summary_chinese')
            else:
                prep_eval_summary = prep_evaluation.get('overall_summary', prep_eval_summary)
    
    # Build prompt
    prompt = f"""
You are a machine learning feature engineering expert tasked with designing feature engineering steps to optimize model performance.
Please design a series of feature engineering steps based on the characteristics of the dataset after the data preparation stage.

### Data Summary:
{df_info}

### Data Preparation Stage Evaluation:
{prep_eval_summary}

### User Instructions:
{user_instructions if user_instructions else "No specific instructions provided"}

### Available Feature Engineering Functions:
- feature_engineer.scale_numerical(df, columns, strategy) - Scale numerical features, strategies: 'standard', 'minmax', 'robust'
- feature_engineer.bin_numerical(df, column, bins, strategy) - Bin numerical features, strategies: 'uniform', 'quantile', 'kmeans'
- feature_engineer.encode_categorical(df, column, strategy) - Encode categorical features, strategies: 'one-hot', 'label', 'target', 'frequency'
- feature_engineer.handle_outliers(df, columns, strategy) - Handle outliers, strategies: 'clip_iqr', 'clip_std', 'winsorize'
- feature_engineer.handle_high_cardinality(df, column, threshold, method) - Handle high cardinality features, methods: 'frequency', 'grouping'
- feature_engineer.create_polynomial_features(df, columns, degree) - Create polynomial features
- feature_engineer.create_interaction_features(df, columns1, columns2, interaction_type) - Create interaction features, types: 'multiply', 'divide'
- feature_engineer.decompose_datetime(df, column) - Decompose datetime features into year, month, day, hour, etc.
- feature_engineer.drop_low_variance(df, threshold) - Remove low variance features
- feature_engineer.drop_highly_correlated(df, threshold) - Remove highly correlated features

Please design a feature engineering plan with 5-10 steps. For each step, specify:
1. The function to use
2. Target columns/features
3. Strategy/method to use
4. Reasoning for performing this step

Return a JSON object with the following keys:
- "plan": A list where each element is a step object containing:
  - "step": Step number
  - "module": Module name (fixed as "feature_engineer")
  - "function": Function name (one of the functions listed above)
  - "args": Object with function parameters
  - "reasoning": Explanation of why this step is necessary"""

    if language == 'cn':
        prompt += """
- "summary_chinese": A concise Chinese summary of the entire feature engineering plan
"""

    prompt += """
For example:
```json
{
  "plan": [
    {
      "step": 1,
      "module": "feature_engineer",
      "function": "handle_outliers",
      "args": {
        "columns": ["value1", "value2"],
        "strategy": "clip_iqr"
      },
      "reasoning": "Clip outliers in numerical features to improve model stability."
    },
    {
      "step": 2,
      "module": "feature_engineer",
      "function": "scale_numerical",
      "args": {
        "columns": ["value1", "value2"],
        "strategy": "standard"
      },
      "reasoning": "Standardize numerical features to improve model performance."
    }
  ]"""

    if language == 'cn':
        prompt += """,
  "summary_chinese": "特征工程计划旨在通过处理异常值、标准化数值特征和编码分类变量来提升模型性能。"
"""
    else:
        prompt += """
"""

    prompt += """}
```

Focus on preparing the data optimally for machine learning. Consider the target variable (if known) when designing your feature engineering steps.
"""

    logger.debug(f"Generated feature engineering prompt with {len(prompt)} characters")
    
    # Call LLM with our prompt
    if language == 'cn':
        logger.info("正在发送请求... (尝试 1/3)")
    else:
        logger.info("Sending request... (attempt 1/3)")
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Use chat_with_llm function which encapsulates the API call
            response = chat_with_llm(prompt)
            
            # Extract the plan steps from the response
            fe_plan = extract_code_plan(response)
            
            if fe_plan:
                # Add the Chinese summary to the first step if available
                if language == 'cn' and isinstance(response, dict):
                    summary_chinese = response.get('summary_chinese', '')
                    if summary_chinese and fe_plan and isinstance(fe_plan[0], dict):
                        fe_plan[0]['summary_chinese'] = summary_chinese
                    logger.info(f"特征工程计划生成完成，包含 {len(fe_plan)} 个步骤")
                    logger.info(f"计划中文总结: {summary_chinese}")
                else:
                    logger.info(f"Feature engineering plan generated with {len(fe_plan)} steps")
                
                return fe_plan
            
            logger.warning(f"Attempt {attempt+1}: Failed to extract valid plan steps, retrying...")
            
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed with error: {str(e)}", exc_info=True)
            if attempt == max_attempts - 1:
                raise
    
    # If we get here, all attempts failed
    logger.error("Failed to generate feature engineering plan after maximum attempts")
    return []

def _get_df_summary(df: pd.DataFrame, target_variable: Optional[str] = None) -> str:
    """生成数据集的摘要描述"""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    summary = f"### 数据摘要\n"
    summary += f"- 形状: {df.shape[0]} 行, {df.shape[1]} 列\n"
    summary += f"- 缺失值总数: {df.isna().sum().sum()} 个\n\n"
    
    # 列类型分布
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    summary += f"### 列类型分布\n"
    summary += f"- 数值列: {len(numeric_cols)} 列\n"
    if numeric_cols:
        summary += f"  - 例如: {', '.join(numeric_cols[:3])}{'...' if len(numeric_cols) > 3 else ''}\n"
    summary += f"- 分类列: {len(cat_cols)} 列\n"
    if cat_cols:
        summary += f"  - 例如: {', '.join(cat_cols[:3])}{'...' if len(cat_cols) > 3 else ''}\n"
    if datetime_cols:
        summary += f"- 日期时间列: {len(datetime_cols)} 列\n"
        summary += f"  - 例如: {', '.join(datetime_cols[:3])}{'...' if len(datetime_cols) > 3 else ''}\n"
    
    # 缺失值分析
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        summary += f"\n### 缺失值分布\n"
        for col, count in missing.items():
            summary += f"- {col}: {count} 个缺失值 ({count/len(df)*100:.1f}%)\n"
    
    # 数值列统计
    if numeric_cols:
        summary += f"\n### 数值列统计\n"
        for col in numeric_cols[:min(5, len(numeric_cols))]:  # 限制显示前5列
            summary += f"- {col}: "
            try:
                stats = df[col].describe()
                summary += f"均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 最小值={stats['min']:.2f}, 最大值={stats['max']:.2f}\n"
            except:
                summary += f"无法计算统计信息\n"
        if len(numeric_cols) > 5:
            summary += f"  (仅显示前5列，共{len(numeric_cols)}列)\n"
    
    # 分类列统计
    if cat_cols:
        summary += f"\n### 分类列统计\n"
        for col in cat_cols[:min(3, len(cat_cols))]:  # 限制显示前3列
            summary += f"- {col}: {df[col].nunique()} 个不同值"
            if df[col].nunique() <= 5:
                summary += f", 值: {', '.join(str(x) for x in df[col].unique()[:5])}\n"
            else:
                summary += f", 前几个值: {', '.join(str(x) for x in df[col].value_counts().head(3).index)}\n"
        if len(cat_cols) > 3:
            summary += f"  (仅显示前3列，共{len(cat_cols)}列)\n"
    
    # 如果存在目标变量，进行特殊分析
    if target_variable and target_variable in df.columns:
        summary += f"\n### 目标变量 '{target_variable}' 分析\n"
        if df[target_variable].dtype.name in ['object', 'category']:
            value_counts = df[target_variable].value_counts()
            summary += f"- 类型: 分类变量\n"
            summary += f"- 唯一值数量: {df[target_variable].nunique()}\n"
            summary += f"- 类别分布:\n"
            for val, count in value_counts.items():
                pct = count / len(df) * 100
                summary += f"  - {val}: {count} ({pct:.1f}%)\n"
            summary += f"- 推断任务类型: 分类\n"
        else:
            summary += f"- 类型: 数值变量\n"
            try:
                stats = df[target_variable].describe()
                summary += f"- 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}, 最小值={stats['min']:.2f}, 最大值={stats['max']:.2f}\n"
            except:
                summary += f"- 无法计算统计信息\n"
            summary += f"- 推断任务类型: 回归\n"
    
    # 潜在的高基数分类特征
    high_cardinality_cols = [col for col in cat_cols if df[col].nunique() > 10]
    if high_cardinality_cols:
        summary += f"\n### 高基数分类特征\n"
        for col in high_cardinality_cols[:min(5, len(high_cardinality_cols))]:
            summary += f"- {col}: {df[col].nunique()} 个不同值\n"
        if len(high_cardinality_cols) > 5:
            summary += f"  (仅显示前5列，共{len(high_cardinality_cols)}列)\n"
    
    # 样本展示
    summary += f"\n### 数据样本 (前3行)\n"
    try:
        summary += df.head(3).to_string()
    except:
        summary += "无法显示样本数据"
    
    return summary

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取JSON对象"""
    try:
        # 尝试直接解析整个响应
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试查找边界字符
        json_start_markers = ["{", "```json\n{", "```\n{"]
        json_end_markers = ["}", "}\n```"]
        
        json_text = None
        
        # 尝试各种可能的组合
        for start_marker in json_start_markers:
            if start_marker in text:
                start_idx = text.find(start_marker)
                # 如果找到起始标记，移动到实际的 { 位置
                if start_marker.startswith("```"):
                    start_idx = text.find("{", start_idx)
                
                for end_marker in json_end_markers:
                    end_idx = text.rfind(end_marker)
                    if end_idx > start_idx:
                        # 确保结束位置包含 }
                        end_idx = end_idx + 1 if end_marker == "}" else end_idx
                        json_text = text[start_idx:end_idx]
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError:
                            continue  # 尝试下一个结束标记
        
        return None

def _generate_fallback_plan(df: pd.DataFrame, target_variable: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    生成备用特征工程计划，当LLM响应无法解析时使用
    """
    logger.info("生成备用特征工程计划...")
    
    # 初始化计划
    plan = []
    step_num = 1
    
    # 获取列分布
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # 排除目标变量
    if target_variable:
        numeric_cols = [col for col in numeric_cols if col != target_variable]
        cat_cols = [col for col in cat_cols if col != target_variable]
        datetime_cols = [col for col in datetime_cols if col != target_variable]
    
    # 1. 处理数值特征的异常值
    if numeric_cols:
        plan.append({
            "step": step_num,
            "module": "feature_engineer",
            "function": "handle_outliers",
            "args": {
                "columns": numeric_cols[:min(5, len(numeric_cols))],  # 取前5个
                "strategy": "clip_iqr",
                "iqr_multiplier": 1.5
            },
            "reasoning": "处理数值特征中的异常值，使用IQR方法裁剪，避免极端值影响模型训练。",
            "summary_chinese": "总结：该特征工程计划包括处理异常值、缩放数值特征、编码分类特征等步骤，旨在提高数据质量并为后续建模做准备。"
        })
        step_num += 1
    
    # 2. 缩放数值特征
    if numeric_cols:
        plan.append({
            "step": step_num,
            "module": "feature_engineer",
            "function": "scale_numerical",
            "args": {
                "columns": numeric_cols[:min(5, len(numeric_cols))],  # 取前5个
                "strategy": "standard"
            },
            "reasoning": "标准化数值特征，使其均值为0，标准差为1，适合大多数机器学习算法。"
        })
        step_num += 1
    
    # 3. 编码分类特征
    if cat_cols:
        for col in cat_cols[:min(3, len(cat_cols))]:  # 处理前3个分类特征
            if df[col].nunique() <= 10:  # 低基数
                plan.append({
                    "step": step_num,
                    "module": "feature_engineer",
                    "function": "encode_categorical",
                    "args": {
                        "column": col,
                        "strategy": "one-hot",
                        "drop_original": True
                    },
                    "reasoning": f"对低基数分类特征'{col}'进行独热编码，适合大多数模型。"
                })
            else:  # 高基数
                plan.append({
                    "step": step_num,
                    "module": "feature_engineer",
                    "function": "handle_high_cardinality",
                    "args": {
                        "column": col,
                        "threshold": 0.95,
                        "method": "frequency"
                    },
                    "reasoning": f"处理高基数分类特征'{col}'，使用频率编码减少类别数量。"
                })
            step_num += 1
    
    # 4. 处理日期时间特征
    if datetime_cols:
        for col in datetime_cols[:min(2, len(datetime_cols))]:  # 处理前2个日期时间特征
            plan.append({
                "step": step_num,
                "module": "feature_engineer",
                "function": "decompose_datetime",
                "args": {
                    "column": col
                },
                "reasoning": f"分解日期时间特征'{col}'为年、月、日、星期几等，提取时间模式。"
            })
            step_num += 1
    
    # 5. 移除高相关特征（只有当数值特征>=5个时）
    if len(numeric_cols) >= 5:
        plan.append({
            "step": step_num,
            "module": "feature_engineer",
            "function": "drop_highly_correlated",
            "args": {
                "threshold": 0.95
            },
            "reasoning": "移除高度相关的特征（相关系数>0.95），减少特征冗余和多重共线性问题。"
        })
        step_num += 1
    
    logger.info(f"备用特征工程计划生成完成，包含 {len(plan)} 个步骤")
    return plan

def _generate_chinese_summary(english_summary: str) -> str:
    """
    为特征工程计划生成中文总结
    这是一个简单的实现，实际应用中可能需要更复杂的翻译逻辑
    """
    # 尝试匹配常见术语的简单翻译
    translations = {
        "feature engineering": "特征工程",
        "plan": "计划",
        "steps": "步骤",
        "focusing on": "专注于",
        "scaling": "特征缩放",
        "handling outliers": "处理异常值",
        "encoding categorical features": "编码分类特征",
        "polynomial features": "多项式特征",
        "interaction features": "交互特征",
        "datetime features": "日期时间特征",
        "high cardinality": "高基数",
        "correlation": "相关性"
    }
    
    # 构建一个简单的中文摘要
    summary = english_summary
    for eng, chn in translations.items():
        summary = re.sub(r'\b' + re.escape(eng) + r'\b', chn, summary, flags=re.IGNORECASE)
    
    # 如果摘要为空，生成默认摘要
    if not summary:
        summary = "特征工程计划包括特征缩放、异常值处理和分类特征编码等步骤，旨在提高数据质量并为后续建模做准备。"
    
    # 确保摘要以句号结尾
    if not summary.endswith('。'):
        summary += '。'
    
    # 统一添加前缀 "总结："
    if not summary.startswith("总结："):
        summary = "总结：" + summary
    
    return summary

# --- End Feature Engineering Planner Logic --- 

def plan_feature_engineering(df: pd.DataFrame, data_prep_results: Dict[str, Any], 
                          existing_columns: List[str] = None, 
                          target_column: str = None,
                          language: str = 'en') -> Dict[str, Any]:
    """
    Create a detailed feature engineering plan using LLM based on data preparation results.
    
    Args:
        df: DataFrame after data preparation
        data_prep_results: Results from the data preparation stage
        existing_columns: List of column names in the original dataset
        target_column: Name of the target column if known
        language: Output language - 'en' for English, 'cn' for Chinese and English
        
    Returns:
        Dict containing the feature engineering plan
    """
    if language == 'cn':
        logger.info("开始生成特征工程计划...")
    else:
        logger.info("Starting generation of feature engineering plan...")
    
    # If existing columns not provided, get them from the dataframe
    if existing_columns is None:
        existing_columns = list(df.columns)
    
    # Get info about the dataframe
    df_info = _get_dataframe_info(df)
    
    # Get info from data_prep_results
    data_prep_plan = data_prep_results.get('data_prep_plan', [])
    data_prep_summary = data_prep_results.get('data_preparation_summary', '')
    data_prep_summary_chinese = data_prep_results.get('data_preparation_summary_chinese', '')
    expected_outcomes = data_prep_results.get('expected_outcomes', [])
    
    # Create simplified info for the LLM
    # Get column types
    column_info = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            dtype = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dtype = 'datetime'
        else:
            # Check if it's likely a categorical column
            if df[col].nunique() < min(20, len(df) * 0.05):  # Less than 5% unique values or less than 20 categories
                dtype = 'categorical'
            else:
                dtype = 'text'
        
        # Get basic stats
        num_missing = df[col].isna().sum()
        pct_missing = 100 * num_missing / len(df) if len(df) > 0 else 0
        num_unique = df[col].nunique()
        
        column_info.append({
            'column': col,
            'python_dtype': col_type,
            'semantic_type': dtype,
            'missing_count': int(num_missing),
            'missing_percentage': float(pct_missing),
            'unique_values': int(num_unique)
        })
    
    # Create prompt for feature engineering
    if language == 'cn':
        # Chinese prompt
        prompt = f"""
你是一位专业的数据科学家，负责数据特征工程。请基于以下信息创建详细的特征工程计划：

### 数据信息
- 行数: {df_info['num_rows']}
- 列数: {df_info['num_cols']}
- 内存使用: {df_info['memory_usage']}
- 缺失值总数: {df_info['missing_values']}

### 列信息
```
{json.dumps(column_info, indent=2, ensure_ascii=False)}
```

### 目标变量
{target_column if target_column else "未指定"}

### 数据准备摘要
{data_prep_summary_chinese if language == 'cn' and data_prep_summary_chinese else data_prep_summary}

### 数据准备计划摘要
```
{json.dumps([{"step_number": step.get("step_number", i+1), 
              "step_name": step.get("step_name", ""), 
              "function": step.get("function", "")} 
             for i, step in enumerate(data_prep_plan)], indent=2, ensure_ascii=False)}
```

请创建一个详细且可操作的特征工程计划，包含以下内容：
1. 一系列具体的特征创建、转换和选择步骤
2. 每个步骤的理由说明
3. 用于每个步骤的Python函数调用

请提供以下JSON格式的计划：
```json
{{
  "feature_eng_plan": [
    {{
      "step_number": 1,
      "step_name": "步骤的简洁名称",
      "reasoning": "为什么这个步骤有用",
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
  "feature_engineering_summary": "英文特征工程计划的总结",
  "feature_engineering_summary_chinese": "中文特征工程计划的总结"
}}
```

请确保：
1. 功能强大的特征工程计划，包含分组、交互、聚合、多项式或其他高级功能
2. 函数名称选自常见的特征工程函数：
   - create_polynomial_features: 创建多项式特征
   - create_interaction_features: 创建交互特征
   - bin_numeric_feature: 将数值特征分箱
   - create_cluster_features: 创建聚类特征
   - extract_text_features: 从文本特征中提取信息
   - encode_categorical: 编码分类特征
   - create_date_features: 从日期创建特征（如果未在数据准备阶段完成）
   - normalize_features: 规范化特征
   - create_aggregation_features: 在分组数据上创建聚合特征
   - reduce_dimensions: 减少数据维度
   - select_features: 基于重要性或其他指标选择特征
   - engineer_domain_specific: 添加领域特定的特征
3. 参数是适合该特定上下文的适当值
4. 计划全面考虑数据的特点
5. 考虑目标变量（如果提供）的特定特征

请确保JSON格式正确，计划逻辑清晰，步骤具体。
"""
    else:
        # English prompt
        prompt = f"""
You are a professional data scientist specialized in feature engineering. Please create a detailed feature engineering plan based on the following information:

### Data Information
- Rows: {df_info['num_rows']}
- Columns: {df_info['num_cols']}
- Memory usage: {df_info['memory_usage']}
- Total missing values: {df_info['missing_values']}

### Column Information
```
{json.dumps(column_info, indent=2)}
```

### Target Variable
{target_column if target_column else "Not specified"}

### Data Preparation Summary
{data_prep_summary}

### Data Preparation Plan Summary
```
{json.dumps([{"step_number": step.get("step_number", i+1), 
              "step_name": step.get("step_name", ""), 
              "function": step.get("function", "")} 
             for i, step in enumerate(data_prep_plan)], indent=2)}
```

Please create a detailed, actionable feature engineering plan that includes:
1. A series of specific feature creation, transformation, and selection steps
2. Reasoning for each step
3. Python function calls for each step

Provide the plan in the following JSON format:
```json
{{
  "feature_eng_plan": [
    {{
      "step_number": 1,
      "step_name": "Concise name for the step",
      "reasoning": "Why this step is useful",
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
  "feature_engineering_summary": "Summary of the feature engineering plan"
}}
```

Please ensure:
1. A powerful feature engineering plan that includes grouping, interactions, aggregations, polynomials, or other advanced features
2. Function names from common feature engineering functions:
   - create_polynomial_features: Create polynomial features
   - create_interaction_features: Create interaction features
   - bin_numeric_feature: Bin numeric features
   - create_cluster_features: Create cluster features
   - extract_text_features: Extract information from text features
   - encode_categorical: Encode categorical features
   - create_date_features: Create features from dates (if not done in data prep)
   - normalize_features: Normalize features
   - create_aggregation_features: Create aggregation features on grouped data
   - reduce_dimensions: Reduce dimensions of the data
   - select_features: Select features based on importance or other metrics
   - engineer_domain_specific: Add domain-specific features
3. Parameters are appropriate values for the specific context
4. The plan comprehensively considers the characteristics of the data
5. Consideration for the target variable (if provided) specific features

Ensure the JSON is well-formed and the plan is logical with concrete steps.
"""

    try:
        # Call LLM with the prompt
        response = call_llm(prompt)
        
        # Parse the response to extract JSON
        feature_plan = _extract_json_from_response(response)
        
        if not feature_plan:
            if language == 'cn':
                logger.warning("LLM未返回有效的JSON结构，将使用基本特征工程计划。")
            else:
                logger.warning("LLM didn't return a valid JSON structure, using basic feature engineering plan.")
            feature_plan = _create_basic_feature_plan(df, target_column, language)
        
        # Validate and fix the plan if needed
        feature_plan = _validate_feature_plan(feature_plan, df, language)
        
        if language == 'cn':
            logger.info("特征工程计划已生成，包含 %d 个步骤", len(feature_plan.get('feature_eng_plan', [])))
            if 'feature_engineering_summary_chinese' in feature_plan:
                logger.info("中文总结: %s", feature_plan.get('feature_engineering_summary_chinese'))
        else:
            logger.info("Feature engineering plan generated with %d steps", len(feature_plan.get('feature_eng_plan', [])))
        
        return feature_plan
        
    except Exception as e:
        if language == 'cn':
            logger.error(f"生成特征工程计划时出错: {str(e)}")
        else:
            logger.error(f"Error generating feature engineering plan: {str(e)}")
        return _create_basic_feature_plan(df, target_column, language)


def _get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic information about the DataFrame"""
    info = {
        'num_rows': len(df),
        'num_cols': len(df.columns),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB",
        'missing_values': df.isna().sum().sum()
    }
    return info


def _create_basic_feature_plan(df: pd.DataFrame, target_column: str = None, language: str = 'en') -> Dict[str, Any]:
    """
    Create a basic feature engineering plan when LLM fails.
    
    Args:
        df: DataFrame to engineer features for
        target_column: Name of the target column if known
        language: Output language - 'en' for English, 'cn' for Chinese and English
        
    Returns:
        Dict containing a basic feature engineering plan
    """
    if language == 'cn':
        logger.info("创建基本特征工程计划...")
    else:
        logger.info("Creating basic feature engineering plan...")
    
    # Initialize empty plan
    feature_eng_plan = []
    step_num = 1
    
    # Find categorical columns
    categorical_cols = []
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or (
                pd.api.types.is_categorical_dtype(df[col]) or 
                (df[col].nunique() < min(20, len(df) * 0.05) and not pd.api.types.is_numeric_dtype(df[col]))
            ):
            if col != target_column:  # Don't include target column
                categorical_cols.append(col)
    
    # Step 1: Encode categorical variables if any exist
    if categorical_cols:
        feature_eng_plan.append({
            "step_number": step_num,
            "step_name": "Encode categorical variables" if language == 'en' else "编码分类变量",
            "reasoning": "Convert categorical variables to numeric format for ML models" if language == 'en' else "将分类变量转换为ML模型的数值格式",
            "function": "encode_categorical",
            "params": {
                "columns": categorical_cols,
                "method": "auto"
            }
        })
        step_num += 1
    
    # Find numeric columns
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col != target_column:
            numeric_cols.append(col)
    
    # Step 2: Create polynomial features for numeric columns
    if len(numeric_cols) >= 2:
        selected_cols = numeric_cols[:min(5, len(numeric_cols))]  # Limit to 5 columns to avoid explosion
        feature_eng_plan.append({
            "step_number": step_num,
            "step_name": "Create polynomial features" if language == 'en' else "创建多项式特征",
            "reasoning": "Capture non-linear relationships between numeric features" if language == 'en' else "捕获数值特征之间的非线性关系",
            "function": "create_polynomial_features",
            "params": {
                "columns": selected_cols,
                "degree": 2
            }
        })
        step_num += 1
    
    # Step 3: Create interaction features
    if len(numeric_cols) >= 2:
        selected_cols = numeric_cols[:min(5, len(numeric_cols))]  # Limit to 5 columns
        feature_eng_plan.append({
            "step_number": step_num,
            "step_name": "Create interaction features" if language == 'en' else "创建交互特征",
            "reasoning": "Capture interactions between features" if language == 'en' else "捕获特征之间的交互",
            "function": "create_interaction_features",
            "params": {
                "columns": selected_cols
            }
        })
        step_num += 1
    
    # Step 4: Normalize numeric features
    if numeric_cols:
        feature_eng_plan.append({
            "step_number": step_num,
            "step_name": "Normalize numeric features" if language == 'en' else "标准化数值特征",
            "reasoning": "Scale features to have similar ranges for better model performance" if language == 'en' else "将特征缩放到相似范围以获得更好的模型性能",
            "function": "normalize_features",
            "params": {
                "columns": numeric_cols,
                "method": "standard"
            }
        })
        step_num += 1
    
    # Create expected outcomes
    expected_outcomes = []
    if language == 'en':
        expected_outcomes.append("Enhanced dataset with engineered features")
        expected_outcomes.append("Features that better capture relationships in the data")
        expected_outcomes.append("Improved model performance through better feature representation")
        summary = "Basic feature engineering plan focusing on categorical encoding, polynomial features, interactions, and feature scaling."
    else:
        expected_outcomes.append("具有工程化特征的增强数据集")
        expected_outcomes.append("更好地捕获数据中关系的特征")
        expected_outcomes.append("通过更好的特征表示提高模型性能")
        summary = "专注于分类编码、多项式特征、交互和特征缩放的基本特征工程计划。"
        summary_chinese = "基本特征工程计划，主要关注分类变量编码、多项式特征创建、特征交互和数值特征标准化。"
    
    # Assemble the plan
    feature_plan = {
        "feature_eng_plan": feature_eng_plan,
        "expected_outcomes": expected_outcomes,
        "feature_engineering_summary": summary
    }
    
    # Add Chinese summary if needed
    if language == 'cn':
        feature_plan["feature_engineering_summary_chinese"] = summary_chinese
    
    if language == 'cn':
        logger.info("基本特征工程计划已创建，包含 %d 个步骤", len(feature_eng_plan))
    else:
        logger.info("Basic feature engineering plan created with %d steps", len(feature_eng_plan))
    
    return feature_plan


def _validate_feature_plan(feature_plan: Dict[str, Any], df: pd.DataFrame, language: str = 'en') -> Dict[str, Any]:
    """
    Validate and fix the feature engineering plan if needed.
    
    Args:
        feature_plan: The LLM-generated feature engineering plan
        df: DataFrame used for feature engineering
        language: Output language - 'en' for English, 'cn' for Chinese and English
        
    Returns:
        Validated and fixed feature engineering plan
    """
    # Check required keys
    required_keys = ["feature_eng_plan", "expected_outcomes", "feature_engineering_summary"]
    missing_keys = [key for key in required_keys if key not in feature_plan]
    
    if missing_keys:
        if language == 'cn':
            logger.warning("特征工程计划缺少以下键: %s", missing_keys)
        else:
            logger.warning("Feature engineering plan missing keys: %s", missing_keys)
        
        # Create missing keys with default values
        for key in missing_keys:
            if key == "feature_eng_plan":
                feature_plan[key] = []
            elif key == "expected_outcomes":
                feature_plan[key] = ["Enhanced dataset with engineered features"]
            elif key == "feature_engineering_summary":
                feature_plan[key] = "Feature engineering plan to enhance the dataset."
    
    # Ensure all necessary keys in each step
    valid_steps = []
    for step in feature_plan.get("feature_eng_plan", []):
        # Check if required step fields are present
        if not all(k in step for k in ["step_number", "step_name", "function", "params"]):
            if language == 'cn':
                logger.warning("跳过无效的步骤: %s", step.get("step_name", "未命名步骤"))
            else:
                logger.warning("Skipping invalid step: %s", step.get("step_name", "Unnamed step"))
            continue
            
        # Add reasoning if missing
        if "reasoning" not in step:
            step["reasoning"] = "Improve model performance" if language == 'en' else "提高模型性能"
        
        # Validate function name
        valid_functions = [
            "create_polynomial_features", "create_interaction_features", "bin_numeric_feature",
            "create_cluster_features", "extract_text_features", "encode_categorical",
            "create_date_features", "normalize_features", "create_aggregation_features",
            "reduce_dimensions", "select_features", "engineer_domain_specific"
        ]
        
        if step["function"] not in valid_functions:
            if language == 'cn':
                logger.warning("无效的函数名称 '%s'，将被跳过", step["function"])
            else:
                logger.warning("Invalid function name '%s', will be skipped", step["function"])
            continue
        
        # Ensure params is a dictionary
        if not isinstance(step["params"], dict):
            if language == 'cn':
                logger.warning("步骤 '%s' 的参数不是字典，设置为空字典", step.get("step_name", ""))
            else:
                logger.warning("Params for step '%s' is not a dictionary, setting to empty dict", step.get("step_name", ""))
            step["params"] = {}
        
        valid_steps.append(step)
    
    # Update with valid steps
    feature_plan["feature_eng_plan"] = valid_steps
    
    # Fix step numbers to ensure they're sequential
    for i, step in enumerate(valid_steps):
        step["step_number"] = i + 1
    
    # Ensure Chinese summary exists if needed
    if language == 'cn' and "feature_engineering_summary_chinese" not in feature_plan:
        english_summary = feature_plan.get("feature_engineering_summary", "特征工程计划")
        feature_plan["feature_engineering_summary_chinese"] = f"中文特征工程总结: {english_summary}"
    
    return feature_plan


def _extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response text"""
    try:
        # Try standard JSON decoding first
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from text that might contain non-JSON elements
        try:
            # Look for JSON in code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                return json.loads(json_match.group(1))
            
            # Try to find any JSON object directly
            json_match = re.search(r'(\{[\s\S]*\})', response)
            if json_match:
                return json.loads(json_match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
    
    return None 
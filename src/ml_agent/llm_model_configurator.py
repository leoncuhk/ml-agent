import json
import logging
import pandas as pd
import os
import io
from typing import Dict, Any, Optional, List

# 导入LLM接口
from .llm_interface import initialize_llm_client
from .llm_feature_planner import call_llm
from .utils import setup_logging

# 配置模块级logger
logger = logging.getLogger("ml_agent.llm_model_configurator")

def get_df_info_summary(df: pd.DataFrame, target_variable: Optional[str] = None, 
                       max_samples: int = 5) -> str:
    """
    为模型配置生成数据摘要
    
    Args:
        df: 特征工程后的DataFrame
        target_variable: 目标变量名
        max_samples: 样本数量上限
    
    Returns:
        str: 数据摘要描述
    """
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    summary = f"### 数据摘要\n"
    summary += f"- 形状: {df.shape[0]} 行, {df.shape[1]} 列\n"
    summary += f"- 缺失值: {df.isna().sum().sum()} 个\n\n"
    
    if target_variable and target_variable in df.columns:
        summary += f"### 目标变量 '{target_variable}' 分析\n"
        if df[target_variable].dtype.name in ['object', 'category']:
            value_counts = df[target_variable].value_counts()
            summary += f"- 类型: 分类变量\n"
            summary += f"- 唯一值数量: {df[target_variable].nunique()}\n"
            summary += f"- 类别分布:\n"
            for val, count in value_counts.items():
                pct = count / len(df) * 100
                summary += f"  - {val}: {count} ({pct:.1f}%)\n"
            task_type = "classification"
        else:
            summary += f"- 类型: 数值变量\n"
            summary += f"- 统计摘要: {df[target_variable].describe().to_dict()}\n"
            task_type = "regression"
        
        # 自动推断任务类型
        summary += f"\n推断的任务类型: {task_type}\n\n"
    
    # 特征摘要
    summary += f"### 特征摘要\n"
    feature_cols = [col for col in df.columns if col != target_variable]
    numeric_cols = df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    cat_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
    
    summary += f"- 数值特征数量: {len(numeric_cols)}\n"
    summary += f"- 分类特征数量: {len(cat_cols)}\n"
    
    # 样本展示
    if len(df) > 0:
        summary += f"\n### 数据样本 (前{min(max_samples, len(df))}行)\n"
        summary += df.head(max_samples).to_string()
    
    return summary

def generate_h2o_config(analysis_report, fe_evaluation, data_snapshot, user_instructions="", language="en", execution_plan=None):
    """
    Generate H2O AutoML configuration based on data analysis, feature engineering evaluation, and user goals.
    
    Args:
        analysis_report (dict): The initial analysis report
        fe_evaluation (dict): The feature engineering evaluation report
        data_snapshot (pd.DataFrame): A sample of the data after feature engineering
        user_instructions (str): User's goal for the project
        language (str): Language for responses ("en" or "cn")
        execution_plan (dict): The feature engineering execution plan for context
        
    Returns:
        dict: Configuration parameters for H2O AutoML with explanations
    """
    logger.info("Generating H2O AutoML configuration parameters")
    
    # Extract key information from reports
    target_var = analysis_report.get('potential_target', '')
    problem_type = analysis_report.get('problem_type', 'unknown')
    data_issues = analysis_report.get('key_issues', [])
    
    # Format feature engineering plan summary if available
    fe_plan_summary = ""
    if execution_plan and 'steps' in execution_plan:
        fe_steps = execution_plan.get('steps', [])
        fe_plan_summary = "\nFeature Engineering Plan:\n"
        for i, step in enumerate(fe_steps):
            step_name = step.get('name', 'Unknown step')
            step_params = step.get('params', {})
            fe_plan_summary += f"- Step {i+1}: {step_name} with parameters {step_params}\n"
    
    # Format data information for prompt
    data_info = f"""
    Target Variable: {target_var}
    Problem Type: {problem_type}
    Data Issues: {', '.join(data_issues) if isinstance(data_issues, list) else data_issues}
    Feature Engineering Evaluation: {fe_evaluation.get('overall_assessment', 'Not available')}
    """
    
    # Format data sample for the prompt
    try:
        data_sample = data_snapshot.to_string()
    except:
        data_sample = "Data sample not available"
    
    # Create prompt for LLM
    prompt = f"""
    You are an expert data scientist configuring an H2O AutoML run.
    
    USER GOAL: {user_instructions}
    
    DATA INFORMATION:
    {data_info}
    
    DATA SAMPLE (after feature engineering):
    {data_sample}
    {fe_plan_summary}
    
    Based on this information, generate the optimal H2O AutoML configuration as a JSON object with these parameters:
    1. target: The target variable name
    2. problem_type: "classification" or "regression"
    3. include_algos: List of algorithms to include (e.g., ["DRF", "GBM", "XGBoost", "DeepLearning", "GLM"])
    4. exclude_algos: List of algorithms to exclude (use either include_algos OR exclude_algos, not both)
    5. max_models: Integer, maximum number of models to train
    6. max_runtime_secs: Integer, maximum runtime in seconds
    7. sort_metric: Metric to sort models by
    8. balance_classes: Boolean, whether to balance classes for classification
    9. standardize: Boolean, whether to standardize numeric columns
    10. ignore_const_cols: Boolean, whether to ignore constant columns
    
    For preprocessing operations, specify the following boolean parameters:
    - standardize: Whether to standardize numeric features
    - ignore_const_cols: Whether to ignore constant columns
    
    Return ONLY a valid JSON object with these fields.
    
    If the language parameter is 'cn', also include a field called 'summary_chinese' with the summary in Chinese.
    """
    
    # Call LLM for configuration
    try:
        response = call_llm(prompt)
        
        # Validate configuration
        required_keys = ["target", "problem_type", "preprocessing", "max_models", "max_runtime_secs"]
        missing_keys = [key for key in required_keys if key not in response]
        
        if missing_keys:
            logger.warning(f"LLM response missing required keys: {missing_keys}. Adding defaults.")
            
            # Add defaults for missing keys
            defaults = {
                "target": target_var,
                "problem_type": "classification" if problem_type == "classification" else "regression",
                "preprocessing": ["missing_values_handling"],
                "max_models": 20,
                "max_runtime_secs": 600,
            }
            
            for key in missing_keys:
                response[key] = defaults.get(key)
        
        # Convert from raw LLM response format to expected agent_workflow format
        formatted_response = {
            "target_variable": response.get("target", target_var),
            "task": response.get("problem_type", "regression"),
            "features": [],  # This would need to be populated if required
            "h2o_automl_parameters": {
                "max_models": response.get("max_models", 20),
                "max_runtime_secs": response.get("max_runtime_secs", 600),
                "sort_metric": response.get("sort_metric", "AUTO"),
                "balance_classes": response.get("balance_classes", False)
            },
            "summary": response.get("summary", "H2O AutoML configuration generated based on data analysis.")
        }
        
        # Process preprocessing options - H2O's preprocessing param only accepts 'target_encoding'
        preprocessing_options = response.get("preprocessing", ["missing_values_handling"])
        
        # Check if target_encoding is in preprocessing options
        if "target_encoding" in preprocessing_options:
            formatted_response["h2o_automl_parameters"]["preprocessing"] = ["target_encoding"]
        
        # Map other preprocessing options to their respective H2O parameters
        if any(option in ["standardize", "normalize"] for option in preprocessing_options):
            formatted_response["h2o_automl_parameters"]["standardize"] = True
        
        if "ignore_const_cols" in preprocessing_options:
            formatted_response["h2o_automl_parameters"]["ignore_const_cols"] = True
            
        if "turn_on_one_hot_encoding" in preprocessing_options:
            formatted_response["h2o_automl_parameters"]["one_hot_encoding"] = True
        
        # Add language specific summary if needed
        if language == "cn" and "summary_chinese" not in response:
            try:
                chinese_summary_prompt = f"""
                Translate the following H2O AutoML configuration summary to Chinese:
                {response.get('summary', 'H2O AutoML configuration generated based on data analysis.')}
                """
                chinese_response = call_llm(chinese_summary_prompt)
                formatted_response["summary_chinese"] = chinese_response.get("translation", 
                                                                 "基于数据分析生成的H2O AutoML配置")
            except Exception as e:
                logger.warning(f"Failed to generate Chinese summary: {e}")
                formatted_response["summary_chinese"] = "基于数据分析生成的H2O AutoML配置"
        
        # Optional parameters - avoid including both include_algos and exclude_algos
        if "include_algos" in response and response["include_algos"]:
            formatted_response["h2o_automl_parameters"]["include_algos"] = response["include_algos"]
        elif "exclude_algos" in response and response["exclude_algos"]:
            formatted_response["h2o_automl_parameters"]["exclude_algos"] = response["exclude_algos"]
        
        if "summary_chinese" in response:
            formatted_response["summary_chinese"] = response["summary_chinese"]
            
        logger.info(f"Successfully generated H2O config with target variable: {formatted_response['target_variable']}")
        return formatted_response
        
    except Exception as e:
        logger.error(f"Error generating H2O configuration: {e}", exc_info=True)
        # Return fallback configuration
        fallback_config = {
            "target_variable": target_var,
            "task": "regression" if problem_type != "classification" else "classification",
            "features": [],
            "h2o_automl_parameters": {
                "max_models": 20,
                "max_runtime_secs": 600,
                "sort_metric": "AUTO",
                "standardize": True,
                "ignore_const_cols": True,
                "seed": 1234
            },
            "summary": "Fallback configuration due to error in generation process.",
            "summary_chinese": "由于生成过程中的错误，返回默认配置。" if language == "cn" else ""
        }
        return fallback_config

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

def _fallback_config_generation(
    llm_response: str, 
    target_variable: str, 
    id_columns: List[str], 
    data_snapshot: pd.DataFrame = None,
    task_type: str = None,
    language: str = 'en'
) -> Dict[str, Any]:
    """
    当LLM响应解析失败时，生成一个基础的H2O AutoML配置
    
    Args:
        llm_response: LLM的原始响应（用于提取可能的信息）
        target_variable: 目标变量名称
        id_columns: ID列列表
        data_snapshot: 数据快照
        task_type: 任务类型（若为None则自动推断）
        language: 输出语言 - 'en'为英文, 'cn'为中文和英文
        
    Returns:
        Dict: 基础的H2O AutoML配置
    """
    if language == 'cn':
        logger.warning("使用替代方法生成H2O配置...")
    else:
        logger.warning("Using fallback method to generate H2O configuration...")
        
    config = {
        "task": "regression",  # 默认任务类型
        "target_variable": target_variable if target_variable else "Value1",  # 若无目标变量，使用默认列
        "features": [],
        "h2o_automl_parameters": {
            "max_runtime_secs": 120,
            "max_models": 20,
            "seed": 1234,
            "preprocessing": ["missing_values_handling", "standardize", "convert_date"]
        },
        "reasoning": "Fallback configuration with standard parameters"
    }
    
    # 推断任务类型
    if task_type:
        config["task"] = task_type
    elif data_snapshot is not None and target_variable and target_variable in data_snapshot.columns:
        # 根据目标变量的类型推断任务
        if pd.api.types.is_numeric_dtype(data_snapshot[target_variable]):
            config["task"] = "regression"
        else:
            config["task"] = "classification"
    
    # 设置特征列表（排除ID列和目标变量）
    if data_snapshot is not None:
        config["features"] = [
            col for col in data_snapshot.columns 
            if col != config["target_variable"] and col not in id_columns
        ]
    
    # 根据任务类型设置适当的指标
    if config["task"] == "classification":
        config["h2o_automl_parameters"]["stopping_metric"] = "AUC"
        config["h2o_automl_parameters"]["sort_metric"] = "AUC"
    else:
        config["h2o_automl_parameters"]["stopping_metric"] = "RMSE"
        config["h2o_automl_parameters"]["sort_metric"] = "RMSE"
    
    # 添加中文摘要
    if language == 'cn':
        task_zh = "分类" if config["task"] == "classification" else "回归"
        config["summary_chinese"] = f"（替代配置）为{task_zh}任务配置了H2O AutoML。目标变量: {config['target_variable']}。包含{len(config['features'])}个特征。最大运行时间: {config['h2o_automl_parameters']['max_runtime_secs']}秒。"
        logger.info(f"已生成替代H2O配置。任务类型: {config['task']}, 目标: {config['target_variable']}")
    else:
        task_en = "classification" if config["task"] == "classification" else "regression"
        logger.info(f"Generated fallback H2O configuration. Task type: {config['task']}, Target: {config['target_variable']}")
    
    return config 
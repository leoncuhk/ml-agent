import json
import logging
import pandas as pd
import os
import re
from typing import Dict, Any, List, Optional

# 导入LLM接口
from .agent import chat_with_llm

# 配置日志
logger = logging.getLogger("ml_agent.llm_model_evaluator")

def evaluate_final_results(h2o_results: Dict[str, Any], 
                          user_goal: Optional[str] = None,
                          initial_analysis: Optional[Dict[str, Any]] = None,
                          fe_evaluation: Optional[Dict[str, Any]] = None,
                          language: str = 'en') -> Dict[str, Any]:
    """
    评估H2O AutoML训练的最终模型结果

    Args:
        h2o_results: H2O AutoML训练结果
        user_goal: 用户指定的目标
        initial_analysis: 初始数据分析报告
        fe_evaluation: 特征工程评估报告
        language: 输出语言 - 'en'为英文, 'cn'为中文和英文

    Returns:
        Dict: 包含评估结果和中文总结的字典
    """
    if language == 'cn':
        logger.info("开始评估模型训练结果...")
    else:
        logger.info("Starting model training result evaluation...")
    
    # 检查是否存在实际错误 - 修改后的错误检测逻辑
    if "error" in h2o_results and h2o_results.get("error") is not None and h2o_results.get("error") != "None":
        if language == 'cn':
            logger.error(f"模型训练过程中出现错误: {h2o_results.get('error')}")
        else:
            logger.error(f"Error during model training: {h2o_results.get('error')}")
        return _generate_error_evaluation(h2o_results.get("error"), user_goal, language)
    
    # 检查是否有模型训练结果
    if not h2o_results.get("best_model_id") and not h2o_results.get("leaderboard"):
        if language == 'cn':
            error_msg = "未能获取有效的模型训练结果或排行榜"
        else:
            error_msg = "Could not obtain valid model training results or leaderboard"
        logger.error(error_msg)
        return _generate_error_evaluation(error_msg, user_goal, language)
    
    # 提取关键信息
    best_model_id = h2o_results.get("best_model_id", "未知" if language == 'cn' else "Unknown")
    best_model_metrics = h2o_results.get("best_model_metrics", {})
    leaderboard = h2o_results.get("leaderboard")
    feature_importance = h2o_results.get("variable_importance", {})
    model_path = h2o_results.get("best_model_path", "未知" if language == 'cn' else "Unknown")
    
    # 格式化排行榜
    leaderboard_text = "未提供排行榜" if language == 'cn' else "Leaderboard not provided"
    if isinstance(leaderboard, pd.DataFrame):
        leaderboard_text = leaderboard.to_string()
    elif isinstance(leaderboard, dict):
        leaderboard_text = json.dumps(leaderboard, indent=2)
    elif isinstance(leaderboard, str):
        leaderboard_text = leaderboard
    
    # 格式化特征重要性
    importance_text = "未提供特征重要性" if language == 'cn' else "Feature importance not provided"
    if feature_importance:
        if isinstance(feature_importance, dict):
            # 如果是字典，排序后展示
            importance_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            importance_text = "\n".join([f"- {feature}: {importance:.4f}" for feature, importance in importance_items[:10]])
            if len(importance_items) > 10:
                importance_text += f"\n(仅显示前10个特征，共{len(importance_items)}个)" if language == 'cn' else f"\n(Showing only top 10 features out of {len(importance_items)})"
        elif isinstance(feature_importance, str):
            importance_text = feature_importance
    
    # 格式化模型指标
    metrics_text = "未提供模型指标" if language == 'cn' else "Model metrics not provided"
    if best_model_metrics:
        if isinstance(best_model_metrics, dict):
            metrics_text = "\n".join([f"- {k}: {v}" for k, v in best_model_metrics.items()])
        elif isinstance(best_model_metrics, str):
            metrics_text = best_model_metrics
    
    # 提取目标变量和推断任务类型
    target_variable = None
    task_type = None
    if initial_analysis:
        target_variable = initial_analysis.get('overall_summary', {}).get('potential_target_variable')
        # 尝试从分析报告推断任务类型
        for col_info in initial_analysis.get('column_analysis', []):
            if col_info.get('column_name') == target_variable:
                data_type = col_info.get('data_type', '')
                if data_type in ['object', 'category']:
                    task_type = "分类" if language == 'cn' else "Classification"
                    unique_values = col_info.get('unique_values')
                    if unique_values and len(unique_values) == 2:
                        task_type = "二分类" if language == 'cn' else "Binary Classification"
                    elif unique_values and len(unique_values) > 2:
                        task_type = "多分类" if language == 'cn' else "Multi-class Classification"
                else:
                    task_type = "回归" if language == 'cn' else "Regression"
    
    # 构建提示
    if language == 'cn':
        prompt = f"""
你是一位机器学习模型评估专家，需要评估H2O AutoML训练的模型结果。请根据以下信息评估模型性能:

### 用户目标:
{user_goal if user_goal else "未提供具体用户目标"}

### 目标变量:
{target_variable if target_variable else "未提供目标变量信息"}

### 任务类型:
{task_type if task_type else "未明确指定任务类型"}

### 最佳模型:
模型ID: {best_model_id}
模型保存路径: {model_path}

### 模型性能指标:
{metrics_text}

### 模型排行榜:
{leaderboard_text[:1000] if len(leaderboard_text) > 1000 else leaderboard_text}

### 特征重要性:
{importance_text}

### 特征工程评估:
{fe_evaluation.get('overall_summary', '未提供特征工程评估') if fe_evaluation else "未提供特征工程评估"}

请提供全面评估，包括以下几点:
1. 模型性能评估: 分析模型的整体性能、主要指标和潜在问题
2. 特征重要性分析: 分析哪些特征对模型贡献最大，有何意义
3. 用户目标匹配度: 模型是否满足用户的目标需求
4. 工作流程评估: 整个流程（数据准备->特征工程->模型训练）效果如何
5. 改进建议: 如何可能进一步提高模型性能
6. 下一步建议: 建议用户采取的后续步骤

请以JSON格式返回评估结果，包含以下键:
- "model_performance_summary": 模型性能和特征重要性分析
- "user_goal_alignment": 与用户目标的匹配程度
- "workflow_effectiveness": 整个工作流程效果评估
- "next_steps_suggestions": 改进和下一步建议列表
- "overall_summary": 英文整体评价
- "overall_summary_chinese": 中文整体评价

请确保JSON格式正确，并提供详细、有见地的评估。
"""
    else:
        prompt = f"""
You are a machine learning model evaluation expert tasked with evaluating the results of H2O AutoML training. Please evaluate the model performance based on the following information:

### User Goal:
{user_goal if user_goal else "No specific user goal provided"}

### Target Variable:
{target_variable if target_variable else "Target variable information not provided"}

### Task Type:
{task_type if task_type else "Task type not specified"}

### Best Model:
Model ID: {best_model_id}
Model Save Path: {model_path}

### Model Performance Metrics:
{metrics_text}

### Model Leaderboard:
{leaderboard_text[:1000] if len(leaderboard_text) > 1000 else leaderboard_text}

### Feature Importance:
{importance_text}

### Feature Engineering Evaluation:
{fe_evaluation.get('overall_summary', 'Feature engineering evaluation not provided') if fe_evaluation else "Feature engineering evaluation not provided"}

Please provide a comprehensive evaluation, including the following points:
1. Model Performance Assessment: Analyze overall model performance, key metrics, and potential issues
2. Feature Importance Analysis: Analyze which features contribute most to the model and their significance
3. User Goal Alignment: Whether the model meets the user's goal requirements
4. Workflow Assessment: How the entire process (data preparation -> feature engineering -> model training) performed
5. Improvement Suggestions: How to potentially further improve model performance
6. Next Steps: Recommended next steps for the user

Please return the evaluation results in JSON format, containing the following keys:
- "model_performance_summary": Model performance and feature importance analysis
- "user_goal_alignment": Alignment with user goals
- "workflow_effectiveness": Overall workflow effectiveness assessment
- "next_steps_suggestions": Improvement and next step suggestions list
- "overall_summary": Overall evaluation in English

Please ensure the JSON format is correct and provide detailed, insightful evaluation.
"""
    
    try:
        # 调用LLM进行评估
        response = chat_with_llm(prompt)
        
        # 尝试从响应中提取JSON
        extracted_json = _extract_json_from_text(response)
        if not extracted_json:
            if language == 'cn':
                logger.warning(f"LLM响应未包含有效JSON，将使用替代评估方法。原始响应: {response[:100]}...")
            else:
                logger.warning(f"LLM response doesn't contain valid JSON, using fallback evaluation. Original response: {response[:100]}...")
            return _generate_fallback_evaluation(h2o_results, user_goal, language)
        
        # 验证必要的键
        required_keys = ["model_performance_summary", "user_goal_alignment", "workflow_effectiveness", "next_steps_suggestions", "overall_summary"]
        missing_keys = [key for key in required_keys if key not in extracted_json]
        
        if missing_keys:
            if language == 'cn':
                logger.warning(f"LLM返回的评估缺少以下必要键: {missing_keys}。将使用替代评估方法。")
            else:
                logger.warning(f"LLM evaluation response missing necessary keys: {missing_keys}. Using fallback evaluation.")
            return _generate_fallback_evaluation(h2o_results, user_goal, language)
        
        # 确保有中文总结 (如果需要的话)
        if language == 'cn' and "overall_summary_chinese" not in extracted_json:
            extracted_json["overall_summary_chinese"] = _generate_chinese_summary(
                extracted_json.get("overall_summary", "模型训练评估完成。" if language == 'cn' else "Model training evaluation completed."), 
                extracted_json.get("model_performance_summary", "")
            )
        
        # 添加MLflow运行ID（如果存在）
        if "mlflow_run_id" in h2o_results:
            extracted_json["mlflow_run_id"] = h2o_results["mlflow_run_id"]
        
        if language == 'cn':
            logger.info("模型评估完成。")
            if "overall_summary_chinese" in extracted_json:
                logger.info(f"中文评估总结: {extracted_json.get('overall_summary_chinese')}")
        else:
            logger.info("Model evaluation completed.")
            
        return extracted_json
        
    except Exception as e:
        if language == 'cn':
            logger.error(f"评估模型结果时出错: {str(e)}")
        else:
            logger.error(f"Error evaluating model results: {str(e)}")
        return _generate_fallback_evaluation(h2o_results, user_goal, language)

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

def _generate_fallback_evaluation(h2o_results: Dict[str, Any], user_goal: Optional[str] = None, language: str = 'en') -> Dict[str, Any]:
    """
    生成备用评估，当LLM响应无法解析时使用
    
    Args:
        h2o_results: H2O AutoML训练结果
        user_goal: 用户指定的目标
        language: 输出语言 - 'en'为英文, 'cn'为中文和英文
        
    Returns:
        Dict: 包含评估结果的字典
    """
    if language == 'cn':
        logger.info("生成备用模型评估...")
    else:
        logger.info("Generating fallback model evaluation...")
    
    # 提取关键信息
    best_model_id = h2o_results.get("best_model_id", "未知" if language == 'cn' else "Unknown")
    best_model_metrics = h2o_results.get("best_model_metrics", {})
    
    # 生成基础评估信息
    if language == 'cn':
        performance_summary = f"最佳模型ID: {best_model_id}."
        if best_model_metrics:
            performance_summary += f" 主要指标: {best_model_metrics}"
    else:
        performance_summary = f"Best model ID: {best_model_id}."
        if best_model_metrics:
            performance_summary += f" Main metrics: {best_model_metrics}"
    
    # 生成默认评估报告
    if language == 'cn':
        evaluation = {
            "model_performance_summary": performance_summary,
            "user_goal_alignment": f"模型训练完成，可能满足用户目标: '{user_goal if user_goal else '未指定'}'",
            "workflow_effectiveness": "工作流程成功执行了数据准备、特征工程和模型训练阶段。",
            "next_steps_suggestions": [
                "检查模型的详细性能指标",
                "考虑使用更长的训练时间来获得更好的模型",
                "分析特征重要性以优化特征工程",
                "尝试部署模型进行推理"
            ],
            "overall_summary": f"H2O AutoML training completed with best model: {best_model_id}. The model is ready for evaluation and potential deployment.",
            "overall_summary_chinese": f"H2O AutoML 训练完成，最佳模型为: {best_model_id}。该模型已准备好进行评估和潜在部署。工作流程顺利完成了全部阶段，包括数据准备、特征工程和模型训练。"
        }
    else:
        evaluation = {
            "model_performance_summary": performance_summary,
            "user_goal_alignment": f"Model training completed, potentially meeting user goal: '{user_goal if user_goal else 'Not specified'}'",
            "workflow_effectiveness": "The workflow successfully executed the data preparation, feature engineering, and model training stages.",
            "next_steps_suggestions": [
                "Check detailed model performance metrics",
                "Consider using longer training time for better models",
                "Analyze feature importance to optimize feature engineering",
                "Try deploying the model for inference"
            ],
            "overall_summary": f"H2O AutoML training completed with best model: {best_model_id}. The model is ready for evaluation and potential deployment."
        }
        
        # Add Chinese summary only if needed
        if language == 'cn':
            evaluation["overall_summary_chinese"] = f"H2O AutoML 训练完成，最佳模型为: {best_model_id}。该模型已准备好进行评估和潜在部署。工作流程顺利完成了全部阶段，包括数据准备、特征工程和模型训练。"
    
    # 添加MLflow运行ID（如果存在）
    if "mlflow_run_id" in h2o_results:
        evaluation["mlflow_run_id"] = h2o_results["mlflow_run_id"]
    
    if language == 'cn':
        logger.info(f"备用评估生成完成。中文总结: {evaluation.get('overall_summary_chinese', '')}")
    else:
        logger.info(f"Fallback evaluation generated. Summary: {evaluation.get('overall_summary', '')}")
    
    return evaluation

def _generate_error_evaluation(error_message: str, user_goal: Optional[str] = None, language: str = 'en') -> Dict[str, Any]:
    """
    在模型训练出错时生成评估报告
    
    Args:
        error_message: 错误信息
        user_goal: 用户指定的目标
        language: 输出语言 - 'en'为英文, 'cn'为中文和英文
        
    Returns:
        Dict: 包含错误评估的字典
    """
    if language == 'cn':
        logger.info("生成错误情况下的模型评估...")
    else:
        logger.info("Generating error evaluation...")
    
    if language == 'cn':
        evaluation = {
            "model_performance_summary": f"模型训练过程中出现错误: {error_message}",
            "user_goal_alignment": "由于训练错误，无法评估与用户目标的匹配度",
            "workflow_effectiveness": "工作流程在模型训练阶段遇到问题，需要排查原因",
            "next_steps_suggestions": [
                "检查训练日志以确定错误原因",
                "验证数据格式和特征是否符合要求",
                "考虑修改模型参数或调整特征工程步骤",
                "排除错误后重新尝试训练"
            ],
            "overall_summary": f"Model training encountered an error: {error_message}. Troubleshooting is required before proceeding.",
            "overall_summary_chinese": f"模型训练遇到错误: {error_message}。在继续之前需要进行故障排除。建议检查训练日志并验证数据格式和特征是否符合要求。"
        }
    else:
        evaluation = {
            "model_performance_summary": f"Error occurred during model training: {error_message}",
            "user_goal_alignment": "Unable to assess alignment with user goals due to training error",
            "workflow_effectiveness": "Workflow encountered issues during the model training stage, requiring troubleshooting",
            "next_steps_suggestions": [
                "Check training logs to determine error cause",
                "Verify data format and features meet requirements",
                "Consider modifying model parameters or adjusting feature engineering steps",
                "Retry training after resolving error"
            ],
            "overall_summary": f"Model training encountered an error: {error_message}. Troubleshooting is required before proceeding."
        }
        
        # Add Chinese summary only if needed
        if language == 'cn':
            evaluation["overall_summary_chinese"] = f"模型训练遇到错误: {error_message}。在继续之前需要进行故障排除。建议检查训练日志并验证数据格式和特征是否符合要求。"
    
    if language == 'cn':
        logger.info(f"错误评估生成完成。中文总结: {evaluation.get('overall_summary_chinese', '')}")
    else:
        logger.info(f"Error evaluation generated. Summary: {evaluation.get('overall_summary', '')}")
        
    return evaluation

def _generate_chinese_summary(english_summary: str, performance_summary: str) -> str:
    """
    为评估生成中文总结
    这是一个简单的实现，实际应用中可能需要更复杂的翻译逻辑
    """
    # 尝试匹配常见术语的简单翻译
    translations = {
        "model": "模型",
        "training": "训练",
        "completed": "完成",
        "best model": "最佳模型",
        "performance": "性能",
        "metrics": "指标",
        "accuracy": "准确率",
        "precision": "精确率",
        "recall": "召回率",
        "AUC": "AUC",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "ROC": "ROC",
        "feature importance": "特征重要性",
        "workflow": "工作流程",
        "deployment": "部署",
        "evaluation": "评估",
        "regression": "回归",
        "classification": "分类"
    }
    
    # 构建一个简单的中文摘要
    summary = english_summary
    for eng, chn in translations.items():
        summary = re.sub(r'\b' + re.escape(eng) + r'\b', chn, summary, flags=re.IGNORECASE)
    
    # 添加性能摘要
    performance = performance_summary
    for eng, chn in translations.items():
        performance = re.sub(r'\b' + re.escape(eng) + r'\b', chn, performance, flags=re.IGNORECASE)
    
    # 如果摘要为空，生成默认摘要
    if not summary:
        summary = "模型训练已完成，准备好进行评估和部署。"
    
    # 确保摘要以句号结尾
    if not summary.endswith('。'):
        summary += '。'
    
    # 如果性能摘要有实质内容，添加到总结中
    if performance and "未提供" not in performance:
        if not performance.endswith('。'):
            performance += '。'
        summary += f" {performance}"
    
    # 添加标准结束语
    summary += " 工作流程已顺利完成全部阶段。"
    
    return summary 
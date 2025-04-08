import json
import logging
import pandas as pd
import io
import re
from typing import Dict, Any, List, Optional

# 从 llm_interface.py 导入 LLM 接口函数
from .agent import chat_with_llm

# 配置日志
logger = logging.getLogger("ml_agent.llm_feature_evaluator")

def evaluate_fe_data(df_processed: pd.DataFrame, 
                    fe_plan: List[Dict[str, Any]], 
                    analysis_report: Dict[str, Any] = None,
                    language: str = 'en') -> Dict[str, Any]:
    """
    评估特征工程后的数据质量和建模就绪程度

    Args:
        df_processed: 特征工程后的DataFrame
        fe_plan: 执行的特征工程计划步骤
        analysis_report: 原始数据分析报告（可选，用于比较）
        language: 输出语言 - 'en'为英文, 'cn'为中文和英文

    Returns:
        Dict: 包含评估结果和中文总结的字典
    """
    logger.info("开始评估特征工程结果...")
    
    # Handle analysis_report properly
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
    
    # 提取目标变量（如果存在）
    target_variable = None
    if analysis_report and isinstance(analysis_report, dict):
        # Try direct access first
        target_variable = analysis_report.get('potential_target_variable')
        # Try nested in overall_summary if it's a dict
        if target_variable is None and isinstance(analysis_report.get('overall_summary'), dict):
            target_variable = analysis_report.get('overall_summary', {}).get('potential_target_variable')
        
        if language == 'cn':
            logger.info(f"从分析报告中提取的目标变量: {target_variable}")
        else:
            logger.info(f"Extracted target variable from analysis: {target_variable}")
    
    # 生成数据摘要
    df_info = _get_fe_data_summary(df_processed, target_variable)
    
    # 提取计划步骤摘要
    plan_summary = _get_fe_plan_summary(fe_plan)
    
    # 构建提示
    if language == 'cn':
        prompt = f"""
你是一位机器学习特征工程专家，需要评估特征工程后的数据集质量。请根据以下信息评估特征工程结果:

### 特征工程计划摘要:
{plan_summary}

### 特征工程后的数据摘要:
{df_info}

请提供全面评估，包括以下几点:
1. 特征质量评估: 分析各特征的质量、分布、缺失值等情况
2. 特征间相关性: 检查特征间是否存在高相关性、多重共线性等问题
3. 建模就绪评估: 数据是否已准备好进行机器学习建模，是否存在障碍
4. 改进建议: 如有需要，建议额外的特征工程步骤
5. 整体评价: 特征工程过程的整体评价

请以JSON格式返回评估结果，包含以下键:
- "feature_assessment": 特征质量和相关性分析
- "ml_readiness_assessment": 建模就绪评估
- "suggestions": 改进建议列表
- "overall_summary": 英文整体评价
- "overall_summary_chinese": 中文整体评价

请确保JSON格式正确，并提供详细、有见地的评估。
"""
    else:
        prompt = f"""
You are a machine learning feature engineering expert tasked with evaluating the quality of a dataset after feature engineering. Please evaluate the feature engineering results based on the following information:

### Feature Engineering Plan Summary:
{plan_summary}

### Post-Feature Engineering Data Summary:
{df_info}

Please provide a comprehensive evaluation, including the following points:
1. Feature Quality Assessment: Analyze the quality, distribution, missing values, etc. of each feature
2. Inter-Feature Correlation: Check if there are high correlations, multicollinearity, etc. between features
3. Modeling Readiness Assessment: Is the data ready for machine learning modeling, are there any obstacles
4. Improvement Suggestions: If needed, suggest additional feature engineering steps
5. Overall Evaluation: An overall evaluation of the feature engineering process

Please return the evaluation results in JSON format, containing the following keys:
- "feature_assessment": Feature quality and correlation analysis
- "ml_readiness_assessment": Modeling readiness assessment
- "suggestions": List of improvement suggestions
- "overall_summary": Overall evaluation in English

Please ensure the JSON format is correct and provide detailed, insightful assessment.
"""
    
    try:
        # 调用LLM进行评估
        response = chat_with_llm(prompt)
        
        # 尝试从响应中提取JSON
        extracted_json = _extract_json_from_text(response)
        if not extracted_json:
            logger.warning(f"LLM响应未包含有效JSON，将使用替代评估方法。原始响应: {response[:100]}...")
            return _generate_fallback_evaluation(df_processed, fe_plan, response, language)
        
        # 验证必要的键
        required_keys = ["feature_assessment", "ml_readiness_assessment", "suggestions", "overall_summary"]
        missing_keys = [key for key in required_keys if key not in extracted_json]
        
        if missing_keys:
            logger.warning(f"LLM返回的评估缺少以下必要键: {missing_keys}。将使用替代评估方法。")
            return _generate_fallback_evaluation(df_processed, fe_plan, response, language)
        
        # 确保有中文总结
        if "overall_summary_chinese" not in extracted_json:
            extracted_json["overall_summary_chinese"] = _generate_chinese_summary(
                extracted_json.get("overall_summary", "特征工程评估完成。"), 
                extracted_json.get("ml_readiness_assessment", "")
            )
        
        logger.info("特征工程评估完成。")
        logger.info(f"中文评估总结: {extracted_json.get('overall_summary_chinese')}")
        return extracted_json
        
    except Exception as e:
        logger.error(f"评估特征工程数据时出错: {str(e)}")
        return _generate_fallback_evaluation(df_processed, fe_plan, f"错误: {str(e)}", language)

def _get_fe_data_summary(df: pd.DataFrame, target_variable: Optional[str] = None) -> str:
    """生成特征工程后数据的摘要描述"""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    summary = f"### 数据摘要\n"
    summary += f"- 形状: {df.shape[0]} 行, {df.shape[1]} 列\n"
    summary += f"- 缺失值总数: {df.isna().sum().sum()} 个\n\n"
    
    # 列类型分布
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    other_cols = [col for col in df.columns if col not in numeric_cols and col not in cat_cols]
    
    summary += f"### 列类型分布\n"
    summary += f"- 数值列: {len(numeric_cols)} 列\n"
    summary += f"- 分类列: {len(cat_cols)} 列\n"
    if other_cols:
        summary += f"- 其他类型列: {len(other_cols)} 列\n"
    
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
        stats = df[numeric_cols].describe().T
        for col in numeric_cols[:min(5, len(numeric_cols))]:  # 限制显示前5列
            summary += f"- {col}: 均值={stats.loc[col, 'mean']:.2f}, 标准差={stats.loc[col, 'std']:.2f}, 最小值={stats.loc[col, 'min']:.2f}, 最大值={stats.loc[col, 'max']:.2f}\n"
        if len(numeric_cols) > 5:
            summary += f"  (仅显示前5列，共{len(numeric_cols)}列)\n"
    
    # 如果存在目标变量，进行特殊分析
    if target_variable and target_variable in df.columns:
        summary += f"\n### 目标变量 '{target_variable}' 摘要\n"
        if df[target_variable].dtype.name in ['object', 'category']:
            value_counts = df[target_variable].value_counts()
            summary += f"- 类型: 分类变量\n"
            summary += f"- 唯一值数量: {df[target_variable].nunique()}\n"
            summary += f"- 类别分布:\n"
            for val, count in value_counts.items():
                pct = count / len(df) * 100
                summary += f"  - {val}: {count} ({pct:.1f}%)\n"
        else:
            summary += f"- 类型: 数值变量\n"
            summary += f"- 统计摘要: {df[target_variable].describe().to_dict()}\n"
    
    # 高基数分类特征
    high_cardinality_cols = [col for col in cat_cols if df[col].nunique() > 10]
    if high_cardinality_cols:
        summary += f"\n### 高基数分类特征\n"
        for col in high_cardinality_cols:
            summary += f"- {col}: {df[col].nunique()} 个不同值\n"
    
    # 样本展示
    summary += f"\n### 数据样本 (前3行)\n{df.head(3)}\n"
    
    return summary

def _get_fe_plan_summary(fe_plan: List[Dict[str, Any]]) -> str:
    """生成特征工程计划的摘要描述"""
    if not fe_plan:
        return "没有提供特征工程计划。"
    
    summary = f"特征工程共执行了 {len(fe_plan)} 个步骤:\n\n"
    
    for i, step in enumerate(fe_plan):
        step_num = step.get('step', i+1)
        module = step.get('module', 'unknown')
        func = step.get('function', 'unknown')
        args = step.get('args', {})
        reason = step.get('reasoning', 'No reasoning provided')
        
        summary += f"步骤 {step_num}: {module}.{func}\n"
        if args:
            summary += f"- 参数: {args}\n"
        summary += f"- 原因: {reason}\n\n"
    
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

def _generate_fallback_evaluation(df: pd.DataFrame, fe_plan: List[Dict[str, Any]], llm_response: str, language: str = 'en') -> Dict[str, Any]:
    """生成备用评估，当LLM响应无法解析时使用"""
    logger.info("生成备用特征工程评估...")
    
    # 生成基础评估信息
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    num_features = df.shape[1]
    num_numeric = len(df.select_dtypes(include=['number']).columns)
    num_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
    
    if language == 'cn':
        readiness = "数据似乎已准备好进行建模" if missing_pct < 5 else "数据中存在较多缺失值，可能需要进一步处理"
        
        evaluation = {
            "feature_assessment": f"数据包含 {num_features} 个特征，其中 {num_numeric} 个数值特征和 {num_categorical} 个分类特征。缺失值比例为 {missing_pct:.2f}%。",
            "ml_readiness_assessment": readiness,
            "suggestions": ["考虑进一步处理缺失值" if missing_pct > 0 else "特征工程已完成，可以进行建模"],
            "overall_summary": f"Feature engineering resulted in a dataset with {num_features} features ({num_numeric} numeric, {num_categorical} categorical). Missing value percentage: {missing_pct:.2f}%.",
            "overall_summary_chinese": f"特征工程后的数据集包含 {num_features} 个特征（{num_numeric} 个数值特征，{num_categorical} 个分类特征）。缺失值百分比：{missing_pct:.2f}%。{readiness}。",
            "llm_raw_response": llm_response[:500] + ("..." if len(llm_response) > 500 else "")
        }
    else:
        readiness = "Data appears ready for modeling" if missing_pct < 5 else "Data contains a significant amount of missing values and may require further processing"
        
        evaluation = {
            "feature_assessment": f"The data contains {num_features} features, with {num_numeric} numeric features and {num_categorical} categorical features. The percentage of missing values is {missing_pct:.2f}%.",
            "ml_readiness_assessment": readiness,
            "suggestions": ["Consider further processing of missing values" if missing_pct > 0 else "Feature engineering is complete, ready for modeling"],
            "overall_summary": f"Feature engineering resulted in a dataset with {num_features} features ({num_numeric} numeric, {num_categorical} categorical). Missing value percentage: {missing_pct:.2f}%. {readiness}.",
            "llm_raw_response": llm_response[:500] + ("..." if len(llm_response) > 500 else "")
        }
        
        # Add Chinese summary if language is cn
        if language == 'cn':
            evaluation["overall_summary_chinese"] = f"特征工程后的数据集包含 {num_features} 个特征（{num_numeric} 个数值特征，{num_categorical} 个分类特征）。缺失值百分比：{missing_pct:.2f}%。{readiness}。"
    
    logger.info(f"备用评估生成完成。总结: {evaluation['overall_summary']}")
    return evaluation

def _generate_chinese_summary(english_summary: str, readiness_assessment: str) -> str:
    """
    为评估生成中文总结
    这是一个简单的实现，实际应用中可能需要更复杂的翻译逻辑
    """
    # 尝试匹配常见术语的简单翻译
    translations = {
        "feature engineering": "特征工程",
        "dataset": "数据集",
        "features": "特征",
        "numeric": "数值",
        "categorical": "分类",
        "missing values": "缺失值",
        "correlation": "相关性",
        "ready for modeling": "已准备好进行建模",
        "needs further processing": "需要进一步处理",
        "high cardinality": "高基数",
        "encoding": "编码",
        "scaling": "缩放",
        "outliers": "异常值"
    }
    
    # 构建一个简单的中文摘要
    summary = english_summary
    for eng, chn in translations.items():
        summary = re.sub(r'\b' + re.escape(eng) + r'\b', chn, summary, flags=re.IGNORECASE)
    
    # 添加就绪性评估
    readiness = readiness_assessment
    for eng, chn in translations.items():
        readiness = re.sub(r'\b' + re.escape(eng) + r'\b', chn, readiness, flags=re.IGNORECASE)
    
    if not summary.endswith('。'):
        summary += '。'
    
    summary += f" {readiness}"
    
    return summary 
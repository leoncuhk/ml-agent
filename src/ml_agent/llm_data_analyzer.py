# src/ml_agent/llm_analyzer.py

import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import logging
import re
from typing import Optional, Dict, Any

# Import chat_with_llm from agent module
from .agent import chat_with_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the create_stats_summary function
def create_stats_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Creates a summary of basic statistics for the DataFrame.
    
    Args:
        df: The DataFrame to analyze
        
    Returns:
        Dict: Dictionary with statistical summaries
    """
    stats = {}
    try:
        # Get basic stats for numeric columns
        numeric_stats = df.describe().to_dict()
        # Convert numpy types to Python types for JSON serialization
        for col, col_stats in numeric_stats.items():
            stats[col] = {k: float(v) if not pd.isna(v) else None for k, v in col_stats.items()}
    except Exception as e:
        logger.warning(f"Could not generate complete statistics summary: {e}")
    
    return stats

def analyze_data_with_llm(df: pd.DataFrame, user_goal: Optional[str] = None, language: str = 'en') -> Dict[str, Any]:
    """
    使用LLM分析数据集并生成报告

    Args:
        df: 需要分析的数据框
        user_goal: 用户指定的分析目标
        language: 输出语言 - 'en'为英文, 'cn'为中文和英文

    Returns:
        Dict: 包含分析结果的字典
    """
    if language == 'cn':
        logger.info(f"开始使用LLM分析数据，包含 {df.shape[0]} 行和 {df.shape[1]} 列...")
    else:
        logger.info(f"Starting LLM data analysis with {df.shape[0]} rows and {df.shape[1]} columns...")
    
    # 获取数据集的统计信息
    try:
        df_stats = create_stats_summary(df)
    except Exception as e:
        if language == 'cn':
            logger.error(f"生成数据统计摘要时出错: {str(e)}")
        else:
            logger.error(f"Error generating data statistics summary: {str(e)}")
        df_stats = {}
    
    # 创建每列详细分析
    column_analyses = []
    
    for column in df.columns:
        col_type = str(df[column].dtype)
        missing_count = df[column].isna().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        col_info = {
            "column_name": column,
            "data_type": col_type,
            "missing_values": {
                "count": int(missing_count),
                "percentage": round(missing_pct, 2)
            }
        }
        
        # 针对不同类型的列添加相关统计信息
        if pd.api.types.is_numeric_dtype(df[column]):
            try:
                col_info.update({
                    "statistics": {
                        "min": float(df[column].min()) if not pd.isna(df[column].min()) else None,
                        "max": float(df[column].max()) if not pd.isna(df[column].max()) else None,
                        "mean": float(df[column].mean()) if not pd.isna(df[column].mean()) else None,
                        "median": float(df[column].median()) if not pd.isna(df[column].median()) else None,
                        "std": float(df[column].std()) if not pd.isna(df[column].std()) else None
                    },
                    "distribution": "numeric"
                })
            except Exception as e:
                if language == 'cn':
                    logger.warning(f"为列 {column} 生成数值统计信息时出错: {str(e)}")
                else:
                    logger.warning(f"Error generating numeric statistics for column {column}: {str(e)}")
        
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            try:
                value_counts = df[column].value_counts(dropna=False).head(10).to_dict()
                # 将键转换为字符串以确保JSON可序列化
                value_counts = {str(k): int(v) for k, v in value_counts.items()}
                
                col_info.update({
                    "unique_values": len(df[column].unique()),
                    "most_common_values": value_counts,
                    "distribution": "categorical"
                })
                
                # 检查是否有日期模式
                date_sample = df[column].dropna().iloc[0] if not df[column].dropna().empty else ""
                if isinstance(date_sample, str) and (
                    re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', date_sample) or 
                    re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', date_sample)
                ):
                    col_info["likely_date_format"] = True
                    col_info["distribution"] = "datetime"
                
                # 检查是否有数值（带有格式）
                numeric_sample = df[column].dropna().iloc[0] if not df[column].dropna().empty else ""
                if isinstance(numeric_sample, str) and re.search(r'^[$£€]?\s*\d+[.,]?\d*\s*[%]?$', numeric_sample):
                    col_info["likely_numeric_format"] = True
                    col_info["distribution"] = "numeric_string"
            except Exception as e:
                if language == 'cn':
                    logger.warning(f"为列 {column} 生成分类统计信息时出错: {str(e)}")
                else:
                    logger.warning(f"Error generating categorical statistics for column {column}: {str(e)}")
        
        elif pd.api.types.is_datetime64_dtype(df[column]):
            try:
                col_info.update({
                    "min_date": str(df[column].min()) if not pd.isna(df[column].min()) else None,
                    "max_date": str(df[column].max()) if not pd.isna(df[column].max()) else None,
                    "distribution": "datetime"
                })
            except Exception as e:
                if language == 'cn':
                    logger.warning(f"为列 {column} 生成日期统计信息时出错: {str(e)}")
                else:
                    logger.warning(f"Error generating datetime statistics for column {column}: {str(e)}")
        
        column_analyses.append(col_info)
    
    # 检测潜在目标变量
    potential_target = None
    if df.shape[1] > 1:
        try:
            # 简单规则：考虑有缺失值较少的数值列或分类列作为潜在目标
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                missing_counts = df[numeric_cols].isna().sum()
                # 选择缺失值最少的数值列
                potential_target = missing_counts.idxmin()
        except Exception as e:
            if language == 'cn':
                logger.warning(f"尝试确定潜在目标变量时出错: {str(e)}")
            else:
                logger.warning(f"Error trying to determine potential target variable: {str(e)}")
    
    # 检测潜在ID列
    potential_id_columns = []
    for column in df.columns:
        try:
            # 检查是否为唯一值且为整数或字符串
            if df[column].nunique() == df.shape[0]:
                if 'id' in column.lower() or pd.api.types.is_integer_dtype(df[column]) or (
                    pd.api.types.is_object_dtype(df[column]) and 
                    all(isinstance(x, str) and re.search(r'\d+', str(x)) for x in df[column].dropna().head(5))
                ):
                    potential_id_columns.append(column)
        except Exception as e:
            if language == 'cn':
                logger.warning(f"检查列 {column} 是否为ID列时出错: {str(e)}")
            else:
                logger.warning(f"Error checking if column {column} is an ID column: {str(e)}")
    
    # 准备向LLM发送的数据摘要
    data_summary = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": df.columns.tolist(),
        "column_analysis": column_analyses,
        "potential_target_variable": potential_target,
        "potential_id_columns": potential_id_columns,
        "statistical_summary": df_stats
    }
    
    # 为了避免超出token限制，限制发送给LLM的详细数据量
    if len(column_analyses) > 15:
        if language == 'cn':
            logger.info(f"列数较多 ({len(column_analyses)}), 限制发送给LLM的详细信息...")
        else:
            logger.info(f"Large number of columns ({len(column_analyses)}), limiting detailed information sent to LLM...")
        data_summary["column_analysis"] = column_analyses[:15]
        data_summary["column_analysis_truncated"] = True
    
    if language == 'cn':
        # 中文提示
        prompt = f"""
你是一位数据科学专家，需要对一个提供给你的数据集进行全面分析。

### 数据集基本信息
- 行数: {df.shape[0]}
- 列数: {df.shape[1]}
- 列名: {', '.join(df.columns.tolist())}

### 用户目标
{user_goal if user_goal else "用户未提供具体目标"}

### 详细列分析
{json.dumps(data_summary["column_analysis"], indent=2, ensure_ascii=False)}

### 潜在目标变量
{potential_target}

### 潜在ID列
{', '.join(potential_id_columns) if potential_id_columns else "未检测到明显的ID列"}

根据以上信息，请提供以下内容:

1. **数据概览**: 数据集的整体特点，包括大小、领域和重要的特征。
2. **数据质量分析**: 
   - 缺失值分析：哪些列存在缺失值以及可能的处理方法
   - 异常值检测：可能存在异常值的列和处理建议
   - 数据类型问题：需要类型转换的列
   - 格式问题：存在格式不一致的列
3. **关键发现**: 数据中的重要模式、相关性或有趣的特征
4. **准备建议**: 
   - 数据清洗：建议的清洗步骤
   - 特征工程：可能的特征创建或转换
   - 目标变量确认：对目标变量的建议和理由
5. **总结**: 对数据集整体质量的评估和建议的后续步骤

请以JSON格式返回分析结果，包含以下键:
- "data_overview": 数据概览
- "quality_issues": 数据质量问题的列表，每个问题包含问题类型、影响列和建议
- "key_insights": 关键发现列表
- "preparation_recommendations": 数据准备建议列表
- "overall_summary": 总体评估的英文总结
- "overall_summary_chinese": 总体评估的中文总结

请确保JSON格式正确，并提供详细、有见地的分析。
"""
    else:
        # English prompt
        prompt = f"""
You are a data science expert tasked with performing a comprehensive analysis of a dataset provided to you.

### Dataset Basic Information
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Column names: {', '.join(df.columns.tolist())}

### User Goal
{user_goal if user_goal else "No specific goal provided by user"}

### Detailed Column Analysis
{json.dumps(data_summary["column_analysis"], indent=2)}

### Potential Target Variable
{potential_target}

### Potential ID Columns
{', '.join(potential_id_columns) if potential_id_columns else "No obvious ID columns detected"}

Based on the information above, please provide the following:

1. **Data Overview**: Overall characteristics of the dataset, including size, domain, and important features.
2. **Data Quality Analysis**: 
   - Missing values analysis: which columns have missing values and potential treatment methods
   - Outlier detection: columns that may have outliers and treatment suggestions
   - Data type issues: columns that need type conversion
   - Format issues: columns with inconsistent formatting
3. **Key Insights**: Important patterns, correlations, or interesting features in the data
4. **Preparation Recommendations**: 
   - Data cleaning: suggested cleaning steps
   - Feature engineering: potential feature creation or transformations
   - Target variable confirmation: suggestions for the target variable and rationale
5. **Summary**: Overall assessment of dataset quality and suggested next steps

Please return the analysis results in JSON format, containing the following keys:
- "data_overview": Data overview
- "quality_issues": List of data quality issues, each containing issue type, affected columns, and recommendations
- "key_insights": List of key insights
- "preparation_recommendations": List of data preparation recommendations
- "overall_summary": Overall assessment summary in English

Please ensure the JSON format is correct and provide detailed, insightful analysis.
"""

    try:
        # 调用LLM进行分析 (使用已有功能)
        response = chat_with_llm(prompt)
        
        # 尝试从响应中提取JSON
        analysis_results = _extract_json_from_text(response)
        
        if not analysis_results:
            if language == 'cn':
                logger.warning("LLM响应未包含有效的JSON，将使用基础分析结果")
            else:
                logger.warning("LLM response didn't contain valid JSON, using basic analysis results")
            # 作为备选，创建一个基本的摘要结果
            analysis_results = _create_basic_analysis(data_summary, user_goal, language)
        
        # 进行验证和修复
        required_keys = ["data_overview", "quality_issues", "key_insights", 
                          "preparation_recommendations", "overall_summary"]
        
        missing_keys = [key for key in required_keys if key not in analysis_results]
        if missing_keys:
            if language == 'cn':
                logger.warning(f"LLM分析结果缺少以下键: {missing_keys}，将添加默认值")
            else:
                logger.warning(f"LLM analysis results missing keys: {missing_keys}, adding defaults")
            for key in missing_keys:
                if key == "data_overview":
                    analysis_results[key] = f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns"
                elif key == "quality_issues":
                    analysis_results[key] = []
                elif key == "key_insights":
                    analysis_results[key] = []
                elif key == "preparation_recommendations":
                    analysis_results[key] = []
                elif key == "overall_summary":
                    analysis_results[key] = f"Basic analysis of dataset with {df.shape[0]} rows and {df.shape[1]} columns."
        
        # 添加/确保中文总结存在 (如果需要)
        if language == 'cn' and "overall_summary_chinese" not in analysis_results:
            analysis_results["overall_summary_chinese"] = _generate_chinese_summary(
                analysis_results.get("overall_summary", "数据分析完成。"),
                analysis_results.get("data_overview", "")
            )
        
        # 添加原始的列分析以供后续步骤使用
        analysis_results["column_analysis"] = column_analyses
        analysis_results["potential_target_variable"] = potential_target
        analysis_results["potential_id_columns"] = potential_id_columns
        
        if language == 'cn':
            logger.info("LLM数据分析完成")
            if "overall_summary_chinese" in analysis_results:
                logger.info(f"中文总结: {analysis_results.get('overall_summary_chinese')}")
        else:
            logger.info(f"LLM data analysis completed")
            
        return analysis_results
        
    except Exception as e:
        if language == 'cn':
            logger.error(f"LLM数据分析过程中出错: {str(e)}")
        else:
            logger.error(f"Error during LLM data analysis: {str(e)}")
        # 返回基本分析结果作为备选
        return _create_basic_analysis(data_summary, user_goal, language)


def _create_basic_analysis(data_summary: Dict[str, Any], user_goal: Optional[str] = None, language: str = 'en') -> Dict[str, Any]:
    """
    创建基本的数据分析结果，作为LLM分析失败时的备选
    
    Args:
        data_summary: 数据摘要字典
        user_goal: 用户指定的目标
        language: 输出语言 - 'en'为英文, 'cn'为中文和英文
        
    Returns:
        Dict: 包含基本分析结果的字典
    """
    if language == 'cn':
        logger.info("生成基本数据分析结果...")
    else:
        logger.info("Generating basic data analysis results...")
    
    # 提取基本信息
    rows = data_summary.get("shape", {}).get("rows", 0)
    cols = data_summary.get("shape", {}).get("columns", 0)
    columns = data_summary.get("columns", [])
    column_analysis = data_summary.get("column_analysis", [])
    potential_target = data_summary.get("potential_target_variable")
    potential_id_columns = data_summary.get("potential_id_columns", [])
    
    # 识别数据质量问题
    quality_issues = []
    for col_info in column_analysis:
        col_name = col_info.get("column_name", "")
        missing_pct = col_info.get("missing_values", {}).get("percentage", 0)
        
        # 检查缺失值
        if missing_pct > 0:
            if language == 'cn':
                issue = {
                    "issue_type": "缺失值",
                    "affected_columns": [col_name],
                    "description": f"列 '{col_name}' 有 {missing_pct}% 的值缺失",
                    "recommendation": "考虑使用均值/中位数/众数填充或删除这些行"
                }
            else:
                issue = {
                    "issue_type": "Missing Values",
                    "affected_columns": [col_name],
                    "description": f"Column '{col_name}' has {missing_pct}% missing values",
                    "recommendation": "Consider imputing with mean/median/mode or dropping these rows"
                }
            quality_issues.append(issue)
        
        # 检查日期格式不一致
        if col_info.get("likely_date_format") and col_info.get("distribution") == "datetime":
            if language == 'cn':
                issue = {
                    "issue_type": "日期格式",
                    "affected_columns": [col_name],
                    "description": f"列 '{col_name}' 可能包含非标准格式的日期",
                    "recommendation": "将此列转换为日期时间类型"
                }
            else:
                issue = {
                    "issue_type": "Date Format",
                    "affected_columns": [col_name],
                    "description": f"Column '{col_name}' may contain dates in non-standard format",
                    "recommendation": "Convert this column to datetime type"
                }
            quality_issues.append(issue)
    
    # 基本建议
    recommendations = []
    # 缺失值处理
    missing_value_cols = [issue["affected_columns"][0] for issue in quality_issues if issue["issue_type"] in ["缺失值", "Missing Values"]]
    if missing_value_cols:
        if language == 'cn':
            recommendations.append("处理缺失值: " + ", ".join(missing_value_cols))
        else:
            recommendations.append("Handle missing values in: " + ", ".join(missing_value_cols))
    
    # 日期转换
    date_cols = [issue["affected_columns"][0] for issue in quality_issues if issue["issue_type"] in ["日期格式", "Date Format"]]
    if date_cols:
        if language == 'cn':
            recommendations.append("转换日期格式: " + ", ".join(date_cols))
        else:
            recommendations.append("Convert date formats in: " + ", ".join(date_cols))
    
    # 移除ID列
    if potential_id_columns:
        if language == 'cn':
            recommendations.append(f"考虑从分析中排除ID列: {', '.join(potential_id_columns)}")
        else:
            recommendations.append(f"Consider excluding ID columns from analysis: {', '.join(potential_id_columns)}")
    
    # 确认目标变量
    if potential_target:
        if language == 'cn':
            recommendations.append(f"确认 '{potential_target}' 是否为目标变量")
        else:
            recommendations.append(f"Confirm if '{potential_target}' is the target variable")
    
    # 生成概述
    if language == 'cn':
        overview = f"数据集包含 {rows} 行和 {cols} 列。"
        if potential_target:
            overview += f" 可能的目标变量是 '{potential_target}'。"
        if potential_id_columns:
            overview += f" 检测到可能的ID列: {', '.join(potential_id_columns)}。"
    else:
        overview = f"The dataset contains {rows} rows and {cols} columns."
        if potential_target:
            overview += f" Potential target variable is '{potential_target}'."
        if potential_id_columns:
            overview += f" Detected potential ID columns: {', '.join(potential_id_columns)}."
    
    # 创建整体摘要
    if language == 'cn':
        overall_summary = f"数据集包含 {rows} 行和 {cols} 列，"
        if quality_issues:
            overall_summary += f"存在 {len(quality_issues)} 个数据质量问题需要解决，"
        overall_summary += "需要进一步数据准备和特征工程以适合建模。"
        
        overall_summary_chinese = f"这个数据集有 {rows} 行和 {cols} 列。"
        if quality_issues:
            overall_summary_chinese += f"存在一些数据质量问题，包括缺失值和格式不一致。"
        overall_summary_chinese += f"建议进行数据清洗和特征工程，以便更好地进行后续分析和建模。"
    else:
        overall_summary = f"The dataset consists of {rows} rows and {cols} columns, "
        if quality_issues:
            overall_summary += f"has {len(quality_issues)} data quality issues to address, "
        overall_summary += "and requires further data preparation and feature engineering for modeling."
    
    # 创建分析结果
    analysis_results = {
        "data_overview": overview,
        "quality_issues": quality_issues,
        "key_insights": [],  # 简单分析无法提供洞察
        "preparation_recommendations": recommendations,
        "overall_summary": overall_summary,
        "column_analysis": column_analysis,
        "potential_target_variable": potential_target,
        "potential_id_columns": potential_id_columns
    }
    
    # 添加中文总结 (如果需要)
    if language == 'cn':
        analysis_results["overall_summary_chinese"] = overall_summary_chinese
    
    if language == 'cn':
        logger.info("基本数据分析生成完成")
    else:
        logger.info("Basic data analysis generation completed")
        
    return analysis_results

def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """从文本中提取JSON"""
    # 尝试直接解析整个文本
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试在文本中查找JSON对象
    try:
        # 尝试查找被三个反引号括起来的代码块（markdown 格式）
        matches = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if matches:
            return json.loads(matches.group(1))
        
        # 尝试查找以大括号开始和结束的部分
        matches = re.search(r'(\{[\s\S]*\})', text)
        if matches:
            return json.loads(matches.group(1))
    except (json.JSONDecodeError, AttributeError):
        pass
    
    return None

# Example Usage (for testing within the module)
if __name__ == '__main__':
    # Create a dummy DataFrame for testing
    data = {
        'StudentID': [101, 102, 103, 104, 105, 106, 101],
        'FullName': ['Alice Smith', ' Bob Johnson ', 'Charlie Brown', 'David Williams', 'Eve Davis', 'Frank Miller', 'Alice Smith'],
        'Score': [85, 92, 78, 88, 95, 60, 85],
        'Grade': ['B', 'A', 'C', 'B', 'A', 'D', 'B'],
        'Attendance (%)': ['95', '100', '80', '92', '98', ' 70', '95'], # Numeric as string, whitespace
        'LastLogin': ['2024-01-10', '2024-01-09', '2024/01/11', '12-Jan-2024', None, '2024-01-08', '2024-01-10'], # Mixed date format, missing
        'Notes': ['Good progress', 'Excellent work', 'Needs improvement', '', 'Top score!', None, 'Good progress'], # Text, empty strings, missing
        'IsActive': [True, True, False, True, True, False, True],
        'Campus': ['Main', 'Main', 'Main', 'Main', 'Main', 'Main', 'Main'] # Constant
    }
    dummy_df = pd.DataFrame(data)
    dummy_df.loc[3, 'Score'] = 999 # Add outlier

    print("--- Running Dummy Analysis ---")
    # Make sure you have a .env file with GOOGLE_API_KEY=your_key
    analysis = analyze_data_with_llm(dummy_df, user_goal="Predict Grade based on other features")
    print("\n--- Analysis Result ---")
    if analysis:
        print(json.dumps(analysis, indent=2))
    else:
        print("Analysis failed.")

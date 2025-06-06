# ML Agent - 数据处理工作流设计

本文档概述了 ML Agent 项目中数据处理工作流的设计。

## 工作流架构：LLM驱动的四阶段智能流程 (Analyze -> Plan -> Execute -> Evaluate)

**注意:** 此处描述的原始四阶段流程现在正在演进。更复杂的特征工程（Feature Engineering, FE）步骤被移至后续的 **特征工程阶段 (Feature Engineering Stage)**，模型特定的FE和训练则由 **模型代理阶段 (Model Agent Stage)** 处理。`data_agent` 现在更专注于数据的初始质量保证。

1.  **分析阶段 (Analyze Stage):**
    *   **核心模块:** `src/ml_agent/llm_data_analyzer.py`
    *   **输入:** 原始 DataFrame (从 CSV 加载), (可选) 用户目标。
    *   **过程:**
        *   计算数据的基本统计信息和样本。
        *   构建包含统计数据、样本和用户目标的 Prompt。
        *   调用 LLM API (例如 Gemini) 进行深度分析。
        *   LLM 推断列的语义类型、识别数据质量问题（缺失、异常、格式错误、PII 等）、评估整体数据质量，并提出初步**清洗和基础转换**建议。
        *   解析 LLM 的响应，生成结构化的 JSON 报告。
    *   **输出:** `analysis_report_*.json` (保存在 `reports/` 目录)，包含对每列的详细分析和建议。

2.  **规划阶段 (Plan Stage):**
    *   **核心模块:** `src/ml_agent/llm_data_planner.py`
    *   **输入:** `analysis_report_*.json`, (可选) 用户目标, `data_preparer.py` 中可用**清洗/基础转换**函数的描述。
    *   **过程:**
        *   构建包含分析报告、用户目标和可用工具列表的 Prompt。
        *   调用 LLM API。
        *   LLM 根据分析结果和可用工具，生成一个逻辑有序的、包含具体函数调用（函数名、参数、策略）的**数据准备执行计划**。
        *   解析 LLM 的响应，生成结构化的 JSON 计划列表。
    *   **输出:** `execution_plan_*.json` (保存在 `reports/` 目录)，包含详细的**数据准备**步骤和原因。

3.  **执行阶段 (Execute Stage - Data Preparation Focus):**
    *   **核心模块:** `agent_workflow.py` (或主控 Agent 逻辑) 调用 `data_preparer.py`
    *   **输入:** 原始 DataFrame (的副本), `execution_plan_*.json`, `analysis_report_*.json` (用于上下文)。
    *   **过程:**
        *   按顺序迭代执行计划中的每个**数据准备**步骤。
        *   处理特殊指令 (例如，`COMMENT` 用于移除列)。
        *   动态地从 `src/ml_agent/data_preparer.py` 模块获取并调用计划中指定的**清洗/基础转换**函数。
        *   记录详细的执行日志。
    *   **输出:**
        *   准备好的 DataFrame (传递给下一阶段，即特征工程阶段)。
        *   详细的执行日志 (`agent_workflow_*.log` 在 `logs/` 目录)。

4.  **评估阶段 (Evaluate Stage - After Data Preparation):**
    *   **核心模块:** `src/ml_agent/llm_data_evaluator.py` (或类似评估逻辑)
    *   **输入:** 准备后的 DataFrame, `analysis_report_*.json`, `execution_plan_*.json`, (可选) 用户目标。
    *   **过程:**
        *   计算准备后数据的统计信息和样本。
        *   构建包含准备后数据摘要、原始分析报告、执行计划和用户目标的 Prompt。
        *   调用 LLM API 进行评估。
        *   LLM 评估准备后数据的质量、问题解决情况，并可能对后续的特征工程提供建议。
        *   解析 LLM 的响应，生成结构化的 JSON 评估报告。
    *   **输出:**
        *   `evaluation_report_*.json` (保存在 `reports/` 目录)，包含详细的评估结果和建议。
        *   准备后的 DataFrame (传递给特征工程阶段)。

## 核心数据准备模块：`data_preparer.py`

**角色:** 作为数据**清洗和基础转换**的**工具箱 (Toolkit)**，被 **执行阶段** 动态调用。

该模块包含用于常见数据清洗和基础准备任务的可重用函数。Agent 的 **规划** 阶段生成对这些函数的调用，**执行** 阶段实际执行这些调用。更复杂的特征工程任务由 `feature_engineer.py` 处理。

### 可用函数 (示例 - 由 Agent 动态检查):

*   **`impute_missing(df, column, strategy='mean'/'median'/'mode')`**
    *   **目的:** 填充单个列中的缺失值 (使用基础策略)。
    *   **注意:** 通常就地修改 DataFrame。

*   **`convert_to_numeric(df, column, errors='coerce')`**
    *   **目的:** 使用 `pd.to_numeric` 将列转换为数值类型。
    *   **注意:** 通常就地修改 DataFrame。

*   **`remove_duplicates(df)`**
    *   **目的:** 使用 `df.drop_duplicates()` 移除重复行。
    *   **注意:** 返回修改后的 DataFrame。

*   **`strip_whitespace(df, column)`**
    *   **目的:** 移除字符串/对象列中的前导/尾随空格。
    *   **注意:** 通常就地修改 DataFrame。

*   **`remove_constant_columns(df)`**
    *   **目的:** 移除只包含单个唯一值的列。
    *   **注意:** 返回修改后的 DataFrame。

*   **`extract_date_features(df, column, drop_original=True)`**
    *   **目的:** 从日期时间列中提取年、月、日、星期几等基础时间特征。
    *   **注意:** 通常就地修改 DataFrame。

*   **(可能还有其他基础清洗函数)**


data prepare阶段执行完成后，进行对结果的中文总结
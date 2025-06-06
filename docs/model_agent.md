# ML Agent - 特征工程与模型代理工作流设计 (model_agent.md)

本文档概述了在数据准备 (`data_agent`) 之后，ML Agent 项目中特征工程 (Feature Engineering, FE) 和模型代理 (Model Agent) 的工作流设计。该设计旨在平衡通用特征处理和模型特定优化的需求，并以 H2O AutoML 作为模型训练和评估的核心引擎示例。

## 工作流架构：衔接数据准备，驱动模型训练

在 `data_agent` 完成数据清洗、验证和基础转换后，工作流进入以下阶段：

1.  **特征工程阶段 (Feature Engineering Stage):**
    *   **核心模块:** `src/ml_agent/llm_feature_planner.py` (规划), `agent.py` / `workflow` (执行控制), `src/ml_agent/feature_engineer.py` (工具箱)
    *   **输入:** `data_agent` 输出的准备好的 DataFrame, (可选) 用户目标, `analysis_report.json` (来自 data_agent)。
    *   **过程:**
        *   **(规划 - Plan):**
            *   LLM (通过 `llm_feature_planner.py`) 分析准备好的数据（可能结合原始分析报告和用户目标）。
            *   LLM 参考 `feature_engineer.py` 中可用的**通用 FE 函数**列表。
            *   LLM 生成一个逻辑有序的、包含具体 FE 函数调用（函数名、参数、策略）的**通用特征工程执行计划**。例如，规划对分类变量进行编码、对数值变量进行缩放、处理异常值、处理高基数类别、创建基础交互特征等。
            *   计划以结构化 JSON 格式输出 (`feature_plan_*.json`)。
        *   **(执行 - Execute):**
            *   主控 Agent (`agent.py` / `workflow`) 按顺序迭代执行 `feature_plan.json` 中的步骤。
            *   动态调用 `src/ml_agent/feature_engineer.py` 中的函数。
            *   存储过程中产生的拟合对象（例如，编码器、缩放器）以便后续应用或逆操作。
            *   记录详细的执行日志。
        *   **(评估 - Evaluate) (可选但推荐):**
            *   LLM (可能需要 `llm_feature_evaluator.py`) 评估特征工程后的数据。
            *   评估生成的特征是否合适、是否存在潜在问题（如数据泄漏、维度灾难）、是否为后续建模准备就绪。
            *   生成评估报告 (`feature_evaluation_report_*.json`)。
    *   **输出:**
        *   经过通用特征工程处理的 DataFrame。
        *   特征工程相关的拟合对象字典。
        *   `feature_plan_*.json`, `feature_evaluation_report_*.json`。
        *   详细执行日志。

2.  **模型代理阶段 (Model Agent Stage - 以 H2O AutoML 为例):**
    *   **核心模块:** `src/ml_agent/llm_model_configurator.py` (规划/配置), `agent.py` / `workflow` (执行控制), `src/ml_agent/h2o_executor.py` (执行引擎)
    *   **输入:** 特征工程阶段输出的 DataFrame, (可选) 用户目标, `analysis_report.json`, `feature_plan.json`。
    *   **过程:**
        *   **(规划/配置 - Plan/Configure):**
            *   LLM (通过 `llm_model_configurator.py`) 分析最终的数据特征、用户目标（分类/回归）、以及之前的处理步骤。
            *   LLM 确定 H2O AutoML 的核心配置：任务类型、目标变量、特征列（可能排除 ID 或高相关特征）、时间/模型数量限制、评估指标。
            *   **模型特定 FE (隐式/配置):** LLM 根据数据特性（如高基数列的存在）和 H2O 文档，决定是否启用 H2O 内置的预处理/FE 功能。例如，如果存在高基数分类特征且目标模型包含树模型，LLM 可能建议在 H2O 配置中加入 `preprocessing = ["target_encoding"]` (参考 H2O 文档的实验性功能)。这是通过配置实现模型特定 FE 的方式。
            *   LLM 生成 H2O AutoML 的执行配置 (`h2o_config_*.json`)。
        *   **(执行 - Execute):**
            *   主控 Agent (`agent.py` / `workflow`) 调用 `src/ml_agent/h2o_executor.py`。
            *   `h2o_executor.py` 根据 `h2o_config.json` 初始化并运行 H2O AutoML。
            *   H2O AutoML 内部进行模型训练、超参数调优、交叉验证和模型集成。
            *   记录 H2O 的执行日志和模型输出。
        *   **(评估 - Evaluate):**
            *   H2O AutoML 自动生成模型排行榜 (Leaderboard)。
            *   主控 Agent 或 `h2o_executor.py` 解析 Leaderboard，提取关键评估指标和最佳模型信息。
            *   LLM (可选，`llm_model_evaluator.py`) 可以对 H2O 的结果进行二次解读，结合用户目标给出更自然的语言总结和建议，或评估整个流程的有效性。
            *   生成最终评估报告 (`final_evaluation_report_*.json`) 和模型摘要。
    *   **输出:**
        *   H2O AutoML Leaderboard (模型排行榜)。
        *   训练好的最佳模型 (通常是 H2O MOJO 或 Pojo 格式)。
        *   模型相关的评估指标和可视化 (如变量重要性、混淆矩阵)。
        *   `h2o_config_*.json`, `final_evaluation_report_*.json`。
        *   H2O 执行日志。

## 核心特征工程模块：`feature_engineer.py`

**角色:** 作为**通用特征工程**的**工具箱 (Toolkit)**，被 **特征工程执行阶段** 动态调用。

该模块包含用于常见特征转换、生成和选择任务的可重用函数。

### 可用函数 (示例 - 由 Agent 动态检查):

*   **`encode_categorical(df, column, strategy='one-hot', drop_original=True, drop_first=False, **kwargs)`**
    *   **目的:** 对分类/对象列进行编码。
    *   **策略:** 'one-hot', 'label', 'target' (需要目标变量), 'frequency' 等。
    *   **返回:** 修改后的 DataFrame 和拟合好的编码器对象。
*   **`scale_numerical(df, columns, strategy='standard', **kwargs)`**
    *   **目的:** 缩放一个或多个数值列。
    *   **策略:** 'standard', 'minmax', 'robust' 等。
    *   **返回:** 修改后的 DataFrame 和拟合好的缩放器对象。
*   **`handle_outliers(df, columns, strategy='clip_iqr', iqr_multiplier=1.5, **kwargs)`**
    *   **目的:** 处理一个或多个数值列中的异常值。
    *   **策略:** 'clip_iqr', 'clip_std', 'remove', 'winsorize' 等。
    *   **返回:** 修改后的 DataFrame。
*   **`handle_high_cardinality(df, column, threshold=0.95, method='frequency', replace_with='Other', **kwargs)`**
    *   **目的:** 处理高基数分类变量。
    *   **方法:** 'frequency' (保留最常见类别), 'grouping' (基于目标变量相似性分组 - 需要目标), 'target_encoding' 等。
    *   **返回:** 修改后的 DataFrame 和可能需要的映射。
*   **`create_interaction_features(df, columns1, columns2, interaction_type='multiply', **kwargs)`**
    *   **目的:** 创建特征之间的交互项。
    *   **类型:** 'multiply', 'divide', 'polynomial' (可能需要更复杂实现) 等。
    *   **返回:** 修改后的 DataFrame。
*   **`bin_numerical(df, column, bins=5, strategy='quantile', **kwargs)`**
    *   **目的:** 对数值特征进行分箱。
    *   **策略:** 'uniform', 'quantile', 'kmeans'。
    *   **返回:** 修改后的 DataFrame 和分箱边界。
*   **(可能还有其他通用 FE 函数，如特征选择、降维等)**

## 模型特定 FE 处理方式

模型特定的 FE 主要在 **模型代理阶段 (Model Agent Stage)** 处理，有两种主要方式：

1.  **通过配置利用模型引擎内置功能:** 如上文 H2O 示例所示，LLM 可以根据模型引擎（如 H2O）的能力，智能地配置其内置的 FE 或预处理选项 (如 `preprocessing = ["target_encoding"]`)。
2.  **显式执行特定转换:** 如果模型需要 H2O (或其他引擎) 无法直接提供的特定 FE (例如，为 Keras MLP 创建复杂的 Embedding 层，或为特定模型实现高级 Target Encoding 变体)，则需要在模型代理阶段**之前**或**作为其第一步**添加一个专门的 LLM 规划和执行环节。这可能涉及：
    *   一个 `model_specific_featurizer.py` 模块。
    *   或者，每个 `model_agent` (如 `mlp_agent`, `xgboost_agent`) 内部包含其特定的 FE 逻辑。
    *   LLM 需要识别出模型类型，并规划调用这些特定的 FE 函数。

当前设计侧重于方式 1，利用 H2O AutoML 的能力简化流程。如果需要更强的模型特定 FE 定制，则需要考虑扩展设计以支持方式 2。

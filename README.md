# ML-Agent: 基于大模型的智能机器学习自动化系统

![screenshot0](./static/img/screenshot0.png "系统界面")

## 项目简介

ML-Agent是一个创新的机器学习自动化平台，它结合了大语言模型(LLM)的智能推理能力和H2O AutoML的高效建模能力，实现了从原始数据到高质量模型的端到端自动化流程。

**核心思想**: 将数据分析、特征工程、模型训练等关键步骤交由大型语言模型（LLM）进行规划，然后自动执行这些规划，最终利用 H2O AutoML 进行高效的多模型训练与评估。

这个系统适合各类用户：
- **数据科学初学者**: 无需深厚的机器学习知识，也能构建高质量模型。
- **经验丰富的数据科学家**: 自动化繁琐任务，专注于更关键的业务问题，显著提高工作效率。
- **业务分析师**: 快速从数据中获取可执行的见解，无需编写复杂的代码。

## ✨ 功能特点

- **智能工作流推荐**: 利用大语言模型分析数据特征和用户需求，智能规划完整的数据处理和建模流程。
- **自动代码生成与执行**: 自动生成并执行数据清洗、特征工程等步骤的代码，全程无需手动编码。
- **企业级AutoML引擎**: 集成H2O AutoML，自动训练和评估多种业界领先的模型，包括广义线性模型(GLM)、随机森林(RF)、梯度提升机(GBM)、XGBoost和深度神经网络(DNN)。
- **模型解释与可视化**: 提供变量重要性、混淆矩阵等模型洞察功能，增强结果的可信度。
- **用户友好的Web界面**: 提供简洁直观的Web界面，用户点击几下即可完成复杂的建模任务。

## 🛠️ 技术架构

### 核心组件

1. **LLM推理引擎**: 接入GPT系列模型，作为系统的"大脑"，负责规划数据准备、特征工程和模型配置。
2. **数据准备模块 (`data_preparer.py`)**: 包含一系列用于执行数据清洗和预处理任务（如缺失值填充、数据类型转换、异常值处理）的函数。
3. **特征工程模块 (`feature_engineer.py`)**: 包含用于执行特征创建和转换（如交互特征、分箱、多项式特征）的函数库。
4. **H2O AutoML执行器 (`h2o_executor.py`)**: 封装了H2O AutoML的调用逻辑，负责模型的并行训练和调优。
5. **主控Agent (`agent.py`)**: 作为系统的总协调者，负责调用LLM进行规划，并调度其他模块执行具体的任务。
6. **Web应用层 (`app.py`)**: 基于Flask的用户交互界面。

### 技术栈

- **后端**: Python, **OpenAI API (直接调用)**, H2O
- **前端**: HTML, CSS, JavaScript, Bootstrap
- **核心数据处理**: Pandas, NumPy
- **MLOps (部分功能)**: MLflow

## 🚀 快速开始

### 1. 环境配置

克隆本仓库到您的本地机器：
```bash
git clone https://github.com/your-repo/ml-agent.git
cd ml-agent
```

**💻 Windows用户**: 请参阅 [Windows兼容性指南](WINDOWS_COMPATIBILITY.md) 获取详细的安装和配置说明。

### 2. 安装依赖

我们强烈建议在虚拟环境中使用本项目。

```bash
# 创建并激活虚拟环境 (例如使用 venv)
python -m venv env
source env/bin/activate  # on Windows: env\Scripts\activate

# 安装所有必需的依赖项
pip install -r requirements.txt
```
**注意**: `requirements.txt` 文件包含了 `xgboost`。请确保其成功安装，以便H2O AutoML可以训练XGBoost模型。

### 3. 设置API密钥

将项目根目录下的 `.env.example` 文件复制为 `.env`，并填入您的 `OPENAI_API_BASE` 和 `OPENAI_API_KEY`。

### 4. 启动Web界面

```bash
python app.py
```
启动成功后，在浏览器中访问 `http://localhost:8000` 即可开始使用。

## 📊 模型支持

ML-Agent支持H2O AutoML提供的多种机器学习任务和模型类型：

- **支持任务**: 二分类、多分类、回归
- **核心模型**: GLM, Random Forest (DRF), GBM, Deep Learning, Stacked Ensembles, StackedEnsemble, AdaBoost, SVM等
- **评估指标**: AUC, Logloss, RMSE, MAE, Precision, Recall等

### 🖥️ 平台兼容性

| 平台 | 支持状态 | 算法数量 | 备注 |
|------|----------|----------|------|
| **Windows** | ✅ 完全支持 | 17个算法 | XGBoost有限制，可用GBM替代 |
| **macOS (Intel)** | ✅ 完全支持 | 18个算法 | 完整XGBoost支持 |
| **macOS (Apple Silicon)** | ✅ 部分支持 | 17个算法 | XGBoost不可用 |
| **Linux** | ✅ 完全支持 | 18个算法 | 最佳兼容性 |

**注意**: 无论哪个平台，ML-Agent都能自动检测并跳过不支持的算法，确保系统正常运行。

## 📝 命令行(CLI)使用示例

除了Web界面，您也可以通过纯Python脚本调用Agent。

```python
# run_main_demo.py
from src.ml_agent.agent import H2OMLAgent
import pandas as pd

# 加载数据
data = pd.read_csv("your_data.csv")

# 创建H2O ML Agent实例
agent = H2OMLAgent(log=True, log_path="logs/", model_directory="models/")

# 定义任务指令和目标变量
user_instructions = "这是一个二分类问题，请为我构建一个分类模型。我希望模型的运行时间不超过60秒。"
target_variable = "target_column_name" # 替换为实际的目标列名

# 调用Agent执行任务
agent.invoke_agent(
    df=data,
    user_instructions=user_instructions,
    target_variable=target_variable,
    max_runtime_secs=60, # 也可以在这里覆盖用户指令中的参数
)

# 打印结果
print(agent.get_workflow_summary(markdown=True))
print("---------- [ H2O 模型排行榜 ] ----------")
print(agent.get_leaderboard())
```

## 📁 项目结构

```
ml-agent/
├── app.py              # Flask Web应用
├── src/
│   └── ml_agent/
│       ├── __init__.py
│       ├── agent.py            # 主控Agent (Orchestrator)
│       ├── llm_interface.py    # LLM API 接口封装
│       ├── data_preparer.py    # 数据准备工具函数
│       ├── feature_engineer.py # 特征工程工具函数
│       ├── h2o_executor.py     # H2O AutoML 执行器
│       └── utils.py            # 辅助工具函数
├── run_main_demo.py    # 命令行运行脚本
├── .env.example        # API密钥配置示例
├── requirements.txt    # 项目依赖
├── static/             # Web静态资源
├── templates/          # HTML模板
├── logs/               # 日志目录
└── models/             # 模型保存目录
```

---

> "将机器学习的复杂性隐藏在智能界面之后，让每个人都能构建高质量的预测模型。"

*本节之后的内容为H2O AutoML的详细技术文档，保留供深度用户参考。*

---

# H2O AutoML在ML-Agent中的完整运行机制文档

## 一、总体架构与工作流程

ML-Agent是一个结合大语言模型(LLM)和H2O AutoML的智能机器学习自动化平台，整个流程分为三个主要阶段：

```
数据处理 → 特征工程 → 模型训练与评估
```

### 完整流程图

```
原始数据 + 用户指令
    ↓
加载数据
    ↓
LLM分析：数据描述与规划
    ↓
执行数据准备步骤
    ↓
LLM分析：特征工程规划
    ↓
执行特征工程
    ↓
LLM分析：配置H2O AutoML参数
    ↓
启动H2O AutoML训练多个模型
    ↓
生成模型排行榜与评估结果
```

## 二、H2O AutoML的运行原理

### 1. 初始化与启动过程

H2O AutoML在本地启动一个基于Java的服务器进程，通过JVM运行H2O的机器学习算法：

```python
# 初始化H2O集群
h2o.init()  # 这会在本地启动一个Java服务器进程
```

从您的运行日志可以看到这个过程：
```
Initializing H2O cluster...
Checking whether there is an H2O instance running at http://localhost:54321..... not found.
Attempting to start a local H2O server...
Java Version: openjdk version "21.0.6" 2025-01-21 LTS
Starting server from .../h2o.jar
Server is running at http://127.0.0.1:54321
```

### 2. 数据转换为H2O格式

ML-Agent将处理过的pandas DataFrame转换为H2O专用的数据格式：

```python
# 转换为H2O格式
h2o_frame = h2o.H2OFrame(processed_df)
```

### 3. 模型训练执行过程

H2O AutoML根据配置参数同时训练多种模型类型：

- **Random Forest (DRF)**
- **Gradient Boosting Machines (GBM)**
- **Extreme Gradient Boosting (XGBoost)** 
- **Deep Learning (神经网络)**
- **Generalized Linear Models (GLM)**
- **Stacked Ensembles** (多模型组合)

训练过程是计算密集型的，因为它会：
1. 并行训练多种模型
2. 对每种模型进行超参数调优
3. 使用交叉验证评估性能
4. 构建Stacked Ensemble模型来组合最佳模型

## 三、目标变量(Target Variable)的指定机制

### 1. 目标变量确定流程

ML-Agent提供了多种方式来确定目标变量：

1. **用户直接指定**：在调用Agent时明确指定
   ```python
   agent.invoke_agent(data_raw=data, target_variable="your_target_column")
   ```

2. **自动检测**：如果用户没有指定，系统会：
   - 在`llm_data_analyzer.py`中通过分析数据来推断可能的目标变量
   - 分析会考虑列名、数据类型、唯一值数量等因素
   - 结果保存在分析报告中的`potential_target`字段

3. **工作流传递**：从数据分析阶段→特征工程→模型配置
   - 工作流会在各个阶段之间传递target_variable
   - 确保整个流程使用一致的目标变量

### 2. 自动目标变量检测的代码位置

从代码分析来看，这主要在两个地方实现：

```python
# 在llm_data_analyzer.py中
def analyze_data(df, ...):
    # 分析数据特征，推断可能的目标变量
    potential_target = identify_potential_target(df)
    analysis_report['potential_target'] = potential_target
    
# 在llm_model_configurator.py中获取并使用
def generate_h2o_config(analysis_report, ...):
    # 从分析报告中提取目标变量
    target_var = analysis_report.get('potential_target', '')
```

## 四、H2O AutoML的配置参数和优化逻辑

### 1. 关键配置参数

H2O AutoML配置由LLM根据数据分析和用户指令生成，主要包括：

| 参数 | 说明 | 影响 |
|------|------|------|
| `max_models` | 最大训练模型数量 | 控制模型训练量 |
| `max_runtime_secs` | 最长运行时间(秒) | 控制总训练时间 |
| `sort_metric` | 排序指标 | 决定最优模型选择标准 |
| `include_algos` | 包含的算法 | 指定要训练的算法类型 |
| `exclude_algos` | 排除的算法 | 指定不训练的算法类型 |
| `balance_classes` | 是否平衡类别 | 处理不平衡数据集 |
| `standardize` | 是否标准化 | 数据预处理选项 |

### 2. 模型训练过程的内部阶段

当H2O AutoML运行时，它会按以下顺序执行操作：

1. **初始化训练**：设置资源和参数
2. **数据预处理**：处理缺失值、编码分类变量等
3. **基础模型训练**：训练各种单一模型
   - 随机森林模型(多个不同配置)
   - GBM模型(多个不同配置)
   - XGBoost模型(如果可用)
   - GLM(线性模型)
   - 深度学习模型
4. **超参数调优**：对每种算法类型进行网格搜索或随机搜索
5. **Stacked Ensemble构建**：组合最佳模型
6. **模型评估与排序**：根据指定指标对所有模型排序
7. **选择最佳模型**：确定排行榜第一名

## 五、资源使用与优化建议

### 1. 为什么H2O AutoML会消耗大量资源

- 并行训练多个模型类型
- 对每种模型执行超参数调优(构建大量候选模型)
- 默认配置过于激进(尤其对个人电脑)
- Java虚拟机(JVM)占用大量内存

### 2. 优化配置建议

针对不同环境，可以调整以下参数控制资源使用：

**个人电脑(轻量级)**：
```python
h2o_params = {
    "max_models": 5,  # 减少模型数量
    "max_runtime_secs": 300,  # 限制为5分钟
    "include_algos": ["GLM", "GBM"],  # 只使用轻量级算法
}
```

**中等规模服务器**：
```python
h2o_params = {
    "max_models": 20,
    "max_runtime_secs": 1800,  # 30分钟
}
```

**高性能服务器**：
```python
h2o_params = {
    "max_models": 50,
    "max_runtime_secs": 7200,  # 2小时
}
```

### 3. 在ML-Agent中应用优化

可以在`llm_model_configurator.py`修改默认配置：

```python
# 修改默认H2O参数
formatted_response = {
    # ...
    "h2o_automl_parameters": {
        "max_models": 10,  # 降低默认值
        "max_runtime_secs": 600,  # 10分钟而非1小时
        # ...其他参数...
    }
}
```

## 六、代码层面的H2O AutoML集成逻辑

H2O AutoML在ML-Agent中的集成主要在以下文件中实现：

1. **h2o_executor.py**：核心模型训练实现
   - 初始化H2O集群
   - 将DataFrame转换为H2O格式
   - 配置和启动AutoML
   - 收集结果和模型评估

2. **llm_model_configurator.py**：生成H2O配置
   - 分析数据特征
   - 根据LLM建议生成H2O参数

3. **agent_workflow.py/agent.py**：协调整个流程
   - 管理目标变量的确定和传递
   - 调用H2O训练
   - 保存结果和报告

## 总结

H2O AutoML是一个强大的自动化机器学习工具，它在本地启动服务器来训练多种模型。在ML-Agent中，它被智能地集成到一个端到端工作流中，从数据分析到特征工程再到模型训练。高资源消耗是由其并行训练多模型的特性导致的，但可以通过配置参数来控制。

对于用户便于使用的考虑，目标变量可以通过多种方式指定：用户明确指定、系统自动检测或在网页界面中选择，满足不同用户的需求和知识水平。

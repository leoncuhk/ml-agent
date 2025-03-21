# 机器学习自动化建模AI Agent

原始数据 → 数据描述 → LLM推荐步骤 → H2O AutoML训练 → 多个模型 → 选择最佳模型

这个项目实现了一个基于大语言模型(LLM)的机器学习自动化建模AI Agent系统。该系统可以根据用户输入的数据和指令，自动推荐机器学习流程，生成并执行代码，创建和评估多种机器学习模型，并提供详细的结果报告。

![screenshot0](./static/img/screenshot0.png "v0版")

## 功能特点

- 基于LLM的智能推荐：利用大语言模型分析数据特征和用户需求，智能推荐合适的机器学习流程
- 自动代码生成与执行：自动生成高质量Python代码并执行，无需手动编码
- H2O AutoML集成：利用H2O AutoML创建和评估多种模型，找出最优解决方案
- 完整的工作流支持：从数据分析、特征工程到模型训练和评估的全流程支持
- 详细结果报告：提供模型性能排名、最佳模型详情等全面报告

## 安装依赖

```bash
pip install openai h2o pandas
```

## 使用方法

### 基本使用

```python
from ml_agent import H2OMLAgent
import pandas as pd

# 加载数据
data = pd.read_csv("your_data.csv")

# 创建H2O ML Agent
agent = H2OMLAgent(log=True, log_path="logs/", model_directory="models/")

# 调用Agent进行自动建模
agent.invoke_agent(
    data_raw=data,
    user_instructions="请执行分类任务，使用最大运行时间30秒。",
    target_variable="target_column_name"  # 替换为实际的目标变量列名
)

# 获取工作流程摘要
summary = agent.get_workflow_summary(markdown=True)
print(summary)

# 获取推荐的机器学习步骤
steps = agent.get_recommended_ml_steps(markdown=True)
print(steps)

# 获取日志摘要
log_summary = agent.get_log_summary(markdown=True)
print(log_summary)
```

### API接口配置

默认使用云服务API接口，你可以修改`ml_agent.py`中的`chat_with_llm`函数来配置自己的API接口：

```python
client = OpenAI(
    base_url="你的API端点",
    api_key="你的API密钥"
)
```

## 项目结构

- `ml_agent.py`: 主要的Agent实现代码
- `logs/`: 日志和生成的代码保存目录
- `models/`: 训练好的模型保存目录

## 示例

以下是一个使用H2O AutoML进行客户流失预测的示例：

```python
import pandas as pd
from ml_agent import H2OMLAgent

# 加载客户流失数据
data = pd.read_csv("churn_data.csv")

# 创建H2O ML Agent
agent = H2OMLAgent(log=True)

# 调用Agent进行客户流失预测
agent.invoke_agent(
    data_raw=data,
    user_instructions="请执行对'Churn'列的分类任务，使用最大运行时间60秒，并重点关注模型精度和召回率。",
    target_variable="Churn"
)

# 获取工作流程摘要
print(agent.get_workflow_summary())
```

## 参考

本项目参考了以下资源：
- H2O AutoML文档
- OpenAI API文档
- 机器学习自动化流水线最佳实践 

---
H2O机器学习代理流程
|
|-- 1. 数据准备
|   |-- 加载数据集（example_data.csv）
|   |-- 数据预览和检查
|
|-- 2. 初始化ML Agent
|   |-- 创建H2OMLAgent实例
|   |-- 设置日志和模型保存路径
|   |-- 初始化H2O环境
|
|-- 3. 调用大模型（LLM）
|   |-- 准备数据描述和用户指令
|   |-- 发送请求到GPT-4o-mini
|   |-- 获取推荐的机器学习步骤
|
|-- 4. 生成AutoML代码
|   |-- 使用内置的H2O AutoML代码模板
|   |-- 将目标变量转换为分类变量
|   |-- 保存代码到日志目录
|
|-- 5. 执行自动机器学习
|   |-- 将数据转换为H2O Frame
|   |-- 设置AutoML参数（max_runtime_secs=30）
|   |-- 训练多个机器学习模型
|   |-- 选择最佳模型
|
|-- 6. 结果收集与报告
|   |-- 获取模型排行榜
|   |-- 保存最佳模型
|   |-- 记录模型路径和ID
|   |-- 生成工作流程摘要
|
|-- 7. 模型使用（predict_with_model.py）
    |-- 加载保存的最佳模型
    |-- 应用模型进行预测
    |-- 显示预测结果和模型性能
    |-- 分析变量重要性
# ML-Agent: 基于大模型的智能机器学习自动化系统

![screenshot0](./static/img/screenshot0.png "系统界面")

## 🌟 项目简介

ML-Agent是一个创新的机器学习自动化平台，它结合了大语言模型(LLM)的智能推理能力和H2O AutoML的高效建模能力，实现了从原始数据到高质量模型的端到端自动化流程。

**核心思想**: 将数据分析、特征工程、模型训练等关键步骤交由大型语言模型（LLM）进行规划，然后自动执行这些规划，最终利用 H2O AutoML 进行高效的多模型训练与评估。

### 🎯 适用用户群体

- **数据科学初学者**: 无需深厚的机器学习知识，也能构建高质量模型
- **经验丰富的数据科学家**: 自动化繁琐任务，专注于更关键的业务问题，显著提高工作效率
- **业务分析师**: 快速从数据中获取可执行的见解，无需编写复杂的代码
- **产品经理**: 验证数据驱动的想法和假设

## ✨ 核心功能特点

- **🧠 智能工作流推荐**: 利用大语言模型分析数据特征和用户需求，智能规划完整的数据处理和建模流程
- **🔄 自动代码生成与执行**: 自动生成并执行数据清洗、特征工程等步骤的代码，全程无需手动编码
- **🚀 企业级AutoML引擎**: 集成H2O AutoML，自动训练和评估多种业界领先的模型
- **📊 模型解释与可视化**: 提供特征重要性图、混淆矩阵等模型洞察功能，增强结果的可信度
- **🎨 用户友好的界面**: 提供Web界面和命令行工具，满足不同用户的使用偏好
- **🔧 灵活的配置选项**: 支持分类和回归任务，可自定义模型训练参数

## 🛠️ 技术架构

### 核心组件

1. **LLM推理引擎**: 接入GPT系列模型，作为系统的"大脑"，负责规划数据准备、特征工程和模型配置
2. **数据准备模块**: 执行数据清洗和预处理任务（缺失值填充、数据类型转换、异常值处理）
3. **特征工程模块**: 执行特征创建和转换（交互特征、分箱、多项式特征）
4. **H2O AutoML执行器**: 封装H2O AutoML调用逻辑，负责模型的并行训练和调优
5. **主控Agent**: 系统总协调者，调用LLM进行规划，并调度其他模块执行任务
6. **Web应用层**: 基于Flask的用户交互界面

### 技术栈

- **后端**: Python 3.8+, OpenAI API, H2O AutoML
- **前端**: HTML5, CSS3, JavaScript, Bootstrap
- **数据处理**: Pandas, NumPy, Polars, PyArrow
- **可视化**: Matplotlib, Seaborn, Plotly
- **部署**: Flask

## 🚀 快速开始

### 系统要求

- **Python**: 3.8 或更高版本
- **Java**: JDK 8 或更高版本（H2O AutoML需要）
- **内存**: 建议 4GB 以上
- **操作系统**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/your-username/ml-agent.git
cd ml-agent

# 创建虚拟环境
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

### 2. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 验证Java环境
java -version
```

### 3. 配置API密钥

```bash
# 复制环境变量模板
cp env.example .env

# 编辑.env文件，添加您的API密钥
# OPENAI_API_BASE=https://api.openai.com/v1
# OPENAI_API_KEY=your_api_key_here
```

### 4. 快速体验

#### 方式一：Web界面
```bash
python app.py
```
访问 `http://localhost:8000` 开始使用

#### 方式二：命令行工具（推荐）
```bash
# 分类任务示例
python run_enhanced_demo.py --data uploads/example_data.csv --target target --task classification

# 回归任务示例  
python run_enhanced_demo.py --data uploads/example_regression_data.csv --target price --task regression

# 自定义配置
python run_enhanced_demo.py \
  --data your_data.csv \
  --target your_target_column \
  --task auto \
  --max_runtime 300 \
  --max_models 15 \
  --instructions "请构建一个高精度的预测模型"
```

## 📊 增强版演示工具

### 命令行参数详解

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--data` | `-d` | `uploads/example_data.csv` | 数据文件路径 |
| `--target` | `-t` | `target` | 目标变量列名 |
| `--task` | `-k` | `auto` | 任务类型：`classification`/`regression`/`auto` |
| `--instructions` | `-i` | 自动生成 | 自定义任务指令 |
| `--max_runtime` | `-r` | `60` | 最大训练时间（秒） |
| `--max_models` | `-m` | `10` | 最大模型数量 |

### 输出文件说明

运行完成后，系统会自动生成以下文件：

```
📁 输出目录结构
├── models/           # 训练好的模型文件
├── logs/            # 详细运行日志
└── results/         # 可视化结果
    ├── feature_importance_*.png  # 特征重要性图
    └── confusion_matrix_*.png    # 混淆矩阵图
```

## 📈 支持的模型类型

### 算法支持

| 算法类型 | Windows | macOS Intel | macOS Silicon | Linux |
|----------|---------|-------------|---------------|-------|
| GLM | ✅ | ✅ | ✅ | ✅ |
| Random Forest | ✅ | ✅ | ✅ | ✅ |
| GBM | ✅ | ✅ | ✅ | ✅ |
| XGBoost | ⚠️ | ✅ | ❌ | ✅ |
| Deep Learning | ✅ | ✅ | ✅ | ✅ |
| Stacked Ensemble | ✅ | ✅ | ✅ | ✅ |

**注释**：
- ✅ 完全支持
- ⚠️ 部分支持（Windows下XGBoost可能不可用，但不影响整体功能）
- ❌ 不支持

### 任务类型支持

- **二分类**: 逻辑回归、决策树等，输出概率和类别
- **多分类**: 支持多个类别的分类问题
- **回归**: 连续值预测，支持线性和非线性模型

## 💡 使用示例

### 示例1: 客户流失预测（分类）

```bash
python run_enhanced_demo.py \
  --data customer_data.csv \
  --target churn \
  --task classification \
  --max_runtime 180 \
  --instructions "构建客户流失预测模型，重点关注模型的召回率"
```

### 示例2: 房价预测（回归）

```bash
python run_enhanced_demo.py \
  --data housing_data.csv \
  --target price \
  --task regression \
  --max_runtime 300 \
  --instructions "预测房价，需要较高的预测精度"
```

### 示例3: 编程API调用

```python
from src.ml_agent.agent import H2OMLAgent
import pandas as pd

# 加载数据
data = pd.read_csv("your_data.csv")

# 初始化Agent
agent = H2OMLAgent(
    log=True, 
    log_path="logs/", 
    model_directory="models/"
)

# 执行建模任务
agent.invoke_agent(
    data_raw=data,
    user_instructions="这是一个分类问题，请构建高精度模型",
    target_variable="target_column",
    max_runtime_secs=300,
    max_models=20
)

# 获取结果
print(agent.get_workflow_summary(markdown=True))
leaderboard = agent.get_leaderboard()
print(leaderboard)
```

## 📁 项目结构

```
ml-agent/
├── README.md              # 项目说明文档
├── requirements.txt       # Python依赖
├── .env.example          # 环境变量模板
├── app.py                # Flask Web应用
├── run_enhanced_demo.py  # 增强版命令行工具
├── src/ml_agent/         # 核心代码库
│   ├── __init__.py
│   ├── agent.py          # 主控Agent
│   ├── llm_interface.py  # LLM接口
│   ├── data_preparer.py  # 数据预处理
│   ├── feature_engineer.py # 特征工程
│   └── utils.py          # 工具函数
├── static/               # Web静态资源
│   ├── css/
│   ├── js/
│   └── img/
├── templates/            # HTML模板
├── uploads/              # 示例数据
│   ├── example_data.csv      # 分类示例数据
│   └── example_regression_data.csv # 回归示例数据
├── data/                 # 用户数据目录
├── logs/                 # 运行日志
├── models/               # 模型保存目录
└── results/              # 可视化结果
```

## 🔧 高级配置

### 性能优化建议

**个人电脑配置**：
```bash
python run_enhanced_demo.py \
  --max_runtime 120 \
  --max_models 8 \
  --data your_data.csv
```

**服务器配置**：
```bash
python run_enhanced_demo.py \
  --max_runtime 1800 \
  --max_models 50 \
  --data your_data.csv
```

### 环境变量配置

在 `.env` 文件中可配置的选项：

```bash
# OpenAI API配置
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=your_api_key

# H2O配置（可选）
H2O_PORT=54321
H2O_MAX_MEM_SIZE=4g

# 日志级别
LOG_LEVEL=INFO
```

## 🐛 故障排除

### 常见问题

**Q1: Java未找到错误**
```bash
# 安装Java (Ubuntu/Debian)
sudo apt update
sudo apt install openjdk-11-jdk

# 验证安装
java -version
```

**Q2: XGBoost不可用**
```bash
# Windows用户
pip install xgboost==1.7.6

# macOS用户
brew install libomp
pip install xgboost
```

**Q3: 内存不足错误**
```bash
# 减少模型数量和运行时间
python run_enhanced_demo.py --max_models 5 --max_runtime 60
```

**Q4: API密钥错误**
- 检查 `.env` 文件是否正确配置
- 确认API密钥有效且有足够余额
- 检查网络连接

### 日志查看

```bash
# 查看最新日志
tail -f logs/ml_agent_*.log

# 查看错误日志
grep -i error logs/ml_agent_*.log
```

## 🤝 贡献指南

我们欢迎社区贡献！请参考以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如果存在

# 运行测试
python -m pytest tests/

# 代码格式化
black src/
isort src/
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [H2O.ai](https://www.h2o.ai/) 提供强大的AutoML引擎
- [OpenAI](https://openai.com/) 提供智能语言模型
- 所有为本项目做出贡献的开发者

---

## 📚 技术文档

> 以下为H2O AutoML的详细技术文档，供开发者和高级用户参考。

### H2O AutoML工作原理

ML-Agent整合了H2O AutoML的强大功能，实现了从数据处理到模型训练的全自动化流程：

```
原始数据 + 用户指令
    ↓
LLM智能分析与规划  
    ↓
自动数据预处理
    ↓
智能特征工程
    ↓
H2O AutoML模型训练
    ↓
模型评估与可视化
```

### 核心技术特性

1. **智能参数调优**: LLM根据数据特征自动配置H2O参数
2. **多模型并行训练**: 同时训练多种算法，自动选择最优模型
3. **交叉验证**: 确保模型泛化能力
4. **集成学习**: 自动构建Stacked Ensemble模型
5. **模型解释**: 提供特征重要性和模型解释

### 支持的评估指标

**分类任务**:
- AUC (Area Under Curve)
- Accuracy (准确率)  
- Precision (精确率)
- Recall (召回率)
- F1 Score

**回归任务**:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (决定系数)

---

> 💡 **提示**: 将机器学习的复杂性隐藏在智能界面之后，让每个人都能构建高质量的预测模型。

**Star ⭐ 这个项目如果它对您有帮助！**

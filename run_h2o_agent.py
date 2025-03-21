#!/usr/bin/env python
# coding: utf-8

# 基于h2o_machine_learning_agent.py创建的简化版本
# 使用我们已有的example_data.csv数据集

import pandas as pd
import h2o
import os
import time
from datetime import datetime

# 使用我们现有的ML Agent代码
from ml_agent import H2OMLAgent

# 设置路径
LOG_PATH = "logs/"
MODEL_PATH = "models/"

print("### 加载示例数据 ###")
# 加载示例数据
data = pd.read_csv("example_data.csv")
print(f"数据形状: {data.shape}")
print(data.head(3))

print("\n### 创建H2O ML Agent ###")
# 创建H2O ML Agent
agent = H2OMLAgent(
    log=True, 
    log_path=LOG_PATH,
    model_directory=MODEL_PATH
)

print("\n### 调用Agent进行自动化机器学习建模 ###")
# 调用Agent运行自动化机器学习
result = agent.invoke_agent(
    data_raw=data,
    user_instructions="请执行分类任务，使用最大运行时间30秒。将target列转换为分类变量。",
    target_variable="target"
)

print("\n### 模型训练完成 ###")
print(f"最佳模型ID: {agent.best_model_id}")
print(f"模型保存路径: {agent.model_path}")

print("\n### 排行榜 ###")
if agent.workflow_output and 'leaderboard' in agent.workflow_output:
    print(agent.workflow_output['leaderboard'].head(5))

print("\n### 完成 ###") 
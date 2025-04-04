#!/usr/bin/env python
# coding: utf-8

# 加载训练好的H2O模型并进行预测

import pandas as pd
import h2o
import os

# 初始化H2O
h2o.init()

# 加载示例数据
print("### 加载示例数据 ###")
data = pd.read_csv("example_data.csv")
print(f"数据形状: {data.shape}")
print(data.head(3))

# 获取最新训练的模型ID
print("\n### 加载模型 ###")
models_dir = "models/"
model_files = [f for f in os.listdir(models_dir) if not f.startswith('.')]
model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)

if model_files:
    best_model_path = os.path.join(models_dir, model_files[0])
    best_model_id = model_files[0].replace(".zip", "")
    print(f"使用最新模型: {best_model_id}")
    
    # 加载模型
    model = h2o.load_model(best_model_path)
    print("模型加载成功")
    
    # 转换数据为H2O帧
    h2o_data = h2o.H2OFrame(data)
    
    # 查看模型性能
    print("\n### 模型性能 ###")
    print(model)
    
    # 进行预测
    print("\n### 预测结果 ###")
    predictions = model.predict(h2o_data)
    print(predictions)
    
    # 合并数据和预测结果
    pred_df = predictions.as_data_frame()
    print("\n### 前5行预测结果 ###")
    print(pd.concat([data.head(5), pred_df.head(5)], axis=1))
else:
    print("没有找到训练好的模型文件。") 
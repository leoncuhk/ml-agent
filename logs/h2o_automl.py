# 免责声明：此函数由AI生成。使用前请仔细检查。
# Agent名称: h2o_ml_agent
# 创建时间: 2025-03-21 11:01:28

import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os
import logging

def h2o_automl(data_raw, model_directory, log_path):
    # 设置日志
    log_file = os.path.join(log_path, "h2o_automl.log") if os.path.isdir(log_path) else log_path
    logging.basicConfig(filename=log_file, level=logging.INFO)
    
    try:
        # 初始化H2O
        h2o.init()
        
        # 将数据转换为H2O Frame
        h2o_data = h2o.H2OFrame(data_raw)
        
        # 确定目标变量和特征
        target = 'target'
        features = h2o_data.columns
        features.remove(target)
        
        # 检查目标变量类型并转换
        unique_values = h2o_data[target].unique().nrows
        if unique_values == 2:  # 二分类问题
            # 将目标变量转换为分类变量
            h2o_data[target] = h2o_data[target].asfactor()
            problem_type = "classification"
        elif unique_values > 2 and unique_values <= 10:  # 多分类问题
            h2o_data[target] = h2o_data[target].asfactor()
            problem_type = "classification"
        else:  # 回归问题
            problem_type = "regression"
        
        # 设置AutoML参数
        aml = H2OAutoML(max_runtime_secs=30, seed=1)
        
        # 训练模型
        logging.info("Starting AutoML for %s task..." % problem_type)
        aml.train(x=features, y=target, training_frame=h2o_data)
        
        # 获取排行榜
        leaderboard = aml.leaderboard
        
        # 获取最佳模型的ID
        best_model_id = aml.leader.model_id
        
        # 保存最佳模型
        model_path = os.path.join(model_directory, best_model_id + ".zip")
        h2o.save_model(model=aml.leader, path=model_directory, force=True)
        
        logging.info("AutoML completed successfully.")
        
        return {
            'leaderboard': leaderboard.as_data_frame(),
            'best_model_id': best_model_id,
            'model_path': model_path
        }
    
    except Exception as e:
        logging.error("An error occurred: %s" % str(e))
        return None
    
    finally:
        # 关闭H2O
        h2o.shutdown(prompt=False)

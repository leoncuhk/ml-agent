import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

class MLflowToolsAgent:
    """MLflow工具代理，用于管理机器学习模型"""
    
    def __init__(self, tracking_uri="http://localhost:5000"):
        """
        初始化MLflow工具代理
        
        :param tracking_uri: MLflow跟踪服务URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        
    def list_experiments(self):
        """列出所有实验"""
        return self.client.list_experiments()
    
    def search_runs(self, experiment_ids, filter_string=""):
        """搜索实验运行"""
        return self.client.search_runs(experiment_ids, filter_string)
    
    def create_experiment(self, name, artifact_location=None):
        """创建新实验"""
        return self.client.create_experiment(name, artifact_location)
    
    def predict_from_run_id(self, run_id, data):
        """使用指定运行ID的模型进行预测"""
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        return model.predict(data)
    
    def launch_ui(self, port=5000):
        """启动MLflow UI"""
        import subprocess
        import os
        
        # 启动MLflow UI
        cmd = f"mlflow ui --port {port}"
        process = subprocess.Popen(cmd, shell=True)
        print(f"MLflow UI已启动，访问 http://localhost:{port}")
        return process
    
    def stop_ui(self, process=None, port=5000):
        """停止MLflow UI"""
        import subprocess
        import signal
        
        if process:
            process.terminate()
        else:
            # 查找使用指定端口的进程
            cmd = f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}'"
            pid = subprocess.check_output(cmd, shell=True).decode().strip()
            if pid:
                os.kill(int(pid), signal.SIGTERM)
        
        print(f"MLflow UI已停止")
    
    def list_registered_models(self):
        """列出所有注册模型"""
        return self.client.list_registered_models()
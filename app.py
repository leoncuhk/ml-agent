import pickle
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
import time
import h2o
from ml_agent import H2OMLAgent
from mlflow_agent import MLflowToolsAgent

app = Flask(__name__, template_folder='templates', static_folder='static')

# 全局变量
current_data = None
ml_agent = None
mlflow_agent = None
mlflow_ui_process = None

# 持久化状态文件路径
AGENT_STATE_PATH = 'agent_state.pkl'

def save_agent_state(agent):
    """保存代理状态到文件"""
    if agent is None:
        return False
    
    try:
        with open(AGENT_STATE_PATH, 'wb') as f:
            pickle.dump(agent, f)
        return True
    except Exception as e:
        print(f"保存代理状态错误: {str(e)}")
        return False

def load_agent_state():
    """从文件加载代理状态"""
    global ml_agent
    
    try:
        if os.path.exists(AGENT_STATE_PATH):
            with open(AGENT_STATE_PATH, 'rb') as f:
                ml_agent = pickle.load(f)
            return True
        return False
    except Exception as e:
        print(f"加载代理状态错误: {str(e)}")
        return False

@app.route('/')
def index():
    """首页"""
    # 尝试加载已保存的代理状态
    load_agent_state()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传数据文件"""
    global current_data
    
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file:
        # 保存文件
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)
        
        # 读取数据
        try:
            if file.filename.endswith('.csv'):
                current_data = pd.read_csv(file_path)
            elif file.filename.endswith(('.xls', '.xlsx')):
                current_data = pd.read_excel(file_path)
            else:
                return jsonify({'error': '不支持的文件格式'})
            
            # 返回数据预览
            preview = current_data.head(5).to_html()
            columns = current_data.columns.tolist()
            return jsonify({
                'success': True,
                'preview': preview,
                'columns': columns,
                'rows': len(current_data)
            })
        except Exception as e:
            return jsonify({'error': f'读取文件错误: {str(e)}'})

@app.route('/train', methods=['POST'])
def train_model():
    """训练模型"""
    global ml_agent
    
    if current_data is None:
        return jsonify({'error': '请先上传数据'})
    
    # 获取请求参数
    data = request.json
    target_var = data.get('target')
    instructions = data.get('instructions', '')
    
    if not target_var:
        return jsonify({'error': '请选择目标变量'})
    
    # 初始化H2O
    try:
        h2o.init()
    except:
        pass
    
    # 创建H2O ML Agent
    ml_agent = H2OMLAgent(
        log=True,
        log_path="logs/",
        model_directory="models/",
        enable_mlflow=True  # 启用MLflow跟踪
    )
    
    # 创建训练状态文件
    with open('training_status.txt', 'w') as f:
        f.write('started')
    
    # 开始训练
    try:
        # 后台任务执行训练
        def train_task():
            result = ml_agent.invoke_agent(
                data_raw=current_data,
                user_instructions=instructions,
                target_variable=target_var
            )
            # 训练完成后保存代理状态
            save_agent_state(ml_agent)
            # 更新训练状态
            with open('training_status.txt', 'w') as f:
                f.write('completed')
            return result
        
        # 异步执行
        from threading import Thread
        task = Thread(target=train_task)
        task.start()
        
        return jsonify({
            'success': True,
            'message': '模型训练已启动，请在"查看结果"页面查看进度'
        })
    except Exception as e:
        return jsonify({'error': f'模型训练错误: {str(e)}'})

@app.route('/training-status')
def training_status():
    """获取训练状态"""
    status_path = 'training_status.txt'
    
    if not os.path.exists(status_path):
        return jsonify({'status': 'not_started'})
    
    try:
        with open(status_path, 'r') as f:
            status = f.read().strip()
        
        # 读取最新日志行
        log_info = "正在训练中..."
        log_files = [f for f in os.listdir('logs') if f.endswith('.log')]
        if log_files:
            latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join('logs', x)))
            with open(os.path.join('logs', latest_log), 'r') as f:
                last_lines = f.readlines()[-5:]  # 获取最后5行
                log_info = ''.join(last_lines)
        
        return jsonify({
            'status': status,
            'log_info': log_info
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/results')
def get_results():
    """获取模型训练结果"""
    global ml_agent
    
    # 如果ml_agent为None，尝试加载保存的状态
    if ml_agent is None:
        if not load_agent_state():
            return jsonify({'success': False, 'error': '模型尚未训练'})
    
    try:
        # 获取结果
        summary = ml_agent.get_workflow_summary()
        
        # 获取排行榜
        leaderboard = None
        if ml_agent.workflow_output and 'leaderboard' in ml_agent.workflow_output:
            leaderboard = ml_agent.workflow_output['leaderboard'].to_html()
        
        return jsonify({
            'success': True,
            'best_model_id': ml_agent.best_model_id,
            'model_path': ml_agent.model_path,
            'leaderboard': leaderboard,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'获取结果错误: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    """使用模型进行预测"""
    global ml_agent
    
    if current_data is None:
        return jsonify({'error': '请先上传数据'})
    
    # 如果ml_agent为None，尝试加载保存的状态
    if ml_agent is None:
        if not load_agent_state():
            return jsonify({'error': '请先训练模型'})
    
    if ml_agent.best_model_id is None:
        return jsonify({'error': '请先训练模型'})
    
    try:
        # 加载模型
        model = h2o.load_model(ml_agent.model_path)
        
        # 转换数据为H2O帧
        h2o_data = h2o.H2OFrame(current_data)
        
        # 预测
        predictions = model.predict(h2o_data)
        pred_df = predictions.as_data_frame()
        
        # 合并结果
        result_df = pd.concat([current_data.head(10), pred_df.head(10)], axis=1)
        
        return jsonify({
            'success': True,
            'predictions': result_df.to_html()
        })
    except Exception as e:
        return jsonify({'error': f'预测错误: {str(e)}'})

@app.route('/launch-mlflow')
def launch_mlflow():
    """启动MLflow UI"""
    global mlflow_agent, mlflow_ui_process
    
    if mlflow_agent is None:
        mlflow_agent = MLflowToolsAgent()
    
    try:
        mlflow_ui_process = mlflow_agent.launch_ui(port=5000)
        return jsonify({
            'success': True,
            'message': 'MLflow UI已启动，请访问 http://localhost:5000'
        })
    except Exception as e:
        return jsonify({'error': f'启动MLflow UI错误: {str(e)}'})

@app.route('/stop-mlflow')
def stop_mlflow():
    """停止MLflow UI"""
    global mlflow_agent, mlflow_ui_process
    
    if mlflow_agent is None:
        return jsonify({'error': 'MLflow未初始化'})
    
    try:
        mlflow_agent.stop_ui(mlflow_ui_process)
        return jsonify({
            'success': True,
            'message': 'MLflow UI已停止'
        })
    except Exception as e:
        return jsonify({'error': f'停止MLflow UI错误: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, port=8000, use_reloader=False)
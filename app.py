import pickle
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
import time
import h2o
import sys
import traceback
import logging
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting in Flask
import matplotlib.pyplot as plt

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import the refactored agent and Celery task
from ml_agent.agent import H2OMLAgent
from celery.result import AsyncResult
from tasks import run_ml_agent_task # Import the Celery task

app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Global Variables & State Management --- 
current_data = None # Store uploaded data in memory (consider alternatives for large data)
ml_agent_instance = None # To hold loaded state for results/predict

AGENT_STATE_PATH = 'logs/agent_state.pkl' # Store state in logs
TRAINING_STATUS_PATH = 'logs/training_status.txt' # Simple status file (legacy, use Celery status)
UPLOAD_FOLDER = 'uploads'
IMG_FOLDER = os.path.join('static', 'img')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs(IMG_FOLDER, exist_ok=True)

# --- State Loading (Shows LAST successful run state) ---
def load_agent_state():
    global ml_agent_instance
    try:
        if os.path.exists(AGENT_STATE_PATH):
            with open(AGENT_STATE_PATH, 'rb') as f:
                saved_state = pickle.load(f)
            print(f"Agent state loaded from {AGENT_STATE_PATH}")
            config = saved_state.get('_agent_config', {})
            # Create a dummy agent to hold results 
            agent = H2OMLAgent(log=False, 
                               log_path=config.get('log_path', 'logs/'), 
                               model_directory=config.get('model_directory', 'models/'))
            agent.results = saved_state
            agent.results.pop('_agent_config', None) 
            ml_agent_instance = agent
            return True
        print("Agent state file not found.")
        return False
    except Exception as e:
        print(f"Error loading agent state: {str(e)}")
        ml_agent_instance = None
        return False

# --- Routes --- 
@app.route('/')
def index():
    load_agent_state()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            if filename.endswith('.csv'):
                current_data = pd.read_csv(file_path)
            elif filename.endswith(('.xls', '.xlsx')):
                current_data = pd.read_excel(file_path)
            else:
                return jsonify({'error': 'Unsupported file format (use CSV or Excel)'}), 400
                
            preview = current_data.head(5).to_html(classes='table table-striped', index=False)
            columns = current_data.columns.tolist()
            return jsonify({
                'success': True,
                'preview': preview,
                'columns': columns,
                'rows': len(current_data),
                'filename': filename
            })
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model_async(): # Renamed route
    global current_data
    
    if current_data is None:
        return jsonify({'error': 'Please upload data first'}), 400
    
    data = request.json
    target_var = data.get('target')
    instructions = data.get('instructions', 'Automatically build the best model.')
    max_runtime = data.get('max_runtime_secs', 60)
    max_models = data.get('max_models', None)
    sort_metric = data.get('sort_metric', None)

    if not target_var:
        return jsonify({'error': 'Target variable not selected'}), 400
    if target_var not in current_data.columns:
         return jsonify({'error': f'Target variable "{target_var}" not found in uploaded data'}), 400

    h2o_kwargs = {
        'max_runtime_secs': max_runtime,
        'max_models': max_models,
        'sort_metric': sort_metric,
    }
    h2o_kwargs = {k: v for k, v in h2o_kwargs.items() if v is not None}

    try:
        # Convert DataFrame to dict for serialization
        data_dict = current_data.to_dict(orient='split') 

        log_path = os.path.abspath("logs/")
        model_dir = os.path.abspath("models/")

        # Queue the Celery task
        task = run_ml_agent_task.delay(
            data_dict,
            instructions,
            target_var,
            h2o_kwargs,
            log_path,
            model_dir
        )
        
        app.logger.info(f"Queued training task with ID: {task.id}")
        # Clear previous status/state immediately (optional, task handles its own state)
        # try:
        #     if os.path.exists(TRAINING_STATUS_PATH): os.remove(TRAINING_STATUS_PATH)
        #     if os.path.exists(AGENT_STATE_PATH): os.remove(AGENT_STATE_PATH)
        #     with open(TRAINING_STATUS_PATH, 'w') as f: f.write(f'pending:{task.id}') # Use task ID
        # except Exception as e:
        #      app.logger.warning(f"Could not clear old status/state files: {e}")

        return jsonify({'success': True, 'task_id': task.id, 'message': f'Training task {task.id} queued.'})

    except Exception as e:
        app.logger.error(f"Failed to queue training task: {e}", exc_info=True)
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

@app.route('/task-status/<task_id>')
def task_status(task_id):
    task_result = AsyncResult(task_id)
    status = task_result.status
    result_data = None
    error_info = None

    if status == 'SUCCESS':
        result_data = task_result.get()
    elif status == 'FAILURE':
        try:
            task_result.get() # Re-raise exception to get info
        except Exception as e:
             error_info = str(e) 
        task_return_value = task_result.result 
        if isinstance(task_return_value, dict) and task_return_value.get('error'):
             error_info = task_return_value.get('error')
        elif isinstance(task_return_value, Exception):
            error_info = str(task_return_value)
    elif status == 'PENDING':
        # Check the simple status file if it exists (provides intermediate info)
         try:
             with open(TRAINING_STATUS_PATH, 'r') as f:
                 file_status = f.read().strip()
                 if file_status.startswith('running') or file_status.startswith('error'):
                      status = file_status # Show more specific status if available
         except FileNotFoundError:
             pass # Keep PENDING status
         except Exception as e:
             app.logger.warning(f"Error reading status file: {e}")
             
    # Fetch recent log lines (consider doing this only when status is PENDING/STARTED)
    log_info = "(Log polling not implemented in this version)"
    # ... (optional: add logic to read last few lines from agent log file) ...

    return jsonify({
        'task_id': task_id,
        'status': status, 
        'result': result_data, # Contains task return dict on SUCCESS
        'error_info': error_info, # Contains exception info on FAILURE
        'log_info': log_info 
    })

@app.route('/results')
def get_results():
    # This route now primarily shows the state of the *last successful run*
    # as saved by the task via pickle. It's disconnected from specific task IDs.
    global ml_agent_instance
    if ml_agent_instance is None:
        if not load_agent_state():
             return jsonify({'success': False, 'error': 'No previous successful run state found.'})
    
    if ml_agent_instance is None:
        return jsonify({'success': False, 'error': 'Agent state is unavailable.'}) # Should not happen

    try:
        summary = ml_agent_instance.get_workflow_summary(markdown=True)
        leaderboard_df = ml_agent_instance.get_leaderboard()
        leaderboard_html = None
        if leaderboard_df is not None:
            leaderboard_html = leaderboard_df.to_html(classes='table table-striped table-hover', index=False)
        elif ml_agent_instance.results.get('error'):
             leaderboard_html = f"<p>Leaderboard not available. Error during execution: {ml_agent_instance.results['error']}</p>"
        else:
            leaderboard_html = "<p>Leaderboard is empty or unavailable.</p>"
            
        best_model_id = ml_agent_instance.results.get('best_model_id')
        model_path = ml_agent_instance.results.get('best_model_path')
        execution_error = ml_agent_instance.results.get('error')
        mlflow_run_id = ml_agent_instance.results.get('mlflow_run_id')

        return jsonify({
            'success': True,
            'best_model_id': best_model_id,
            'model_path': model_path,
            'leaderboard': leaderboard_html,
            'summary': summary,
            'error': execution_error,
            'mlflow_run_id': mlflow_run_id
        })
    except Exception as e:
        app.logger.error(f"Error retrieving results from state: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Error retrieving results: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global ml_agent_instance
    global current_data
    if current_data is None:
        return jsonify({'error': 'Please upload data first for prediction.'}), 400
    if ml_agent_instance is None:
        if not load_agent_state():
            return jsonify({'error': 'Model state not found. Please train a model first.'}), 400
    model_path = ml_agent_instance.results.get('best_model_path')
    if not model_path or not os.path.exists(model_path):
        error_msg = ml_agent_instance.results.get('error', 'Model path not found or model does not exist.')
        return jsonify({'error': f'Cannot predict: {error_msg}'}), 400
    try:
        app.logger.info(f"Loading model for prediction from: {model_path}")
        h2o.init()
        model = h2o.load_model(model_path)
        app.logger.info("Model loaded successfully.")
        app.logger.info("Converting prediction data to H2O Frame...")
        h2o_data = h2o.H2OFrame(current_data)
        app.logger.info("H2O Frame created.")
        app.logger.info("Performing predictions...")
        predictions = model.predict(h2o_data)
        pred_df = predictions.as_data_frame()
        app.logger.info("Predictions complete.")
        pred_df = pred_df.head(10)
        orig_data_head = current_data.head(10)
        pred_df.columns = [f"pred_{col}" for col in pred_df.columns]
        result_df = pd.concat([orig_data_head.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        return jsonify({
            'success': True,
            'predictions': result_df.to_html(classes='table table-striped table-hover', index=False)
        })
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/interpret', methods=['POST'])
def interpret_model():
    global ml_agent_instance
    if ml_agent_instance is None:
        if not load_agent_state():
            return jsonify({'error': 'Model state not found.'}), 400
    model_path = ml_agent_instance.results.get('best_model_path')
    if not model_path or not os.path.exists(model_path):
        error_msg = ml_agent_instance.results.get('error', 'Model path not found or model does not exist.')
        return jsonify({'error': f'Cannot interpret model: {error_msg}'}), 400
    interpretation_results = {}
    try:
        app.logger.info(f"Loading model for interpretation from: {model_path}")
        h2o.init()
        model = h2o.load_model(model_path)
        app.logger.info("Model loaded successfully for interpretation.")
        try:
            varimp_plot_path = os.path.join(IMG_FOLDER, 'variable_importance.png')
            plt.figure()
            model.varimp_plot(server=False, save_plot_path=varimp_plot_path)
            plt.close()
            interpretation_results['varimp_plot'] = f'/static/img/variable_importance.png?t={time.time()}'
            app.logger.info(f"Variable importance plot saved to {varimp_plot_path}")
        except Exception as plot_e:
             app.logger.warning(f"Could not generate variable importance plot: {plot_e}")
             interpretation_results['varimp_error'] = str(plot_e)
        return jsonify({
            'success': True,
            'interpretations': interpretation_results
        })
    except Exception as e:
        app.logger.error(f"Error during model interpretation: {str(e)}", exc_info=True)
        return jsonify({'error': f'Interpretation failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Use Flask's logger
    app.logger.setLevel(logging.INFO) 
    # Ensure compatibility with waitress or other prod servers
    port = int(os.environ.get("PORT", 8000))
    # Use debug=False for production, True for development
    # use_reloader=False is often needed with Celery workers
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
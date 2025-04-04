# tasks.py
from celery_app import celery
import os
import sys
import pandas as pd
import pickle
import logging

# Ensure src is in path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ml_agent.agent import H2OMLAgent

# Use the logger configured by the agent/utils
logger = logging.getLogger('ml_agent')

# Paths for state/status (relative to project root where worker runs)
AGENT_STATE_PATH = 'logs/agent_state.pkl' 
TRAINING_STATUS_PATH = 'logs/training_status.txt'

# --- State saving needs to be handled carefully in Celery ---
# Simple pickle saving might not be ideal if tasks run on different machines
# For now, we'll keep the logic but acknowledge its limitations.
def save_task_state(agent_results, log_path, model_dir):
    try:
        state_to_save = agent_results
        state_to_save['_agent_config'] = {
            'log_path': log_path,
            'model_directory': model_dir
        }
        with open(AGENT_STATE_PATH, 'wb') as f:
            pickle.dump(state_to_save, f)
        logger.info(f"Task state saved to {AGENT_STATE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error saving task state: {str(e)}")
        return False

@celery.task(bind=True)
def run_ml_agent_task(self, data_dict, instructions, target_variable, h2o_kwargs, log_path, model_dir):
    """Celery task to run the H2O ML Agent."""
    task_id = self.request.id
    logger.info(f"Starting Celery task {task_id}...")
    
    # Update status file
    try:
        with open(TRAINING_STATUS_PATH, 'w') as f:
            f.write(f'running:{task_id}') # Include task ID in status
    except Exception as e:
         logger.error(f"Failed to write initial status for task {task_id}: {e}")
         # Continue execution anyway?

    final_status = 'error: Unknown task failure'
    results = {'error': 'Task initialization failed'} # Default result in case of early exit

    try:
        # Re-create DataFrame from dictionary
        # Note: Passing large dataframes directly isn't recommended for Celery
        # Consider passing a path or using shared storage for larger data.
        data_raw = pd.DataFrame.from_dict(data_dict)
        logger.info(f"Task {task_id}: Data reconstructed, shape {data_raw.shape}")
        
        # Initialize agent within the task
        # The agent itself isn't passed, avoiding serialization issues
        agent = H2OMLAgent(log=True, log_path=log_path, model_directory=model_dir)
        logger.info(f"Task {task_id}: Agent initialized.")

        # Invoke the agent
        results = agent.invoke_agent(
            data_raw=data_raw,
            user_instructions=instructions,
            target_variable=target_variable,
            **h2o_kwargs
        )
        logger.info(f"Task {task_id}: Agent invocation finished.")

        execution_error = results.get('error')
        if execution_error:
             logger.error(f"Task {task_id} finished with error: {execution_error}")
             final_status = f"error: {execution_error}"
        else:
             logger.info(f"Task {task_id} completed successfully.")
             final_status = 'completed'
             # Save state only on success
             save_task_state(results, log_path, model_dir)

    except Exception as e:
        logger.error(f"Exception in task {task_id}: {e}", exc_info=True)
        final_status = f"error: Task failed - {e}"
        results['error'] = final_status # Ensure error is in results
        
    finally:
        # Update status file
        try:
            with open(TRAINING_STATUS_PATH, 'w') as f:
                f.write(final_status) 
            logger.info(f"Task {task_id}: Final status written: {final_status}")
        except Exception as e:
             logger.error(f"Failed to write final status for task {task_id}: {e}")

    # Celery task should return serializable results
    # Convert DataFrame leaderboard to JSON if present
    if 'leaderboard' in results and isinstance(results['leaderboard'], pd.DataFrame):
        results['leaderboard'] = results['leaderboard'].to_json(orient='split')
        
    # Convert data description (potentially long string) if needed? Maybe omit from direct return
    results.pop('data_description', None) 

    return results # Return results dictionary 
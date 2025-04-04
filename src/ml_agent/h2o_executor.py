# src/ml_agent/h2o_executor.py
import h2o
import pandas as pd
import logging
import traceback
import os # Needed for save_model path
import mlflow
import mlflow.h2o

logger = logging.getLogger('ml_agent') # Use the agent's logger

def run_h2o_automl(data_raw: pd.DataFrame, target_variable: str, model_directory: str, h2o_params: dict):
    """Runs H2O AutoML with the given data and parameters."""
    logger.info("--- Starting H2O AutoML Execution with MLflow Tracking ---")
    results = {"leaderboard": None, "best_model_id": None, "best_model_path": None, "error": None, "mlflow_run_id": None}
    
    # Start MLflow Run
    try:
        # Use context manager for robust run termination
        with mlflow.start_run() as run:
            results["mlflow_run_id"] = run.info.run_id
            logger.info(f"Started MLflow Run ID: {results['mlflow_run_id']}")
            
            # Log effective H2O parameters
            logger.info("Logging H2O parameters to MLflow...")
            # Convert list values to strings for MLflow compatibility if needed
            params_to_log = {k: str(v) if isinstance(v, list) else v for k, v in h2o_params.items()}
            mlflow.log_params(params_to_log)
            logger.info(f"Logged parameters: {params_to_log}")
            
            # H2O Initialization and Training
            try:
                logger.info("Initializing H2O cluster...")
                h2o.init()
                logger.info("H2O initialized.")
                logger.info("Converting data to H2O Frame...")
                h2o_df = h2o.H2OFrame(data_raw)
                logger.info("H2O Frame created.")
                
                y = target_variable
                x = h2o_df.columns
                x.remove(y)
                logger.info(f"Target variable (y): {y}")
                logger.info(f"Predictor variables (x): {x}")
                
                # Determine problem type and prepare target column
                problem_type = None
                target_col = h2o_df[y]
                num_unique = target_col.nlevels()[0] if target_col.isfactor()[0] else target_col.unique().nrows
                is_numeric = target_col.isnumeric()[0]
                is_factor = target_col.isfactor()[0]
                target_type_str = target_col.types[y]
                logger.info(f"Target '{y}' info: unique/levels={num_unique}, numeric={is_numeric}, factor={is_factor}, type={target_type_str}")
                
                if num_unique <= 10 and target_type_str != 'real': 
                    problem_type = "classification"
                    if not is_factor:
                        logger.info(f"Target '{y}' converting to factor.")
                        h2o_df[y] = target_col.asfactor()
                    else:
                        logger.info(f"Target '{y}' already factor.")
                elif is_numeric:
                    problem_type = "regression"
                    logger.info(f"Target '{y}' is numeric.")
                else:
                    logger.warning(f"Could not reliably determine problem type for target '{y}'.")
                    if is_factor:
                        problem_type = "classification"
                        logger.warning(f"Target '{y}' defaulting to classification.")
                        
                current_sort_metric = h2o_params.get('sort_metric')
                if problem_type == "regression" and current_sort_metric and current_sort_metric.upper() in ['AUC', 'LOGLOSS', 'AUCPR']:
                    logger.warning(f"Removing incompatible sort metric {current_sort_metric} for regression.")
                    h2o_params.pop('sort_metric', None)
                elif problem_type == "classification" and current_sort_metric and current_sort_metric.upper() in ['RMSE', 'MAE', 'RMSLE']:
                    logger.warning(f"Sort metric {current_sort_metric} unusual for classification.")

                logger.info(f"Starting H2O AutoML for: {problem_type}")
                logger.info(f"H2O Params: {h2o_params}")
                
                aml = h2o.automl.H2OAutoML(**h2o_params)
                aml.train(x=x, y=y, training_frame=h2o_df)
                logger.info("H2O AutoML training complete.")
                
                logger.info("Fetching AutoML leaderboard...")
                lb = aml.leaderboard
                if lb is not None and lb.nrows > 0:
                    results["leaderboard"] = lb.as_data_frame()
                    logger.debug(f"Leaderboard:\n{results['leaderboard']}")
                    if aml.leader:
                        leader_model = aml.leader
                        results["best_model_id"] = leader_model.model_id
                        logger.info(f"Best model ID: {results['best_model_id']}")
                        
                        # Log Leader Model Metrics to MLflow
                        try:
                            perf = leader_model.model_performance(train=True) # Use training metrics for simplicity
                            metrics_to_log = {}
                            if problem_type == 'classification':
                                metrics_to_log['auc'] = perf.auc()
                                metrics_to_log['logloss'] = perf.logloss()
                                metrics_to_log['mean_per_class_error'] = perf.mean_per_class_error()
                            else: # Regression
                                metrics_to_log['rmse'] = perf.rmse()
                                metrics_to_log['mae'] = perf.mae()
                                metrics_to_log['r2'] = perf.r2()
                                
                            logger.info(f"Logging metrics to MLflow: {metrics_to_log}")
                            mlflow.log_metrics(metrics_to_log)
                        except Exception as metric_e:
                             logger.error(f"Failed to log metrics to MLflow: {metric_e}")
                             
                        # Log Leader Model Artifact to MLflow
                        try:
                            logger.info("Logging H2O model artifact to MLflow...")
                            mlflow.h2o.log_model(leader_model, artifact_path="h2o-model")
                            logger.info("Model logged to MLflow artifact path: h2o-model")
                        except Exception as log_model_e:
                             logger.error(f"Failed to log model artifact to MLflow: {log_model_e}")
                             
                        # Save model locally as well (redundant if using MLflow registry, but good backup)
                        try:
                            os.makedirs(model_directory, exist_ok=True) 
                            model_path = h2o.save_model(model=leader_model, path=model_directory, force=True)
                            results["best_model_path"] = model_path
                            logger.info(f"Best model saved locally to: {model_path}")
                        except Exception as save_e:
                            logger.error(f"Failed to save the best model locally: {save_e}")
                            # Don't overwrite MLflow error if logging succeeded
                            if not results.get("error"):
                                 results["error"] = f"Training complete but failed to save model locally: {save_e}"
                    else:
                        logger.warning("Leaderboard found, but no leader model.")
                else:
                    logger.warning("H2O AutoML finished, but the leaderboard is empty.")
                    results["error"] = "AutoML finished with an empty leaderboard."
                    mlflow.log_metric("run_failed", 1) # Log failure
                    
            except Exception as e:
                logger.error(f"An error occurred during H2O AutoML execution: {e}")
                logger.error(traceback.format_exc())
                results["error"] = str(e)
                mlflow.log_metric("run_failed", 1) # Log failure
                mlflow.log_param("error_message", str(e)[:250]) # Log truncated error

    except Exception as mlflow_e:
         logger.error(f"Failed to start or manage MLflow run: {mlflow_e}")
         # Store MLflow error if H2O didn't have one already
         if not results.get("error"):
             results["error"] = f"MLflow setup failed: {mlflow_e}"

    logger.info("--- H2O AutoML Execution Finished (with MLflow) ---")
    return results 
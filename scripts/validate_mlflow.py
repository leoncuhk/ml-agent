import mlflow
import os

def validate_mlflow_logging():
    """
    Validates MLflow logging by starting a run, logging a parameter and a metric,
    and printing the run ID. Assumes MLflow tracking server is running or
    will use local file storage (mlruns directory).
    """
    print("Attempting to validate MLflow logging...")

    try:
        # MLflow will automatically use a local ./mlruns directory if no tracking URI is set.
        # You can optionally set a tracking URI:
        # mlflow.set_tracking_uri("http://127.0.0.1:5000") # Example for local server

        # Start an MLflow run. Use nested with statement for safety.
        print("Starting MLflow run...")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            print(f"MLflow run started successfully. Run ID: {run_id}")

            # Log a parameter
            param_name = "validation_param"
            param_value = "test_value"
            print(f"Logging parameter: {param_name}={param_value}")
            mlflow.log_param(param_name, param_value)

            # Log a metric
            metric_name = "validation_metric"
            metric_value = 0.95
            print(f"Logging metric: {metric_name}={metric_value}")
            mlflow.log_metric(metric_name, metric_value)

            # Log an artifact (optional - example: create a dummy file)
            artifact_path = "validation_artifact.txt"
            print(f"Logging artifact: {artifact_path}")
            with open(artifact_path, "w") as f:
                f.write("This is a validation artifact.")
            mlflow.log_artifact(artifact_path)
            os.remove(artifact_path) # Clean up the dummy file

        print("MLflow run completed and artifacts logged.")

        # Optionally, try fetching the run to confirm
        # client = mlflow.tracking.MlflowClient()
        # fetched_run = client.get_run(run_id)
        # print(f"\nSuccessfully fetched run {run_id} from tracking server.")
        # print(f"Parameters: {fetched_run.data.params}")
        # print(f"Metrics: {fetched_run.data.metrics}")

        return True

    except Exception as e:
        print(f"An error occurred during MLflow validation: {e}")
        print("Ensure the MLflow tracking server is running or accessible,")
        print("or that you have write permissions for the local 'mlruns' directory.")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("------------------------------------------")
    print("MLflow Validation Script")
    print("------------------------------------------")
    print("This script will attempt to log a test run to MLflow.")
    print("You can view the results in the MLflow UI.")
    print("If you haven't already, run 'mlflow ui' in your terminal")
    print("in the project directory and open the provided URL in your browser.")
    print("------------------------------------------\n")

    if validate_mlflow_logging():
        print("\nMLflow Logging Validation Passed.")
    else:
        print("\nMLflow Logging Validation Failed.") 
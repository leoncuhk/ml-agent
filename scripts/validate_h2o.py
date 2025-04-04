import h2o
import pandas as pd
from h2o.estimators import H2OGeneralizedLinearEstimator

def validate_h2o_setup(data_path="example_data.csv"):
    """
    Validates the H2O setup by initializing H2O, loading data,
    training a simple GLM model, and printing the model summary.
    """
    print("Attempting to validate H2O setup...")

    try:
        # Initialize H2O cluster
        print("Initializing H2O cluster...")
        h2o.init()
        print("H2O cluster initialized successfully.")

        # Load data using pandas first for robustness
        print(f"Loading data from {data_path} using pandas...")
        try:
            pd_df = pd.read_csv(data_path)
            print("Pandas DataFrame loaded successfully.")
            # Select a subset of columns for simplicity if needed
            # pd_df = pd_df[['col1', 'col2', 'target']] # Adjust column names
        except FileNotFoundError:
            print(f"Error: Data file not found at {data_path}")
            return False
        except Exception as e:
            print(f"Error loading data with pandas: {e}")
            return False

        # Convert pandas DataFrame to H2O Frame
        print("Converting pandas DataFrame to H2O Frame...")
        try:
            h2o_df = h2o.H2OFrame(pd_df)
            print("H2O Frame created successfully.")
            print("Data preview:")
            print(h2o_df.head())
        except Exception as e:
            print(f"Error converting to H2O Frame: {e}")
            return False

        # --- Model Training (Example - Adjust based on example_data.csv) ---
        print("\nAttempting to train a simple H2O GLM model...")
        try:
            # Identify predictors and response variable
            # !! IMPORTANT: Adjust 'target_column' and predictor columns based on your actual example_data.csv !!
            response_column = "target" # Replace with the actual target column name
            if response_column not in h2o_df.columns:
                 print(f"Error: Target column '{response_column}' not found in the H2O Frame.")
                 print(f"Available columns: {h2o_df.columns}")
                 print("Please update the 'response_column' variable in validate_h2o.py.")
                 # Try to infer target if it's the last column
                 potential_target = h2o_df.columns[-1]
                 print(f"Attempting to use the last column '{potential_target}' as target.")
                 response_column = potential_target
                 #return False # Option to exit if target is not explicitly found

            predictors = [col for col in h2o_df.columns if col != response_column]
            if not predictors:
                print("Error: No predictor columns found after excluding the target.")
                return False

            print(f"Using '{response_column}' as the target variable.")
            print(f"Using predictors: {predictors}")

            # Define and train the GLM model
            # Infer family based on target type
            if h2o_df[response_column].isnumeric():
                family = "gaussian" # For regression
                print("Target appears numeric. Setting GLM family to Gaussian (regression).")
            else:
                h2o_df[response_column] = h2o_df[response_column].asfactor()
                family = "binomial" # For binary classification (adjust if multiclass)
                print("Target appears categorical. Converting to factor and setting GLM family to Binomial (classification).")


            glm_model = H2OGeneralizedLinearEstimator(family=family, seed=1234)
            print("Training GLM model...")
            glm_model.train(x=predictors, y=response_column, training_frame=h2o_df)
            print("GLM model training complete.")

            # Print model summary
            print("\nModel Summary:")
            print(glm_model)
            # For classification, show confusion matrix
            if family != "gaussian":
                print("\nModel Performance (Training Data):")
                print(glm_model.model_performance(train=True))

        except Exception as e:
            print(f"Error during H2O model training or evaluation: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback
            return False

        print("\nH2O setup validation appears successful.")
        return True

    except Exception as e:
        print(f"An unexpected error occurred during H2O validation: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Shutdown H2O cluster (optional, depends on workflow)
        # print("Shutting down H2O cluster...")
        # h2o.cluster().shutdown(prompt=False)
        pass

if __name__ == "__main__":
    if validate_h2o_setup():
        print("\nH2O Validation Passed.")
    else:
        print("\nH2O Validation Failed.") 
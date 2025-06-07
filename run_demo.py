#!/usr/bin/env python3
"""
Enhanced ML-Agent Demo Script

This script demonstrates the capabilities of the ML-Agent with support for:
- Command-line arguments for data path, target variable, and instructions
- Both classification and regression problems
- Enhanced visualizations and model evaluation

Usage:
    python run_enhanced_demo.py --data uploads/example_data.csv --target target --task classification
    python run_enhanced_demo.py --data uploads/example_regression_data.csv --target price --task regression
"""

import pandas as pd
from src.ml_agent.agent import H2OMLAgent
import warnings
import os
import platform
import h2o
import sys
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Define constants
MODEL_DIR = "models/"
LOG_DIR = "logs/"
RESULTS_DIR = "results/"

# Setup logger for error handling
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_output_directories():
    """Create necessary output directories."""
    for directory in [MODEL_DIR, LOG_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)

def setup_native_xgboost_on_apple_silicon():
    """
    On Apple Silicon, XGBoost is not currently supported by H2O.
    This function just provides information about the limitation.
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("--- [XGBoost Support Information for Apple Silicon] ---")
        try:
            import xgboost
            xgboost_path = os.path.dirname(xgboost.__file__)
            native_lib_path = os.path.join(xgboost_path, 'lib', 'libxgboost.dylib')

            if os.path.exists(native_lib_path):
                print(f"✅ Native arm64 XGBoost library found at: {native_lib_path}")
                print("ℹ️  Note: H2O currently does not support XGBoost on Apple Silicon (ARM64).")
                print("ℹ️  XGBoost will be skipped during H2O AutoML training, but other algorithms will work.")
            else:
                print(f"⚠️ Native XGBoost library not found at expected path: {native_lib_path}")
        except ImportError:
            print("⚠️ xgboost package is not installed.")
        except Exception as e:
            print(f"ℹ️ XGBoost package detected but H2O integration not available on Apple Silicon: {e}")
        finally:
            print("--------------------------------------------------\n")

def create_feature_importance_plot(results, output_dir):
    """Create and save feature importance visualization."""
    if 'feature_importance' in results and results['feature_importance']:
        fi_df = pd.DataFrame(results['feature_importance'])
        
        plt.figure(figsize=(10, 6))
        plt.barh(fi_df['variable'], fi_df['relative_importance'])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance plot saved to: {plot_path}")
        return plot_path
    return None

def create_confusion_matrix_plot(results, output_dir):
    """Create and save confusion matrix visualization."""
    if 'confusion_matrix' in results and results['confusion_matrix']:
        cm_df = pd.DataFrame(results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df.set_index(cm_df.columns[0]).iloc[:, 1:].astype(float), 
                    annot=True, fmt='.0f', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix plot saved to: {plot_path}")
        return plot_path
    return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced ML-Agent Demo')
    parser.add_argument('--data', '-d', type=str, 
                        default='uploads/example_data.csv',
                        help='Path to the CSV data file')
    parser.add_argument('--target', '-t', type=str, 
                        default='target',
                        help='Name of the target variable column')
    parser.add_argument('--task', '-k', type=str, 
                        choices=['classification', 'regression', 'auto'],
                        default='auto',
                        help='Type of machine learning task')
    parser.add_argument('--instructions', '-i', type=str,
                        default=None,
                        help='Custom instructions for the ML agent')
    parser.add_argument('--max_runtime', '-r', type=int,
                        default=60,
                        help='Maximum runtime in seconds for AutoML')
    parser.add_argument('--max_models', '-m', type=int,
                        default=10,
                        help='Maximum number of models to train')
    return parser.parse_args()

def generate_instructions(task_type, target_variable, max_runtime):
    """Generate appropriate instructions based on task type."""
    if task_type == 'classification':
        return f"这是一个分类问题，请为我构建一个分类模型来预测'{target_variable}'。在建模前请分析并处理好数据。我希望模型的运行时间不超过{max_runtime}秒。"
    elif task_type == 'regression':
        return f"这是一个回归问题，请为我构建一个回归模型来预测'{target_variable}'。在建模前请分析并处理好数据。我希望模型的运行时间不超过{max_runtime}秒。"
    else:  # auto
        return f"请分析数据并自动确定这是分类还是回归问题，然后构建适当的模型来预测'{target_variable}'。在建模前请分析并处理好数据。我希望模型的运行时间不超过{max_runtime}秒。"

def run_enhanced_demo(args):
    """
    Enhanced demo function with configurable parameters.
    """
    print("=" * 60)
    print("🚀 Enhanced ML-Agent Demo")
    print("=" * 60)
    print(f"📊 Dataset: {args.data}")
    print(f"🎯 Target: {args.target}")
    print(f"📋 Task: {args.task}")
    print(f"⏱️  Max Runtime: {args.max_runtime}s")
    print(f"🔢 Max Models: {args.max_models}")
    
    # Setup directories
    setup_output_directories()
    
    # 1. Load and validate data
    print(f"\n[1/6] 🔍 Loading and validating data: '{args.data}'...")
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file not found at '{args.data}'.")
        return
        
    data = pd.read_csv(args.data)
    print(f"✅ Data loaded successfully. Shape: {data.shape}")
    
    if args.target not in data.columns:
        print(f"❌ Error: Target variable '{args.target}' not found in dataset.")
        print(f"Available columns: {list(data.columns)}")
        return
    
    print(f"📈 Data preview:")
    print(data.head())
    print(f"📊 Target variable '{args.target}' summary:")
    print(data[args.target].describe())

    # 2. Initialize ML-Agent
    print(f"\n[2/6] 🤖 Initializing H2O ML Agent...")
    agent = H2OMLAgent(log=True, log_path=LOG_DIR, model_directory=MODEL_DIR)
    print("✅ Agent initialized successfully.")

    # 3. Generate instructions
    instructions = args.instructions or generate_instructions(args.task, args.target, args.max_runtime)
    print(f"\n[3/6] 📝 Generated instructions:")
    print(f"'{instructions}'")
    
    print(f"\n[4/6] ⚙️  Executing AutoML workflow...")
    print("⏳ This may take some time, H2O AutoML is training and evaluating models in the background...")
    
    # 4. Execute ML workflow
    agent.invoke_agent(
        data,
        user_instructions=instructions,
        target_variable=args.target,
        max_runtime_secs=args.max_runtime,
        max_models=args.max_models
    )
    print("✅ Agent execution completed.")

    # 5. Display results
    print(f"\n[5/6] 📋 Retrieving and displaying results...")
    
    # Workflow summary
    print("\n" + "="*60)
    print(" " * 20 + "🔄 Workflow Summary")
    print("="*60)
    summary = agent.get_workflow_summary(markdown=True)
    print(summary)

    # Model leaderboard
    leaderboard = agent.get_leaderboard()
    if leaderboard is not None and not leaderboard.empty:
        print("\n" + "="*60)
        print(" " * 20 + "🏆 Model Leaderboard")
        print("="*60)
        print(leaderboard.to_string())
    else:
        print("\n⚠️  Could not retrieve model leaderboard.")

    # Enhanced evaluation metrics
    final_results = agent.get_results()

    # Confusion Matrix (for classification)
    if 'confusion_matrix' in final_results and final_results['confusion_matrix']:
        print("\n" + "="*60)
        print(" " * 18 + "📊 Confusion Matrix")
        print("="*60)
        cm_df = pd.DataFrame(final_results['confusion_matrix'])
        print(cm_df.to_string())

    # Feature Importance
    if 'feature_importance' in final_results and final_results['feature_importance']:
        print("\n" + "="*60)
        print(" " * 17 + "⭐ Feature Importance")
        print("="*60)
        fi_df = pd.DataFrame(final_results['feature_importance'])
        print(fi_df.to_string(index=False))

    # 6. Create visualizations
    print(f"\n[6/6] 📊 Creating visualizations...")
    
    # Create plots
    fi_plot = create_feature_importance_plot(final_results, RESULTS_DIR)
    cm_plot = create_confusion_matrix_plot(final_results, RESULTS_DIR)
    
    # Final summary
    print("\n" + "="*60)
    print(" " * 22 + "🎉 Demo Completed")
    print("="*60)
    print(f"📁 Models saved to: '{MODEL_DIR}'")
    print(f"📜 Logs saved to: '{LOG_DIR}'")
    print(f"📊 Results saved to: '{RESULTS_DIR}'")
    if fi_plot:
        print(f"📈 Feature importance plot: '{fi_plot}'")
    if cm_plot:
        print(f"🔥 Confusion matrix plot: '{cm_plot}'")
    print("="*60)

def main():
    """Main function."""
    # Run XGBoost setup check
    setup_native_xgboost_on_apple_silicon()

    # Suppress H2O warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Parse arguments
    args = parse_arguments()
    
    try:
        run_enhanced_demo(args)
        if h2o.cluster():
            h2o.cluster().shutdown()
    except Exception as e:
        logger.error(f"An error occurred during the demo run: {e}", exc_info=True)
        # Ensure H2O cluster is shut down even if an error occurs
        if h2o.cluster():
            h2o.cluster().shutdown()
        sys.exit(1)

if __name__ == "__main__":
    main() 
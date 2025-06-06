import pandas as pd
from src.ml_agent.agent import H2OMLAgent
import warnings
import os
import platform
import h2o

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

# Run the setup before H2O is initialized
setup_native_xgboost_on_apple_silicon()

# 忽略H2O产生的一些繁琐的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def run_main_demo():
    """
    一个最小化的演示脚本，用于运行并展示ML-Agent的核心功能。
    """
    print("--- [ML-Agent Demo] ---")
    print("目标: 使用 'uploads/example_data.csv' 数据集跑通一个完整的自动化建模流程。")
    
    # 1. 定义并加载数据
    data_path = "uploads/example_data.csv"
    target_variable = "target"
    
    print(f"\n[1/5] 正在加载数据: '{data_path}'...")
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件未找到 at '{data_path}'。演示中断。")
        return
        
    data = pd.read_csv(data_path)
    print(f"✅ 数据加载成功. 数据形状: {data.shape}")
    print(f"🎯 预测目标 (Target Variable): '{target_variable}'")

    # 2. 初始化 ML-Agent
    print("\n[2/5] 正在初始化 H2O ML Agent...")
    # 日志和模型将保存在各自的目录中
    agent = H2OMLAgent(log=True, log_path="logs/", model_directory="models/")
    print("✅ Agent 初始化成功.")

    # 3. 定义用户指令并调用Agent
    user_instructions = "这是一个二分类问题，请为我构建一个分类模型。在建模前请分析并处理好数据。我希望模型的运行时间不超过60秒。"
    
    print("\n[3/5] 正在调用 Agent 执行 AutoML 任务...")
    print('💬 用户指令: "' + user_instructions + '"' + " " * 100 + "⏳ 这可能需要一点时间，H2O AutoML 正在后台进行模型训练与评估...")
    
    agent.invoke_agent(
        data,
        user_instructions=user_instructions,
        target_variable=target_variable,
        max_runtime_secs=60,  # 限制运行时间，用于快速演示
        max_models=10          # 限制生成的模型数量
    )
    print("✅ Agent 执行完成。")

    # 4. 获取并展示结果
    print("\n[4/5] 正在获取任务结果...")
    
    # 获取工作流的文字总结
    summary = agent.get_workflow_summary(markdown=True)
    print("\n---------- [ 工作流总结 ] ----------")
    print(summary)

    # 获取H2O生成的模型性能排行榜
    leaderboard = agent.get_leaderboard()
    print("\n---------- [ H2O 模型排行榜 ] ----------")
    if leaderboard is not None and not leaderboard.empty:
        # 使用 to_string() 以获得更好的控制台打印效果
        print(leaderboard.to_string())
    else:
        print("未能获取模型排行榜。")
        
    # 5. 结束
    print("\n[5/5] 演示成功结束。")
    print("您可以检查 'models/' 目录查看保存的最佳模型，以及在 'logs/' 目录中查看详细的运行日志。")
    print("----------------------------")

if __name__ == "__main__":
    run_main_demo() 
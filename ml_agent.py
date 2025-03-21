import pandas as pd
import h2o
import os
import time
import random
from datetime import datetime
import json
from openai import OpenAI
import logging

# LLM调用函数，参考yunwu.py
def chat_with_llm(prompt, max_retries=3):
    """
    与LLM模型对话，包含重试机制
    :param prompt: 用户输入
    :param max_retries: 最大重试次数
    :return: 模型回复
    """
    # 创建客户端
    client = OpenAI(
        # 可以根据需要替换为其他API端点和密钥
        base_url="https://yunwu.ai/v1",
        api_key="sk-kZpsgjS8XplmWbO0VO4RBPBujvHpl30erAXestY8CmbLygel"
    )

    for attempt in range(max_retries):
        try:
            # 添加随机延迟，避免并发问题
            if attempt > 0:
                delay = random.uniform(1, 3)
                print(f"等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
            
            print(f"正在发送请求... (尝试 {attempt + 1}/{max_retries})")
            
            response = client.chat.completions.create(
                # model="deepseek-v3",
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=0.7,
                max_tokens=2000
            )
            
            # 检查是否有错误信息
            if hasattr(response, 'error'):
                error_msg = response.error.get('message', '未知错误')
                if error_msg == 'concurrency exceeded':
                    print(f"并发超限，将重试...")
                    continue
                return f"API错误: {error_msg}"
            
            # 尝试获取响应内容
            if hasattr(response, 'choices') and response.choices:
                if hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
            
            print(f"响应格式异常: {response}")
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"发生错误: {str(e)}")
                continue
            return f"所有重试都失败了: {str(e)}"
    
    return "达到最大重试次数，仍未获得有效响应"

class MLAgent:
    """
    机器学习自动化建模的AI Agent
    """
    def __init__(self, log=True, log_path="logs/", model_directory="models/", enable_mlflow=False):
        """
        初始化机器学习Agent
        
        :param log: 是否记录日志
        :param log_path: 日志路径
        :param model_directory: 模型保存路径
        :param enable_mlflow: 是否启用MLflow跟踪
        """
        self.log = log
        self.log_path = log_path
        self.model_directory = model_directory
        self.enable_mlflow = enable_mlflow
        
        # 创建必要的目录
        if self.log and not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        
        # 存储模型信息
        self.function_path = None
        self.function_name = None
        self.best_model_id = None
        self.model_path = None
        self.error = None
        self.recommended_steps = None
        self.workflow_output = None
        
        # 初始化H2O（如果可用）
        try:
            h2o.init()
        except:
            print("H2O初始化失败，将在后续使用时再次尝试初始化")
    
    def invoke_agent(self, data_raw, user_instructions, target_variable):
        """
        调用AI Agent进行自动化机器学习建模
        
        :param data_raw: 原始数据，Pandas DataFrame
        :param user_instructions: 用户指令，字符串
        :param target_variable: 目标变量名，字符串
        :return: 模型输出结果
        """
        print("---ML AGENT----")
        print("    * 推荐机器学习步骤")
        
        # 1. 获取推荐的机器学习步骤
        self.recommended_steps = self._get_ml_steps_from_llm(data_raw, user_instructions, target_variable)
        
        # 2. 生成代码
        print("    * 创建自动机器学习代码")
        code = self._generate_code_from_llm(data_raw, user_instructions, target_variable)
        
        # 3. 保存代码到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.function_path = f"{self.log_path}/automl_{timestamp}.py"
        self.function_name = "automl_function"
        
        with open(self.function_path, "w") as f:
            f.write(code)
        
        print(f"      文件已保存到: {self.function_path}")
        
        # 4. 执行代码
        print("    * 执行Agent代码")
        try:
            # 动态加载生成的模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("automl_module", self.function_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 获取函数引用
            automl_func = getattr(module, self.function_name)
            
            # 执行函数
            result = automl_func(data_raw, self.model_directory, self.log_path)
            
            # 存储结果
            if isinstance(result, dict):
                if "best_model_id" in result:
                    self.best_model_id = result["best_model_id"]
                if "model_path" in result:
                    self.model_path = result["model_path"]
                self.workflow_output = result
            
        except Exception as e:
            self.error = str(e)
            print(f"执行代码时发生错误: {self.error}")
        
        # 5. 输出结果
        print("    * 报告Agent输出")
        return self.get_workflow_summary()
    
    def _get_ml_steps_from_llm(self, data_raw, user_instructions, target_variable):
        """从LLM获取推荐的机器学习步骤"""
        # 准备数据描述
        data_description = self._get_data_description(data_raw)
        
        # 构建提示
        prompt = f"""
        作为一个机器学习专家，我需要你推荐一系列步骤来解决以下机器学习问题。
        
        ## 数据信息:
        {data_description}
        
        ## 目标变量:
        {target_variable}
        
        ## 用户指令:
        {user_instructions}
        
        请提供一个详细的机器学习流程建议，包括数据预处理、特征工程、模型选择和评估步骤。
        请给出清晰的步骤，并解释每一步的重要性。
        """
        
        # 调用LLM获取回复
        response = chat_with_llm(prompt)
        return response
    
    def _generate_code_from_llm(self, data_raw, user_instructions, target_variable):
        """从LLM生成执行代码"""
        # 准备数据描述
        data_description = self._get_data_description(data_raw)
        
        # 构建提示
        prompt = f"""
        作为一个机器学习工程师，我需要你生成Python代码来实现自动化机器学习建模。
        
        ## 数据信息:
        {data_description}
        
        ## 目标变量:
        {target_variable}
        
        ## 用户指令:
        {user_instructions}
        
        请生成完整的Python函数代码，要求：
        1. 函数名应为 'automl_function'
        2. 函数应接受参数：data_raw（pandas DataFrame），model_directory（模型保存路径），log_path（日志路径）
        3. 使用h2o的AutoML功能进行自动化建模
        4. 函数应返回一个字典，包含：leaderboard（排行榜），best_model_id（最佳模型ID），model_path（模型保存路径）
        5. 代码应处理异常并提供适当的日志
        6. 代码应考虑用户指令中的特定要求
        
        只需返回完整的Python代码，不需要额外的解释。
        """
        
        # 调用LLM获取回复
        response = chat_with_llm(prompt)
        
        # 格式化代码，添加必要的头部信息
        code = f"""# 自动生成的机器学习代码
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{response}
"""
        return code
    
    def _get_data_description(self, data_raw):
        """获取数据的描述信息"""
        description = f"""
        数据集包含 {data_raw.shape[0]} 行和 {data_raw.shape[1]} 列。
        列名: {', '.join(data_raw.columns)}
        
        数据类型:
        {data_raw.dtypes.to_string()}
        
        数据预览:
        {data_raw.head(3).to_string()}
        
        基本统计:
        {data_raw.describe().to_string()}
        """
        return description
    
    def get_recommended_ml_steps(self, markdown=False):
        """获取推荐的机器学习步骤"""
        if not self.recommended_steps:
            return "尚未生成推荐步骤"
        
        if markdown:
            from IPython.display import Markdown
            return Markdown(f"# 推荐的机器学习步骤:\n\n{self.recommended_steps}")
        
        return self.recommended_steps
    
    def get_workflow_summary(self, markdown=False):
        """获取工作流程摘要"""
        leaderboard_str = ""
        if hasattr(self, 'workflow_output') and self.workflow_output and 'leaderboard' in self.workflow_output:
            leaderboard = self.workflow_output['leaderboard']
            # 将排行榜转换为字符串
            if isinstance(leaderboard, pd.DataFrame):
                leaderboard_str = f"\n\n## ---模型排行榜----\n```\n{leaderboard.head(10).to_string()}\n```"
        
        summary = f"""
        # 机器学习Agent输出摘要
        
        ## ---推荐步骤----
        {self.recommended_steps}
        
        ## ---自动机器学习函数----
        ```python
        {self._get_function_content()}
        ```
        
        ## ---函数路径----
        ```python
        {self.function_path}
        ```
        
        ## ---函数名称----
        ```python
        {self.function_name}
        ```
        
        ## ---执行错误----
        {self.error if self.error else "无"}
        
        ## ---模型路径----
        {self.model_path if self.model_path else "无"}
        
        ## ---最佳模型ID----
        {self.best_model_id if self.best_model_id else "无"}
        {leaderboard_str}
        """
        
        if markdown:
            from IPython.display import Markdown
            return Markdown(summary)
        
        return summary
    
    def get_log_summary(self, markdown=False):
        """获取日志摘要"""
        summary = f"""
        ## 机器学习Agent日志摘要:
        
        函数路径: {self.function_path}
        
        函数名称: {self.function_name}
        
        最佳模型ID: {self.best_model_id}
        
        模型路径: {self.model_path}
                """
        
        if markdown:
            from IPython.display import Markdown
            return Markdown(summary)
            
        return summary
    
    def _get_function_content(self):
        """获取函数内容"""
        if not self.function_path:
            return "尚未生成函数"
            
        try:
            with open(self.function_path, "r") as f:
                return f.read()
        except:
            return "无法读取函数内容"

# 添加一个更专门化的H2O ML Agent类
class H2OMLAgent(MLAgent):
    """
    专门使用H2O进行自动化机器学习的Agent
    """
    def __init__(self, model=None, log=True, log_path="logs/", model_directory="models/", enable_mlflow=False):
        """
        初始化H2O机器学习Agent
        
        :param model: LLM模型（未使用，为了与原始接口兼容）
        :param log: 是否记录日志
        :param log_path: 日志路径
        :param model_directory: 模型保存路径
        :param enable_mlflow: 是否启用MLflow跟踪
        """
        super().__init__(log, log_path, model_directory, enable_mlflow)
        self.function_name = "h2o_automl"
    
    def invoke_agent(self, data_raw, user_instructions, target_variable):
        """
        调用H2O AutoML Agent进行自动化机器学习建模
        
        :param data_raw: 原始数据，Pandas DataFrame
        :param user_instructions: 用户指令，字符串
        :param target_variable: 目标变量名，字符串
        :return: 模型输出结果
        """
        print("---H2O ML AGENT----")
        print("    * 推荐机器学习步骤")
        
        # 1. 获取推荐的机器学习步骤
        self.recommended_steps = self._get_ml_steps_from_llm(data_raw, user_instructions, target_variable)
        
        # 2. 使用预定义的代码，不通过LLM生成
        print("    * 使用内置H2O AutoML代码")
        code = """# 免责声明：此函数由AI生成。使用前请仔细检查。
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
"""
        
        # 3. 保存代码到文件
        self.function_path = f"{self.log_path}/h2o_automl.py"
        
        with open(self.function_path, "w") as f:
            f.write(code)
        
        print(f"      文件已保存到: {self.function_path}")
        
        # 4. 执行代码
        print("    * 执行Agent代码")
        try:
            # 动态加载生成的模块
            import importlib.util
            spec = importlib.util.spec_from_file_location("h2o_automl_module", self.function_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 获取函数引用
            automl_func = getattr(module, self.function_name)
            
            # 执行函数
            result = automl_func(data_raw, self.model_directory, self.log_path)
            
            # 存储结果
            if isinstance(result, dict):
                if "best_model_id" in result:
                    self.best_model_id = result["best_model_id"]
                if "model_path" in result:
                    self.model_path = result["model_path"]
                self.workflow_output = result
            
        except Exception as e:
            self.error = str(e)
            print(f"执行代码时发生错误: {self.error}")
        
        # 5. 输出结果
        print("    * 报告Agent输出")
        return self.get_workflow_summary()

# 示例用法
if __name__ == "__main__":
    # 加载示例数据
    try:
        # 尝试加载一个示例数据集
        data = pd.read_csv("example_data.csv")
        
        # 创建H2O ML Agent
        agent = H2OMLAgent(log=True)
        
        # 调用Agent
        agent.invoke_agent(
            data_raw=data,
            user_instructions="请执行分类任务，使用最大运行时间30秒。",
            target_variable="target"  # 替换为实际的目标变量
        )
        
        # 获取摘要
        print(agent.get_workflow_summary())
        
    except Exception as e:
        print(f"运行示例时发生错误: {str(e)}")
        print("请确保您有可用的示例数据，或修改主程序以使用您自己的数据。") 
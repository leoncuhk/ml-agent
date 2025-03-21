from openai import OpenAI
import time
import random

# 创建客户端
client = OpenAI(
    # base_url="https://api.lkeap.cloud.tencent.com/v1",
    # api_key="sk-oSZAOu7tC2wHsev4RFtcFqw7LuMF5dC2Sc5TUOmFk9s8B5d2"

    base_url="https://yunwu.ai/v1",
    api_key="sk-kZpsgjS8XplmWbO0VO4RBPBujvHpl30erAXestY8CmbLygel"
)

def chat_with_llm(prompt, max_retries=3):
    """
    与 LLM 模型对话，包含重试机制
    :param prompt: 用户输入
    :param max_retries: 最大重试次数
    :return: 模型回复
    """
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

def main():
    print("LLM聊天程序已启动 (输入'退出'结束)")
    print("使用模型: ")
    print("已启用重试机制，最大重试3次")
    
    while True:
        user_input = input("\n你: ")
        
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("再见!")
            break
        
        response = chat_with_llm(user_input)
        print("\nAI: ", response)

if __name__ == "__main__":
    main()

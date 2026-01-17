from zhipuai import ZhipuAI

# 替换为你的智谱AI API Key
API_KEY = "0a3238e655184f7e9aa9360671979cc6.uMhPGKLm6GEyjtHm"

def test_zhipu_api():
    try:
        # 初始化智谱AI客户端
        client = ZhipuAI(api_key=API_KEY)
        
        # 调用最简单的聊天接口（测试连通性）
        response = client.chat.completions.create(
            model="glm-4.5-flash",
            messages=[{"role": "user", "content": "请回复“测试成功”"}]
        )
        
        # 打印响应结果
        print("API调用成功，返回结果：")
        print(response.choices[0].message.content)
        return True
    
    except Exception as e:
        print(f"API调用失败，错误原因：{str(e)}")
        return False

if __name__ == "__main__":
    test_zhipu_api()
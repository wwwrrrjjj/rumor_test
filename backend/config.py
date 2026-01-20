# backend/config_final.py
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "123456a66",
    "database": "rumor_detection"
}

SECRET_KEY = "0a3238e655184f7e9aa9360671979cc6.uMhPGKLm6GEyjtHm"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

LLM_CONFIG = {
    "model_name": "glm-4",
    "api_key": "0a3238e655184f7e9aa9360671979cc6.uMhPGKLm6GEyjtHm",
    "temperature": 0.3,
    "max_tokens": 10240,
    "base_url": "https://open.bigmodel.cn/api/coding/paas/v4"
}

LLM_FAKE = False

# 搜索配置 - 更新为使用DuckDuckGo
SEARCH_CONFIG = {
    "enable": True,
    "provider": "duckduckgo",  # 主要搜索引擎
    "max_results": 3,  # 每次搜索最大结果数
    "max_queries": 2,  # 最多搜索查询数
    "timeout": 15,  # 搜索超时时间
    "cooldown": 0.5,  # 搜索间隔时间
}
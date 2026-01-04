# backend/config.py
# 数据库配置（修改为你的MySQL账号密码）
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",       # 你的MySQL用户名（默认root）
    "password": "123456a66", # 你的MySQL密码
    "database": "rumor_detection" # 第一步创建的数据库名
}

# Token加密配置（生产环境要改复杂，开发环境先用这个）
SECRET_KEY = "my-secret-key-1234567890-abcdefg"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300  # Token有效期5小时

# 大语言模型临时配置（先模拟返回，后续替换真实模型）
LLM_FAKE = True  # 先开模拟，避免依赖API密钥
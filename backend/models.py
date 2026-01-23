# backend/models.py
from sqlalchemy import Column, Integer, String, Text, Boolean, DECIMAL, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import hashlib
from config import DB_CONFIG

# 1. 初始化基础类
Base = declarative_base()

# 计算文本内容哈希值的函数
def calculate_content_hash(content: str) -> str:
    """计算文本内容的MD5哈希值，用于去重"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# 2. 用户表
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    create_time = Column(DateTime, default=datetime.now)

# 3. 推理记录表
class ReasoningRecord(Base):
    __tablename__ = "reasoning_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)  # 确保有索引
    content = Column(Text, nullable=False)
    content_hash = Column(String(32), nullable=False, index=True)
    type = Column(String(50), nullable=False)
    rumor_prob = Column(DECIMAL(5,4), nullable=False)
    is_ai_generated = Column(Boolean, default=False)
    reasoning_steps = Column(Text, nullable=False)
    keywords = Column(Text, nullable=False)
    conclusion = Column(Text, default="")  # 新增：存储最终结论
    use_count = Column(Integer, default=1)
    create_time = Column(DateTime, default=datetime.now)
    last_used_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)

# 4. 登录日志表
class LoginLog(Base):
    __tablename__ = "login_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    login_time = Column(DateTime, default=datetime.now)
    ip = Column(String(50), default="")

# 5. 数据库连接 + 自动建表
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 自动创建所有表
Base.metadata.create_all(bind=engine)

print("数据库表创建/更新成功！")
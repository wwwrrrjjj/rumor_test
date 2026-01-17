# backend/models.py
from sqlalchemy import Column, Integer, String, Text, Boolean, DECIMAL, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
# 导入数据库配置（原文件有这部分，必须保留）
from config import DB_CONFIG

# 1. 初始化基础类（你提供的部分，保留）
Base = declarative_base()

# 2. 用户表（你提供的部分，字段和原文件一致，保留）
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(100), nullable=False)
    create_time = Column(DateTime, default=datetime.now)

# 3. 推理记录表（核心修改：新增keywords字段，你的版本正确）
class ReasoningRecord(Base):
    __tablename__ = "reasoning_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    type = Column(String(50), nullable=False)
    rumor_prob = Column(DECIMAL(5,4), nullable=False)
    is_ai_generated = Column(Boolean, default=False)
    reasoning_steps = Column(Text, nullable=False)
    # ---------------------- 新增：关键字字段 ----------------------
    keywords = Column(Text, nullable=False)  # 存储JSON字符串格式的关键字列表
    create_time = Column(DateTime, default=datetime.now)

# 4. 登录日志表（原文件有这部分，建议保留，不影响核心功能）
class LoginLog(Base):
    __tablename__ = "login_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    login_time = Column(DateTime, default=datetime.now)
    ip = Column(String(50), default="")

# 5. 数据库连接 + 自动建表（原文件核心逻辑，必须保留）
# 拼接数据库连接URL
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
# 创建引擎（echo=False 避免打印冗余SQL日志）
engine = create_engine(DATABASE_URL, echo=False)
# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# 自动创建所有表（首次运行时执行，后续运行不会重复创建）
Base.metadata.create_all(bind=engine)

# 打印提示（可选，确认建表成功）
print("数据库表创建/更新成功！")
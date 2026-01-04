from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, DECIMAL, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config import DB_CONFIG

# 1. 连接MySQL数据库
DATABASE_URL = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
engine = create_engine(DATABASE_URL, echo=False)  # echo=False不打印SQL日志
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
# 用户表（存储登录账号密码）
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)  # 用户名
    password = Column(String(100), nullable=False)              # 存储加密后的密码
    create_time = Column(DateTime, default=datetime.now)

# 推理记录表（存储用户检测记录）
class ReasoningRecord(Base):
    __tablename__ = "reasoning_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)  # 关联用户
    content = Column(Text, nullable=False)                            # 检测的文本
    type = Column(String(50), nullable=False)                          # 谣言类型
    rumor_prob = Column(DECIMAL(5,4), nullable=False)                  # 谣言概率（0-1）
    is_ai_generated = Column(Boolean, default=False)                   # 是否AI生成
    reasoning_steps = Column(Text, nullable=False)                     # 推理步骤（JSON字符串）
    create_time = Column(DateTime, default=datetime.now)

# 登录日志表（记录登录行为）
class LoginLog(Base):
    __tablename__ = "login_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    login_time = Column(DateTime, default=datetime.now)
    ip = Column(String(50), default="")

# 3. 自动创建表（首次运行时执行）
Base.metadata.create_all(bind=engine)
print("数据库表创建成功！")
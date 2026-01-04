from fastapi import FastAPI, Depends, HTTPException, Request,Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from datetime import datetime, timedelta    
from jose import jwt, JWTError
from passlib.context import CryptContext
import json
import random

# 导入自己的配置和模型
import config
from models import SessionLocal, User, ReasoningRecord, LoginLog

# ---------------------- 基础配置 ----------------------
app = FastAPI(title="谣言甄别系统API")

# 跨域配置（开发环境允许所有前端访问，生产环境要限制）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 允许所有域名
    allow_credentials=True,
    allow_methods=["*"],        # 允许所有请求方法（GET/POST）
    allow_headers=["*"],        # 允许所有请求头
)

# 密码加密工具（避免明文存储密码）
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------- 工具函数 ----------------------
# 1. 获取数据库连接（每次请求自动创建/关闭）
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. 密码加密/验证
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_pwd: str, hashed_pwd: str) -> bool:
    return pwd_context.verify(plain_pwd, hashed_pwd)

# 3. 生成/验证Token（登录后给前端返回Token，后续请求带Token验证身份）
def create_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    token_data = {"sub": str(user_id), "exp": expire}
    return jwt.encode(token_data, config.SECRET_KEY, algorithm=config.ALGORITHM)

def verify_token(token: str) -> int:
    try:
        payload = jwt.decode(token, config.SECRET_KEY, algorithms=[config.ALGORITHM])
        user_id = int(payload.get("sub"))
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Token无效/过期")

# 4. 模拟大语言模型检测（先不用真实API，避免依赖）
def fake_llm_detect(content: str, type: str):
    # 随机生成谣言概率（0-1）
    rumor_prob = round(random.uniform(0, 1), 4)
    # 模拟推理步骤
    reasoning_steps = [
        f"事件主干：{content[:20]}...（类型：{type}）",
        f"事实验证：{'符合客观事实' if rumor_prob < 0.5 else '不符合客观事实'}",
        f"逻辑判断：{'非谣言' if rumor_prob < 0.5 else '谣言'}，AI生成概率：{round(random.uniform(0, 1), 2)}"
    ]
    return {
        "rumor_prob": rumor_prob,
        "is_ai_generated": random.choice([True, False]),
        "reasoning_steps": reasoning_steps
    }

# ---------------------- 数据模型（请求/返回格式） ----------------------
# 登录请求格式（前端传过来的参数）
class LoginRequest(BaseModel):
    username: str
    password: str

# 检测请求格式
class DetectRequest(BaseModel):
    content: str  # 检测的文本
    type: str     # 谣言类型（疫情/食品安全等）

# ---------------------- 核心接口 ----------------------
# 1. 登录接口（POST /api/login）
@app.post("/api/login")
def login(request: LoginRequest, req: Request, db: Session = Depends(get_db)):
    # 1. 校验用户名密码是否为空
    if not request.username or not request.password:
        raise HTTPException(status_code=400, detail="用户名/密码不能为空")
    
    # 2. 查询用户是否存在
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户名不存在")
    
    # 3. 验证密码是否正确
    if not verify_password(request.password, user.password):
        raise HTTPException(status_code=401, detail="密码错误")
    
    # 4. 记录登录日志（IP+时间）
    login_log = LoginLog(user_id=user.id, ip=req.client.host)
    db.add(login_log)
    db.commit()
    
    # 5. 生成Token返回给前端
    token = create_token(user.id)
    return {
        "code": 200,
        "msg": "登录成功",
        "data": {
            "token": token,
            "user_id": user.id,
            "username": user.username
        }
    }
# 2. 谣言检测接口（POST /api/detect）
@app.post("/api/detect")
def detect(
    request: DetectRequest,
    authorization: str = Header(None),  # 从Authorization头取Token
    db: Session = Depends(get_db)
):
    # 1. 验证Authorization头是否存在
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    # 2. 提取Token（去掉Bearer前缀）
    token = authorization.split(" ")[1]
    # 3. 验证Token
    try:
        user_id = verify_token(token)
    except:
        raise HTTPException(status_code=401, detail="Token无效/过期")
    
    # 下面的代码（校验文本长度、调用模型等）保持不变
    # 2. 校验文本长度
    if len(request.content) < 1 or len(request.content) > 500:
        raise HTTPException(status_code=400, detail="文本长度需1-500字")
    
    # 3. 调用模型检测（先模拟）
    llm_result = fake_llm_detect(request.content, request.type)
    
    # 4. 存储检测记录到数据库
    record = ReasoningRecord(
        user_id=user_id,
        content=request.content,
        type=request.type,
        rumor_prob=llm_result["rumor_prob"],
        is_ai_generated=llm_result["is_ai_generated"],
        reasoning_steps=json.dumps(llm_result["reasoning_steps"], ensure_ascii=False)
    )
    db.add(record)
    db.commit()
    
    # 5. 返回结果给前端
    return {
        "code": 200,
        "msg": "检测成功",
        "data": {
            "rumor_prob": llm_result["rumor_prob"],
            "is_ai_generated": llm_result["is_ai_generated"],
            "reasoning_steps": llm_result["reasoning_steps"],
            "record_id": record.id
        }
    } 
# 3. 历史记录查询接口（GET /api/history）
@app.get("/api/history")
def get_history(
    authorization: str = Header(None),  # 从Authorization头取Token
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    # 1. 验证Authorization头是否存在
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    # 2. 提取Token（去掉Bearer前缀）
    token = authorization.split(" ")[1]
    # 3. 验证Token
    try:
        user_id = verify_token(token)
    except:
        raise HTTPException(status_code=401, detail="Token无效/过期")
    # 2. 分页查询历史记录
    offset = (page - 1) * size
    records = db.query(ReasoningRecord).filter(ReasoningRecord.user_id == user_id).order_by(ReasoningRecord.create_time.desc()).offset(offset).limit(size).all()
    
    # 3. 格式化结果
    history_list = []
    for r in records:
        history_list.append({
            "record_id": r.id,
            "content": r.content,
            "type": r.type,
            "rumor_prob": float(r.rumor_prob),
            "is_ai_generated": r.is_ai_generated,
            "create_time": r.create_time.strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # 4. 返回总数+列表
    total = db.query(ReasoningRecord).filter(ReasoningRecord.user_id == user_id).count()
    return {
        "code": 200,
        "msg": "查询成功",
        "data": {
            "total": total,
            "page": page,
            "size": size,
            "list": history_list
        }
    }
# ---------------------- 启动后端 ----------------------
if __name__ == "__main__":
    db = SessionLocal()
    try:
        
        if not db.query(User).filter(User.username == "test").first():
            # 密码截断到72字节，避免bcrypt长度报错
            password = str("123456")[:72]
            test_user = User(username="test", password=hash_password(password))
            db.add(test_user)
            db.commit()
            print("测试用户创建成功：用户名test，密码123456")
        else:
            print("测试用户已存在，无需重复创建")
    except Exception as e:
        # 捕获创建用户时的异常
        print(f"创建测试用户失败：{str(e)}")
        db.rollback()  # 出错回滚，避免数据库事务异常
    finally:
        # 关闭数据库连接
        db.close()
    
    # 启动服务（端口8000）
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session, sessionmaker 
from datetime import datetime, timedelta    
from jose import jwt, JWTError
from passlib.context import CryptContext
from zhipuai import ZhipuAI
import simplejson as json
import random

# 导入自己的配置和模型
import config
from models import Base, SessionLocal, User, ReasoningRecord, LoginLog, create_engine

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

# 初始化数据库
from models import SessionLocal  # models.py初始化

# 密码加密工具（避免明文存储密码）
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------- 初始化大语言模型客户端 ----------------------
llm_client = ZhipuAI(api_key=config.LLM_CONFIG["api_key"])

# ---------------------- 大语言模型提示词模板 ----------------------
# 修复后：所有普通 {} 转义为 {{}}，无格式失衡，紧凑单行示例减少解析干扰
PROMPT_TEMPLATE = """
你的任务是作为谣言甄别专家，分析输入文本，**严格遵守以下所有要求返回结果，不得添加任何额外内容、注释、标点或格式说明**：

1.  关键字拆分：提取文本核心关键字，仅保留有实际意义的词汇，返回非空列表格式；
2.  推理步骤：严格按「识别内容→检查事实→评估合理性→得出结论」4步流程输出，每一步表述简洁明了，返回非空列表格式，必须包含4个步骤，不可多不可少；
3.  是否AI生成：仅返回布尔值（true 或 false，小写，无引号），不得返回其他任何文字；
4.  谣言概率：返回0-1之间的小数，严格保留4位小数，不得返回百分比或其他格式；
5.  输出格式强制要求：
    -  必须是完整闭合的标准JSON格式，无任何多余内容（无```json标记、无前置说明、无后置补充）；
    -  仅包含且必须包含以下4个字段：keywords、reasoning_steps、is_ai_generated、rumor_prob；
    -  JSON字段名必须用双引号包裹，列表元素必须用双引号包裹，键值对之间用英文逗号分隔，结尾必须以}}闭合；
    -  禁止使用单引号、禁止换行、禁止遗漏逗号、禁止字段值为null或空列表（keywords和reasoning_steps至少包含1个元素）；
6.  错误兜底：若无法判断或信息不足，按以下默认值返回：
    -  keywords：包含输入文本核心词汇的列表；
    -  reasoning_steps：按4步流程填充「信息不足，无法完成精准判断」相关内容；
    -  is_ai_generated：false；
    -  rumor_prob：0.5000；

=== 输入信息 ===
输入文本：{content}
文本类型：{type}

=== 输出要求（再次强调）===
1.  直接返回JSON，无任何前置、后置附加内容；
2.  严格匹配示例格式，字段不可缺失、格式不可偏差；
3.  小数保留4位，布尔值小写，列表非空，JSON完整闭合。

=== 输出示例（仅可复制此格式，替换对应内容，禁止修改格式结构）===
{{"keywords": ["星期八"], "reasoning_steps": ["识别文本内容：今天是星期八", "检查事实：星期只有七天，星期八不存在", "评估合理性：该陈述与客观事实相悖", "得出结论：属于不实信息"], "is_ai_generated": false, "rumor_prob": 0.8500}}
"""

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
    # 模拟推理步骤（确保4步）
    reasoning_steps = [
        f"识别内容：{content[:20]}...（类型：{type}）",
        f"检查事实：{'符合客观事实' if rumor_prob < 0.5 else '不符合客观事实'}",
        f"评估合理性：{'非谣言' if rumor_prob < 0.5 else '谣言'}，AI生成概率：{round(random.uniform(0, 1), 2)}",
        f"得出结论：{'判定为非谣言' if rumor_prob < 0.5 else '判定为谣言'}"
    ]
    # 模拟关键字
    keywords = [content[:8] + "..."] if len(content) > 0 else ["未知"]
    return {
        "keywords": keywords,
        "rumor_prob": rumor_prob,
        "is_ai_generated": random.choice([True, False]),
        "reasoning_steps": reasoning_steps
    }

# 5. 真实大语言模型检测函数
def real_llm_detect(content: str, type: str):
    try:
        # 修复：转义内容中的 { 和 }，避免 format() 语法冲突
        escaped_content = content.replace("{", "{{").replace("}", "}}")
        escaped_type = type.replace("{", "{{").replace("}", "}}")
        # 用转义后的内容进行格式化
        prompt_content = PROMPT_TEMPLATE.format(content=escaped_content, type=escaped_type)
        response = llm_client.chat.completions.create(
            model=config.LLM_CONFIG["model_name"],
            messages=[{"role": "user", "content": prompt_content}],
            temperature=config.LLM_CONFIG["temperature"],
            max_tokens=config.LLM_CONFIG["max_tokens"]
        )
        result_str = response.choices[0].message.content.strip()
        print("模型原始返回：", result_str)
        
        # 增强容错：去除markdown代码块（模型可能返回```json包裹的内容）
        if result_str.startswith("```json"):
            result_str = result_str.replace("```json", "").replace("```", "").strip()
        # 多次尝试解析
        result = None
        for _ in range(2):
            try:
                result = json.loads(result_str)
                break
            except:
                # 修复常见格式问题：单引号→双引号、缺失逗号等
                result_str = result_str.replace("'", "\"").replace("\n", "").strip()
                if not result_str.endswith("}"):
                    result_str += "}"
        if result is None:
            raise Exception("JSON解析失败")
        
        # 补全缺失字段/修正格式
        required_fields = ["keywords", "reasoning_steps", "is_ai_generated", "rumor_prob"]
        for field in required_fields:
            if field not in result:
                if field in ["keywords", "reasoning_steps"]:
                    result[field] = ["信息不足"] if field == "keywords" else ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
                elif field == "is_ai_generated":
                    result[field] = False
                else:
                    result[field] = 0.5000
        
        # 确保推理步骤是4步
        if len(result["reasoning_steps"]) != 4:
            result["reasoning_steps"] = ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
        
        # 确保谣言概率格式正确
        result["rumor_prob"] = round(float(result["rumor_prob"]), 4) if isinstance(result["rumor_prob"], (int, float)) else 0.5000
        
        return result
    except Exception as e:
        print(f"模型调用/解析失败：{str(e)}")
        # 降级返回完整默认结果（包含keywords）
        return {
            "keywords": [content[:8] + "..."] if len(content) > 0 else ["未知"],
            "reasoning_steps": ["识别内容：检测失败", "检查事实：无相关数据", "评估合理性：无法判断", "得出结论：信息不足"],
            "is_ai_generated": False,
            "rumor_prob": 0.5000
        }

# ---------------------- 数据模型（请求/返回格式） ----------------------
# 注册请求格式
class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str

# 登录请求格式（前端传过来的参数）
class LoginRequest(BaseModel):
    username: str
    password: str

# 检测请求格式
class DetectRequest(BaseModel):
    content: str  # 检测的文本
    type: str     # 谣言类型（疫情/食品安全等）

# ---------------------- 核心接口 ----------------------
# 新增：注册接口（POST /api/register）
@app.post("/api/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    # 1. 校验参数非空
    if not request.username or not request.password or not request.confirm_password:
        raise HTTPException(status_code=400, detail="所有字段不能为空")
    
    # 2. 校验密码长度（6-72位）
    if len(request.password) < 6 or len(request.password) > 72:
        raise HTTPException(status_code=400, detail="密码长度需6-72位")
    
    # 3. 校验两次密码一致
    if request.password != request.confirm_password:
        raise HTTPException(status_code=400, detail="两次输入的密码不一致")
    
    # 4. 校验用户名是否已存在
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    # 5. 加密密码并创建用户（截断到72位避免bcrypt报错）
    hashed_pwd = hash_password(request.password[:72])
    new_user = User(
        username=request.username,
        password=hashed_pwd,
        create_time=datetime.now()
    )
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)  # 刷新获取新增用户的ID
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"创建用户失败：{str(e)}")
    
    return {
        "code": 200,
        "msg": "注册成功，请登录",
        "data": {
            "user_id": new_user.id,
            "username": new_user.username
        }
    }

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
    try:
        login_log = LoginLog(user_id=user.id, ip=req.client.host)
        db.add(login_log)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"记录登录日志失败：{str(e)}")
    
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
    
    # 2. 校验文本长度
    if len(request.content) < 1 or len(request.content) > 500:
        raise HTTPException(status_code=400, detail="文本长度需1-500字")
    
    # 3. 调用大语言模型（优先使用模拟模式避免API依赖）
    if config.LLM_FAKE:
        llm_result = fake_llm_detect(request.content, request.type)
    else:
        llm_result = real_llm_detect(request.content, request.type)
    
    # 4. 存储检测记录到数据库
    try:
        record = ReasoningRecord(
            user_id=user_id,
            content=request.content,
            type=request.type,
            rumor_prob=llm_result["rumor_prob"],
            is_ai_generated=llm_result["is_ai_generated"],
            reasoning_steps=json.dumps(llm_result["reasoning_steps"], ensure_ascii=False),
            keywords=json.dumps(llm_result["keywords"], ensure_ascii=False)  # 新增：存储关键字JSON字符串
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"存储检测记录失败：{str(e)}")
    
    # 5. 返回结果给前端
    return {
        "code": 200,
        "msg": "检测成功",
        "data": {
            "rumor_prob": llm_result["rumor_prob"],
            "is_ai_generated": llm_result["is_ai_generated"],
            "reasoning_steps": llm_result["reasoning_steps"],
            "keywords": llm_result["keywords"],  # 新增：返回关键字列表
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
    
    # 校验分页参数
    if page < 1:
        page = 1
    if size < 1 or size > 50:
        size = 10
    
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
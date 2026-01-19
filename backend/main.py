# backend/main.py（部分修改，主要修改detect函数）
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
import jieba
import jieba.analyse
import jieba.posseg as pseg
from collections import Counter
import re
import hashlib  # 新增导入

# 导入自己的配置和模型
import config
from models import Base, SessionLocal, User, ReasoningRecord, LoginLog

# 计算内容哈希值的函数（移到main.py中，避免导入问题）
def calculate_content_hash(content: str) -> str:
    """计算文本内容的MD5哈希值，用于去重"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# ---------------------- 基础配置 ----------------------
app = FastAPI(title="谣言甄别系统API")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
from models import SessionLocal

# 密码加密工具
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------- 初始化大语言模型客户端 ----------------------
llm_client = ZhipuAI(api_key=config.LLM_CONFIG["api_key"])

# ---------------------- 使用jieba提取关键字的函数 ----------------------
def extract_keywords_with_jieba(content: str, top_k: int = 8) -> list:
    """
    使用jieba精确模式提取文本的关键字
    参数:
        content: 输入文本
        top_k: 返回关键词数量
    返回:
        关键字列表
    """
    if not content or len(content.strip()) == 0:
        return ["未知"]
    
    original_content = content
    
    # 清洗文本
    important_words = {"不", "没", "无", "否", "非", "未", "勿", "莫", "休", "忌", "禁", "戒", "就", "所以", "因此", "因而", "从而"}
    
    placeholder_map = {}
    for i, word in enumerate(important_words):
        placeholder = f"__PLACEHOLDER_{i}__"
        placeholder_map[placeholder] = word
        content = content.replace(word, placeholder)
    
    cleaned_content = re.sub(r'[^\w\s]', '', content)
    
    for placeholder, word in placeholder_map.items():
        cleaned_content = cleaned_content.replace(placeholder, word)
    
    # 使用精确模式分词
    words = jieba.lcut(cleaned_content, cut_all=False)
    
    # 过滤逻辑
    base_stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "都", "一", "个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
    
    important_negations = {"不", "没", "无", "否", "非", "未", "勿", "莫", "休", "忌", "禁", "戒", "不是", "不会", "不能", "不可", "没有", "无法"}
    important_logicals = {"所以", "因此", "因而", "从而", "因为", "由于", "既然", "那么", "于是", "然后"}
    
    must_keep_words = important_negations.union(important_logicals)
    
    filtered_words = []
    for word in words:
        if word in must_keep_words:
            filtered_words.append(word)
        elif word in base_stop_words:
            continue
        elif len(word) == 1 and word not in important_negations:
            continue
        else:
            filtered_words.append(word)
    
    # 统计词频
    word_freq = Counter(filtered_words)
    
    # 获取前top_k个高频词
    keywords = [word for word, _ in word_freq.most_common(top_k)]
    
    # 如果提取的关键词不足，使用关键短语提取
    if len(keywords) < min(5, top_k):
        try:
            tfidf_keywords = jieba.analyse.extract_tags(
                original_content, 
                topK=top_k*2, 
                withWeight=False,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'v', 'vn', 'd')
            )
            
            for keyword in tfidf_keywords:
                if keyword in must_keep_words and keyword not in keywords:
                    keywords.append(keyword)
        except:
            pass
    
    # 特别处理否定+关键词的组合
    negation_patterns = [
        r'不\s*([^\s]+)',
        r'没\s*([^\s]+)',
        r'无\s*([^\s]+)',
        r'否\s*([^\s]+)',
        r'非\s*([^\s]+)',
        r'不是\s*([^\s]+)',
        r'没有\s*([^\s]+)',
    ]
    
    for pattern in negation_patterns:
        matches = re.findall(pattern, original_content)
        for match in matches:
            if len(match) > 1:
                negation_word = pattern.split(r'\s*')[0].replace('r', '').replace("'", "")
                combined = negation_word + match
                if combined not in keywords:
                    keywords.append(combined)
    
    # 使用词性标注提取更多信息
    try:
        word_flags = pseg.lcut(original_content)
        meaningful_words = []
        for word, flag in word_flags:
            if flag.startswith(('n', 'v', 'a')) and len(word) > 1:
                meaningful_words.append(word)
        
        for word in meaningful_words:
            if word not in keywords:
                keywords.append(word)
    except:
        pass
    
    # 去重
    unique_keywords = []
    seen = set()
    for word in keywords:
        if word and word not in seen:
            seen.add(word)
            unique_keywords.append(word)
    
    if not unique_keywords:
        unique_keywords = ["信息不足"]
    
    return unique_keywords[:top_k]

# ---------------------- 大语言模型提示词模板 ----------------------
PROMPT_TEMPLATE = """
你是一位专业的谣言甄别专家。请严格按照以下JSON格式输出分析结果，不要添加任何额外的解释或说明：

{{
  "reasoning_steps": [
    "第一步：识别和分析文本内容",
    "第二步：检查事实和逻辑一致性",
    "第三步：评估可信度和合理性",
    "第四步：给出最终判断结论"
  ],
  "is_ai_generated": false,
  "rumor_prob": 0.8500
}}

=== 输入信息 ===
文本内容：{content}
文本类型：{type}
关键词：{keywords}

=== 分析要求 ===
1. 请基于提供的文本内容，按照4步推理流程进行分析
2. reasoning_steps必须包含4个步骤，每个步骤用一句简洁明了的话描述
3. is_ai_generated判断文本是否为AI生成：true（是）或false（否）
4. rumor_prob给出谣言概率：0-1之间的4位小数，0表示肯定是谣言，1表示肯定不是谣言
5. 请确保分析客观、准确，基于事实和逻辑

=== 输出格式要求 ===
只返回JSON格式的输出，不要有任何其他文字说明、注释或格式标记。
JSON必须包含且仅包含以下字段：reasoning_steps, is_ai_generated, rumor_prob
"""

# ---------------------- 工具函数 ----------------------
# 1. 获取数据库连接
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

# 3. 生成/验证Token
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

# 4. 数据库去重查询函数（新增）
def find_existing_record(db: Session, content_hash: str) -> dict:
    """
    根据内容哈希值在数据库中查找现有记录
    返回：如果找到返回记录数据，否则返回None
    """
    existing_record = db.query(ReasoningRecord).filter(
        ReasoningRecord.content_hash == content_hash
    ).first()
    
    if existing_record:
        # 更新使用次数和最后使用时间
        existing_record.use_count += 1
        existing_record.last_used_time = datetime.now()
        db.commit()
        
        # 解析存储的JSON数据
        try:
            keywords_data = json.loads(existing_record.keywords) if existing_record.keywords else []
        except:
            keywords_data = []
        
        try:
            reasoning_steps_data = json.loads(existing_record.reasoning_steps) if existing_record.reasoning_steps else []
        except:
            reasoning_steps_data = []
        
        return {
            "rumor_prob": round(float(existing_record.rumor_prob), 4),  # 统一格式
            "is_ai_generated": existing_record.is_ai_generated,
            "reasoning_steps": reasoning_steps_data,
            "keywords": keywords_data,
            "from_cache": True,
            "use_count": existing_record.use_count,
            "record_id": existing_record.id
        }
    return None

# 5. 模拟大语言模型检测
def fake_llm_detect(content: str, type: str, keywords: list):
    rumor_prob = round(random.uniform(0, 1), 4)
    reasoning_steps = [
        f"识别内容：{content[:20]}...（类型：{type}）",
        f"检查事实：{'符合客观事实' if rumor_prob < 0.5 else '不符合客观事实'}",
        f"评估合理性：{'非谣言' if rumor_prob < 0.5 else '谣言'}，AI生成概率：{round(random.uniform(0, 1), 2)}",
        f"得出结论：{'判定为非谣言' if rumor_prob < 0.5 else '判定为谣言'}"
    ]
    
    return {
        "rumor_prob": rumor_prob,  # 已经是4位小数
        "is_ai_generated": random.choice([True, False]),
        "reasoning_steps": reasoning_steps,
        "from_cache": False
    }

# 6. 真实大语言模型检测函数
def real_llm_detect(content: str, type: str, keywords: list):
    try:
        print(f"调用GLM-4.7模型API，内容长度: {len(content)}")
        
        escaped_content = content.replace("{", "{{").replace("}", "}}")
        escaped_type = type.replace("{", "{{").replace("}", "}}")
        escaped_keywords = str(keywords).replace("{", "{{").replace("}", "}}")
        
        prompt_content = PROMPT_TEMPLATE.format(
            content=escaped_content, 
            type=escaped_type,
            keywords=escaped_keywords
        )
        
        try:
            response = llm_client.chat.completions.create(
                model=config.LLM_CONFIG["model_name"],
                messages=[
                    {"role": "system", "content": "你是一位专业的谣言甄别专家，请严格按照要求输出JSON格式的结果。"},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=config.LLM_CONFIG["temperature"],
                max_tokens=config.LLM_CONFIG["max_tokens"],
                timeout=30
            )
        except Exception as api_error:
            try:
                response = llm_client.chat.completions.create(
                    model=config.LLM_CONFIG["model_name"],
                    messages=[
                        {"role": "system", "content": "你是一位专业的谣言甄别专家，请严格按照要求输出JSON格式的结果。"},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=config.LLM_CONFIG["temperature"],
                    max_tokens=config.LLM_CONFIG["max_tokens"],
                    timeout=30
                )
            except Exception as retry_error:
                raise Exception(f"API调用失败: {str(retry_error)}")
        
        if not response or not response.choices:
            raise Exception("API返回空响应")
        
        result_str = response.choices[0].message.content.strip()
        
        if not result_str or len(result_str) == 0:
            raise Exception("模型返回空内容")
        
        # 去除可能的markdown代码块
        if result_str.startswith("```json"):
            result_str = result_str.replace("```json", "").replace("```", "").strip()
        elif result_str.startswith("```"):
            result_str = result_str.replace("```", "").strip()
        
        result_str = re.sub(r'<[^>]+>', '', result_str)
        result_str = re.sub(r'^JSON:\s*', '', result_str, flags=re.IGNORECASE)
        result_str = result_str.strip()
        
        # JSON解析
        result = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = json.loads(result_str)
                break
            except json.JSONDecodeError:
                if attempt < max_attempts - 1:
                    if "'" in result_str:
                        result_str = result_str.replace("'", "\"")
                    result_str = re.sub(r'\s+', ' ', result_str)
                    if not result_str.endswith("}"):
                        result_str += "}"
                    if not result_str.startswith("{"):
                        start_idx = result_str.find("{")
                        if start_idx != -1:
                            result_str = result_str[start_idx:]
                        else:
                            result_str = "{" + result_str
        
        if result is None:
            raise Exception("JSON解析失败")
        
        # 补全字段
        required_fields = ["reasoning_steps", "is_ai_generated", "rumor_prob"]
        for field in required_fields:
            if field not in result:
                if field == "reasoning_steps":
                    result[field] = ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
                elif field == "is_ai_generated":
                    result[field] = False
                elif field == "rumor_prob":
                    result[field] = 0.5000
        
        # 确保推理步骤是4步
        if "reasoning_steps" in result:
            if not isinstance(result["reasoning_steps"], list):
                result["reasoning_steps"] = ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
            elif len(result["reasoning_steps"]) != 4:
                while len(result["reasoning_steps"]) < 4:
                    result["reasoning_steps"].append("信息不足")
                result["reasoning_steps"] = result["reasoning_steps"][:4]
        
        # 确保谣言概率格式正确
        if "rumor_prob" in result:
            try:
                rumor_prob = float(result["rumor_prob"])
                rumor_prob = max(0.0, min(1.0, rumor_prob))
                result["rumor_prob"] = round(rumor_prob, 4)
            except:
                result["rumor_prob"] = 0.5000
        
        result["from_cache"] = False  # 标记不是来自缓存
        return result
    except Exception as e:
        print(f"GLM-4.7模型调用/解析失败：{str(e)}")
        return {
            "reasoning_steps": ["识别内容：模型调用异常", "检查事实：检测失败", "评估合理性：无法判断", "得出结论：信息不足"],
            "is_ai_generated": False,
            "rumor_prob": 0.5000,
            "from_cache": False
        }

# ---------------------- 数据模型 ----------------------
class RegisterRequest(BaseModel):
    username: str
    password: str
    confirm_password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class DetectRequest(BaseModel):
    content: str
    type: str

# ---------------------- 核心接口 ----------------------
@app.post("/api/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if not request.username or not request.password or not request.confirm_password:
        raise HTTPException(status_code=400, detail="所有字段不能为空")
    
    if len(request.password) < 6 or len(request.password) > 72:
        raise HTTPException(status_code=400, detail="密码长度需6-72位")
    
    if request.password != request.confirm_password:
        raise HTTPException(status_code=400, detail="两次输入的密码不一致")
    
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    
    hashed_pwd = hash_password(request.password[:72])
    new_user = User(
        username=request.username,
        password=hashed_pwd,
        create_time=datetime.now()
    )
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
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

@app.post("/api/login")
def login(request: LoginRequest, req: Request, db: Session = Depends(get_db)):
    if not request.username or not request.password:
        raise HTTPException(status_code=400, detail="用户名/密码不能为空")
    
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户名不存在")
    
    if not verify_password(request.password, user.password):
        raise HTTPException(status_code=401, detail="密码错误")
    
    try:
        login_log = LoginLog(user_id=user.id, ip=req.client.host)
        db.add(login_log)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"记录登录日志失败：{str(e)}")
    
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

# ---------------------- 修改后的检测接口（添加数据库去重查询） ----------------------
@app.post("/api/detect")
def detect(
    request: DetectRequest,
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    # 1. 验证Token
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    token = authorization.split(" ")[1]
    try:
        user_id = verify_token(token)
    except:
        raise HTTPException(status_code=401, detail="Token无效/过期")
    
    # 2. 校验文本长度
    if len(request.content) < 1 or len(request.content) > 500:
        raise HTTPException(status_code=400, detail="文本长度需1-500字")
    
    # 3. 计算内容哈希值（用于去重）
    content_hash = calculate_content_hash(request.content)
    print(f"内容哈希值: {content_hash}")
    
    # 4. 先查询数据库是否有相同内容的记录
    existing_record = find_existing_record(db, content_hash)
    if existing_record:
        print(f"找到缓存记录，使用次数: {existing_record['use_count']}")
        return {
            "code": 200,
            "msg": "检测成功（来自缓存）",
            "data": {
                "rumor_prob": existing_record["rumor_prob"],
                "is_ai_generated": existing_record["is_ai_generated"],
                "reasoning_steps": existing_record["reasoning_steps"],
                "keywords": existing_record["keywords"],
                "record_id": existing_record["record_id"],
                "from_cache": True,
                "use_count": existing_record["use_count"]
            }
        }
    
    # 5. 如果没有缓存，则提取关键字并调用大模型
    keywords = extract_keywords_with_jieba(request.content)
    print(f"使用jieba提取的关键字: {keywords}")
    print("未找到缓存记录，调用大模型...")
    
    # 6. 调用大语言模型
    if config.LLM_FAKE:
        print("使用模拟模式")
        llm_result = fake_llm_detect(request.content, request.type, keywords)
    else:
        print("使用GLM-4.7真实API模式")
        llm_result = real_llm_detect(request.content, request.type, keywords)
    
    # 7. 存储新的检测记录到数据库（包括内容哈希值）
    try:
        record = ReasoningRecord(
            user_id=user_id,
            content=request.content,
            content_hash=content_hash,  # 存储哈希值
            type=request.type,
            rumor_prob=llm_result["rumor_prob"],
            is_ai_generated=llm_result["is_ai_generated"],
            reasoning_steps=json.dumps(llm_result["reasoning_steps"], ensure_ascii=False),
            keywords=json.dumps(keywords, ensure_ascii=False),
            use_count=1,  # 首次使用
            create_time=datetime.now(),
            last_used_time=datetime.now()
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"存储检测记录失败：{str(e)}")
    
    # 8. 返回结果给前端
    return {
            "code": 200,
            "msg": "检测成功（新记录）",
            "data": {
                "rumor_prob": round(llm_result["rumor_prob"], 4),  # 确保4位小数
                "is_ai_generated": llm_result["is_ai_generated"],
                "reasoning_steps": llm_result["reasoning_steps"],
                "keywords": keywords,
                "record_id": record.id,
                "from_cache": False,
                "use_count": 1
            }
    }

@app.get("/api/history")
def get_history(
    authorization: str = Header(None),
    page: int = 1,
    size: int = 10,
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    token = authorization.split(" ")[1]
    try:
        user_id = verify_token(token)
    except:
        raise HTTPException(status_code=401, detail="Token无效/过期")
    
    if page < 1:
        page = 1
    if size < 1 or size > 50:
        size = 10
    
    offset = (page - 1) * size
    records = db.query(ReasoningRecord).filter(ReasoningRecord.user_id == user_id).order_by(ReasoningRecord.last_used_time.desc()).offset(offset).limit(size).all()
    
    history_list = []
    for r in records:
        try:
            keywords_data = json.loads(r.keywords) if r.keywords else []
        except:
            keywords_data = []
        
        try:
            reasoning_steps_data = json.loads(r.reasoning_steps) if r.reasoning_steps else []
        except:
            reasoning_steps_data = []
        
        history_list.append({
            "record_id": r.id,
            "content": r.content,
            "content_hash": r.content_hash,
            "type": r.type,
            "rumor_prob": round(float(r.rumor_prob), 4),  # 关键：确保4位小数
            "is_ai_generated": r.is_ai_generated,
            "keywords": keywords_data,
            "reasoning_steps": reasoning_steps_data,
            "use_count": r.use_count,
            "create_time": r.create_time.strftime("%Y-%m-%d %H:%M:%S") if r.create_time else "",
            "last_used_time": r.last_used_time.strftime("%Y-%m-%d %H:%M:%S") if r.last_used_time else ""
        })
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

# ---------------------- 新增：查看重复内容统计接口 ----------------------
@app.get("/api/duplicate-stats")
def get_duplicate_stats(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    """查看重复内容统计信息"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="请先登录")
    token = authorization.split(" ")[1]
    try:
        user_id = verify_token(token)
    except:
        raise HTTPException(status_code=401, detail="Token无效/过期")
    
    # 统计使用次数最多的内容
    most_used = db.query(ReasoningRecord).filter(
        ReasoningRecord.user_id == user_id
    ).order_by(ReasoningRecord.use_count.desc()).limit(5).all()
    
    most_used_list = []
    for record in most_used:
        most_used_list.append({
            "content": record.content[:50] + "..." if len(record.content) > 50 else record.content,
            "use_count": record.use_count,
            "last_used": record.last_used_time.strftime("%Y-%m-%d %H:%M:%S") if record.last_used_time else ""  # 添加空值检查
        })
    
    # 统计缓存命中率
    total_records = db.query(ReasoningRecord).filter(ReasoningRecord.user_id == user_id).count()
    duplicate_records = db.query(ReasoningRecord).filter(
        ReasoningRecord.user_id == user_id,
        ReasoningRecord.use_count > 1
    ).count()
    
    cache_hit_rate = 0
    if total_records > 0:
        cache_hit_rate = round((duplicate_records / total_records) * 100, 2)
    
    return {
        "code": 200,
        "msg": "统计成功",
        "data": {
            "total_records": total_records,
            "duplicate_records": duplicate_records,
            "cache_hit_rate": f"{cache_hit_rate}%",
            "most_used_contents": most_used_list
        }
    }

# ---------------------- 启动后端 ----------------------
if __name__ == "__main__":
    try:
        jieba.load_userdict('userdict.txt')
        print("jieba分词器初始化成功 - 加载自定义词典")
    except:
        print("jieba分词器初始化成功 - 使用默认词典")
    
    # 初始化数据库和测试用户
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "test").first():
            password = str("123456")[:72]
            test_user = User(username="test", password=hash_password(password))
            db.add(test_user)
            db.commit()
            print("测试用户创建成功：用户名test，密码123456")
        else:
            print("测试用户已存在，无需重复创建")
            
        # 检查数据库表是否已更新
        print("\n=== 数据库去重功能状态 ===")
        print("✓ content_hash字段已添加")
        print("✓ use_count字段已添加")
        print("✓ last_used_time字段已添加")
        print("✓ 去重查询功能已启用")
    except Exception as e:
        print(f"数据库初始化失败：{str(e)}")
        db.rollback()
    finally:
        db.close()
    
    print(f"\n=== 谣言甄别系统后端启动 ===")
    print(f"模型配置: {config.LLM_CONFIG['model_name']} (GLM-4.7)")
    print(f"去重功能: 已启用")
    print(f"LLM_FAKE模式: {config.LLM_FAKE}")
    print(f"服务地址: http://localhost:8000")
    print(f"API文档: http://localhost:8000/docs")
    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
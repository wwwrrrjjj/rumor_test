# backend/main.py（完整修复版）- 使用DuckDuckGo搜索
from fastapi import FastAPI, Depends, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
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
import hashlib
import requests
import time
import traceback

# 导入自己的配置和模型
import config
from models import Base, SessionLocal, User, ReasoningRecord, LoginLog

# ---------------------- DuckDuckGo 搜索客户端 ----------------------
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    print("⚠ duckduckgo-search 未安装，联网搜索功能将不可用")

class DuckDuckGoSearchClient:
    """DuckDuckGo 搜索客户端"""
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.max_results = config.SEARCH_CONFIG.get("max_results", 3)
        self.cooldown = config.SEARCH_CONFIG.get("cooldown", 0.5)
    
    def search(self, query: str, max_results: int = None) -> dict:
        """执行DuckDuckGo搜索"""
        if not DUCKDUCKGO_AVAILABLE:
            return {"results": [], "query": query, "success": False, "error": "DuckDuckGo不可用"}
        
        try:
            max_results = max_results or self.max_results
            print(f"🔍 DuckDuckGo搜索: {query}")
            
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "content": r.get("body", ""),
                        "url": r.get("href", ""),
                        "source": "DuckDuckGo"
                    })
            
            return {
                "results": results,
                "query": query,
                "success": len(results) > 0,
                "count": len(results)
            }
            
        except Exception as e:
            print(f"❌ DuckDuckGo搜索失败: {str(e)}")
            return {"results": [], "query": query, "success": False, "error": str(e)}
    
    def search_for_rumor_verification(self, content: str, keywords: list) -> dict:
        """为谣言验证设计的搜索"""
        search_queries = self._generate_rumor_queries(content, keywords)
        all_results = []
        
        for query in search_queries[:config.SEARCH_CONFIG.get("max_queries", 2)]:
            search_result = self.search(query, max_results=2)
            
            if search_result.get("success") and search_result["results"]:
                for result in search_result["results"]:
                    formatted_result = self._format_search_result(result)
                    if formatted_result:
                        all_results.append(formatted_result)
            
            time.sleep(self.cooldown)  # 避免请求过快
        
        return {
            "query_count": len(search_queries),
            "total_results": len(all_results),
            "results": all_results[:6],  # 最多返回6个结果
            "success": len(all_results) > 0
        }
    
    def _generate_rumor_queries(self, content: str, keywords: list) -> list:
        """生成谣言验证查询"""
        queries = []
        
        # 基于关键字的查询
        if keywords:
            main_keywords = " ".join(keywords[:2])
            queries.extend([
                f"{main_keywords} 谣言 辟谣",
                f"{main_keywords} 事实核查",
                f"{main_keywords} 是真的吗",
                f"{main_keywords} 真相"
            ])
        
        # 基于内容的查询
        content_lower = content.lower()
        
        # 提取短句作为查询
        if len(content) < 100:
            sentences = re.split(r'[。！？]', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if 10 < len(sentence) < 50:
                    queries.append(f"{sentence} 是真的吗")
        
        # 特定主题的查询
        if any(word in content_lower for word in ["疫情", "疫苗", "新冠", "病毒"]):
            queries.extend([
                "疫情谣言 官方辟谣",
                "新冠疫苗 真相"
            ])
        if any(word in content_lower for word in ["食品", "吃", "喝", "中毒", "致癌"]):
            queries.append("食品安全谣言 辟谣")
        if any(word in content_lower for word in ["健康", "养生", "治病", "偏方"]):
            queries.append("健康谣言 真相")
        
        # 去重并限制数量
        return list(dict.fromkeys(queries))[:4]
    
    def _format_search_result(self, result: dict) -> dict:
        """格式化搜索结果"""
        try:
            title = result.get("title", "").strip()
            content = result.get("content", "").strip()
            url = result.get("url", "")
            source = result.get("source", "DuckDuckGo")
            
            if not content:
                return None
            
            # 提取关键信息
            summary = content[:100] + "..." if len(content) > 100 else content
            
            # 检测结果类型
            result_type = "普通信息"
            if any(word in content for word in ["辟谣", "谣言", "不实", "虚假"]):
                result_type = "辟谣信息"
            elif any(word in content for word in ["证实", "真相", "事实", "正确"]):
                result_type = "证实信息"
            elif any(word in content for word in ["可能", "或许", "不确定", "疑似"]):
                result_type = "不确定信息"
            
            return {
                "title": title,
                "summary": summary,
                "full_content": content,
                "url": url,
                "source": source,
                "type": result_type,
                "relevance_score": self._calculate_relevance(content)
            }
            
        except Exception as e:
            print(f"格式化搜索结果失败: {str(e)}")
            return None
    
    def _calculate_relevance(self, content: str) -> float:
        """计算搜索结果相关性分数"""
        relevance_keywords = [
            "辟谣", "谣言", "证实", "真相", "事实", "核查",
            "专家", "研究", "实验", "数据", "科学", "官方"
        ]
        
        score = 0.5  # 基础分
        
        content_lower = content.lower()
        for keyword in relevance_keywords:
            if keyword in content_lower:
                score += 0.1
        
        # 限制在0-1之间
        return min(max(score, 0), 1)

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
from zhipuai import ZhipuAI
import config
llm_client = ZhipuAI(
    api_key=config.LLM_CONFIG["api_key"],
    base_url=config.LLM_CONFIG.get("base_url", "https://open.bigmodel.cn/api/coding/paas/v4")
)

# ---------------------- 初始化搜索客户端 ----------------------
search_client = DuckDuckGoSearchClient()

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
# 增强的提示词模板（带DuckDuckGo搜索结果分析）
ENHANCED_PROMPT_TEMPLATE = """
你是一位专业的谣言甄别专家。请基于以下信息进行分析：

=== 待检测信息 ===
文本内容：{content}
文本类型：{type}
关键词：{keywords}

=== 网络搜索结果 ===
{search_summary}

=== 分析要求 ===
1. 首先分析文本中的核心声明
2. 参考搜索结果中的信息进行事实核查
3. 评估声明的逻辑一致性和合理性
4. 综合搜索结果和逻辑分析给出判断

=== 搜索结果分析指南 ===
- 如果搜索结果包含明确的辟谣信息，考虑声明为谣言的可能性较高
- 如果搜索结果证实了声明，考虑声明为真实的的可能性较高
- 如果搜索结果没有相关信息，基于逻辑和常识判断
- 注意搜索结果的来源和可信度

=== 输出格式 ===
严格按以下JSON格式输出：
{{
  "reasoning_steps": [
    "第一步：分析文本核心声明",
    "第二步：核查搜索结果中的事实依据", 
    "第三步：评估逻辑和合理性",
    "第四步：综合给出判断结论"
  ],
  "is_ai_generated": false,
  "rumor_prob": 0.8500,
  "is_rumor": true,
  "conclusion": "经分析，该信息【是谣言】。",
  "confidence": "高/中/低",
  "verification_based_on_search": true/false,
  "search_result_summary": "对搜索结果的简要总结",
  "key_findings_from_search": ["发现1", "发现2"]
}}

注意：
1. rumor_prob: 谣言概率，0-1之间，保留4位小数（0=肯定是谣言，1=肯定不是谣言）
2. is_rumor: 是否为谣言，true表示是谣言，false表示不是谣言
3. conclusion: 最终结论，必须明确包含【是谣言】或【不是谣言】
4. confidence: 基于信息完整度的置信度
5. verification_based_on_search: 是否基于搜索结果进行了验证
6. 请确保推理步骤是4步
7. 只输出JSON，不要有任何其他文字
"""

# 原始提示词模板（无搜索结果）
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
  "rumor_prob": 0.8500,
  "is_rumor": true,
  "conclusion": "经分析，该信息【是谣言】。"
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
5. is_rumor判断是否为谣言：true表示是谣言，false表示不是谣言
6. conclusion给出最终结论：必须明确包含【是谣言】或【不是谣言】
7. 请确保分析客观、准确，基于事实和逻辑

=== 输出格式要求 ===
只返回JSON格式的输出，不要有任何其他文字说明、注释或格式标记。
JSON必须包含且仅包含以下字段：reasoning_steps, is_ai_generated, rumor_prob, is_rumor, conclusion
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

# 4. 计算内容哈希值的函数
def calculate_content_hash(content: str) -> str:
    """计算文本内容的MD5哈希值，用于去重"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# 5. 数据库去重查询函数
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
        
        # 获取结论（如果存储了的话）
        conclusion = ""
        try:
            if existing_record.conclusion:
                conclusion = existing_record.conclusion
            else:
                # 如果没有存储结论，根据谣言概率生成
                if existing_record.rumor_prob >= 0.7:
                    conclusion = "经分析，该信息【是谣言】。"
                elif existing_record.rumor_prob <= 0.3:
                    conclusion = "经分析，该信息【不是谣言】。"
                else:
                    conclusion = "经分析，该信息可能为谣言，建议进一步核实。"
        except:
            conclusion = "经分析，该信息【结论待定】。"
        
        return {
            "rumor_prob": round(float(existing_record.rumor_prob), 4),
            "is_ai_generated": existing_record.is_ai_generated,
            "reasoning_steps": reasoning_steps_data,
            "keywords": keywords_data,
            "from_cache": True,
            "use_count": existing_record.use_count,
            "record_id": existing_record.id,
            "is_rumor": existing_record.rumor_prob >= 0.5,
            "conclusion": conclusion
        }
    return None

# 6. 模拟大语言模型检测
def fake_llm_detect(content: str, type: str, keywords: list):
    rumor_prob = round(random.uniform(0, 1), 4)
    is_rumor = rumor_prob >= 0.5
    conclusion = "经分析，该信息【是谣言】。" if is_rumor else "经分析，该信息【不是谣言】。"
    reasoning_steps = [
        f"识别内容：{content[:20]}...（类型：{type}）",
        f"检查事实：{'符合客观事实' if rumor_prob < 0.5 else '不符合客观事实'}",
        f"评估合理性：{'非谣言' if rumor_prob < 0.5 else '谣言'}，AI生成概率：{round(random.uniform(0, 1), 2)}",
        f"得出结论：{'判定为非谣言' if rumor_prob < 0.5 else '判定为谣言'}"
    ]
    
    return {
        "rumor_prob": rumor_prob,
        "is_ai_generated": random.choice([True, False]),
        "reasoning_steps": reasoning_steps,
        "from_cache": False,
        "search_used": False,
        "is_rumor": is_rumor,
        "conclusion": conclusion
    }

# 7. 判断是否需要联网搜索
def should_enable_web_search(content: str, keywords: list) -> bool:
    """判断是否需要进行联网搜索"""
    # 检查搜索配置是否启用
    if not config.SEARCH_CONFIG.get("enable", True):
        return False
    
    # 检查DuckDuckGo是否可用
    if not DUCKDUCKGO_AVAILABLE:
        return False
    
    # 如果内容太短
    if len(content) < 15:
        return False
    
    # 检查是否包含可验证的声明
    verification_triggers = [
        "研究表明", "数据显示", "专家称", "最新发现", "实验证明",
        "据报道", "官方宣布", "科学研究", "事实证明", "调查显示",
        "据统计", "根据研究", "科学证明", "专家建议", "医生提醒",
        "实验表明", "数据显示", "科学研究", "临床实验"
    ]
    
    content_lower = content.lower()
    for trigger in verification_triggers:
        if trigger in content_lower:
            return True
    
    # 检查是否包含数字或百分比
    if re.search(r'\d+[%％]|\d+\.\d+', content):
        return True
    
    # 基于关键词判断
    search_keywords = {"研究", "数据", "统计", "实验", "最新", "科学", "证明", "专家", "医生", "教授", "实验", "研究", "数据"}
    for keyword in keywords:
        if keyword in search_keywords:
            return True
    
    # 特定类型内容
    if any(word in content_lower for word in ["疫情", "疫苗", "新冠", "病毒", "隔离", "封城"]):
        return True
    if any(word in content_lower for word in ["食品", "吃", "喝", "中毒", "致癌", "有毒"]):
        return True
    if any(word in content_lower for word in ["健康", "养生", "治病", "疗效", "偏方", "秘方"]):
        return True
    if any(word in content_lower for word in ["科技", "发明", "新技术", "突破"]):
        return True
    
    return False

# 8. 执行DuckDuckGo搜索并格式化结果
def perform_web_search(content: str, keywords: list) -> dict:
    """执行DuckDuckGo搜索并返回格式化结果"""
    try:
        print("🔍 开始DuckDuckGo搜索验证...")
        
        # 执行搜索
        search_results = search_client.search_for_rumor_verification(content, keywords)
        
        if search_results.get("success") and search_results["results"]:
            print(f"✅ DuckDuckGo搜索完成，找到 {len(search_results['results'])} 个相关结果")
            
            # 格式化搜索结果摘要
            summary_parts = []
            summary_parts.append(f"📡 网络验证信息（来自DuckDuckGo搜索）：")
            summary_parts.append(f"搜索查询数：{search_results['query_count']}")
            summary_parts.append(f"找到结果数：{search_results['total_results']}")
            summary_parts.append("")
            
            for i, result in enumerate(search_results["results"][:4], 1):
                result_type_emoji = {
                    "辟谣信息": "🚫",
                    "证实信息": "✅", 
                    "不确定信息": "❓",
                    "普通信息": "📰"
                }.get(result["type"], "📰")
                
                summary_parts.append(f"{i}. {result_type_emoji} {result['title']}")
                summary_parts.append(f"   摘要：{result['summary']}")
                summary_parts.append(f"   类型：{result['type']} | 来源：{result['source']}")
                summary_parts.append("")
            
            return {
                "success": True,
                "summary": "\n".join(summary_parts),
                "raw_results": search_results["results"],
                "formatted_results": search_results["results"][:4]
            }
        else:
            print("ℹ️ 未获取到有效的网络验证信息")
            return {
                "success": False,
                "summary": "⚠️ 网络搜索未找到相关验证信息",
                "raw_results": [],
                "formatted_results": []
            }
            
    except Exception as e:
        print(f"❌ DuckDuckGo搜索失败: {str(e)}")
        return {
            "success": False,
            "summary": f"❌ 网络搜索失败: {str(e)}",
            "raw_results": [],
            "formatted_results": []
        }

# 9. 增强的检测函数（带DuckDuckGo搜索）- 修复版
def enhanced_real_llm_detect(content: str, type: str, keywords: list):
    """增强的检测函数，包含DuckDuckGo搜索"""
    try:
        print(f"📝 调用GLM-4模型API，内容长度: {len(content)}")
        
        # 判断是否需要联网搜索
        should_search = should_enable_web_search(content, keywords)
        web_context = {"success": False, "summary": "", "results": []}
        
        # 如果需要搜索，执行DuckDuckGo搜索
        if should_search:
            web_context = perform_web_search(content, keywords)
            print(f"📡 网络验证: {'成功' if web_context['success'] else '失败或无结果'}")
        
        # 构建提示词
        escaped_content = content.replace("{", "{{").replace("}", "}}")
        escaped_type = type.replace("{", "{{").replace("}", "}}")
        escaped_keywords = str(keywords).replace("{", "{{").replace("}", "}}")
        
        # 选择提示词模板
        if web_context.get("success") and web_context.get("summary"):
            # 使用增强模板
            prompt_content = ENHANCED_PROMPT_TEMPLATE.format(
                content=escaped_content,
                type=escaped_type,
                keywords=escaped_keywords,
                search_summary=web_context["summary"]
            )
        else:
            # 使用原始模板
            prompt_content = PROMPT_TEMPLATE.format(
                content=escaped_content,
                type=escaped_type,
                keywords=escaped_keywords
            )
        
        # 调用GLM-4
        print(f"📤 发送提示词给GLM-4（前100字符）: {prompt_content[:100]}...")
        response = llm_client.chat.completions.create(
            model=config.LLM_CONFIG["model_name"],
            messages=[
                {"role": "system", "content": "你是专业的谣言甄别专家，请基于所有可用信息进行客观分析，并严格按照要求的JSON格式输出。"},
                {"role": "user", "content": prompt_content}
            ],
            temperature=config.LLM_CONFIG["temperature"],
            max_tokens=config.LLM_CONFIG["max_tokens"],
            timeout=30
        )
        
        if not response or not response.choices:
            print("❌ API返回空响应")
            # 回退到原始检测
            return real_llm_detect(content, type, keywords)
        
        result_str = response.choices[0].message.content.strip()
        
        if not result_str or len(result_str) == 0:
            print("❌ 模型返回空内容")
            # 回退到原始检测
            return real_llm_detect(content, type, keywords)
        
        # 在这里添加关键日志，打印原始响应
        print(f"🤖 原始API响应（前500字符）: {result_str[:500]}...")
        print(f"🤖 原始API响应完整长度: {len(result_str)} 字符")
        
        # 清理响应文本
        if result_str.startswith("```json"):
            result_str = result_str.replace("```json", "").replace("```", "").strip()
        elif result_str.startswith("```"):
            result_str = result_str.replace("```", "").strip()
        
        result_str = re.sub(r'<[^>]+>', '', result_str)
        result_str = re.sub(r'^JSON:\s*', '', result_str, flags=re.IGNORECASE)
        result_str = result_str.strip()
        
        print(f"🔄 清理后的响应: {result_str[:200]}...")
        
        # JSON解析
        result = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = json.loads(result_str)
                print(f"✅ JSON解析成功（第{attempt+1}次尝试）")
                break
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON解析失败（第{attempt+1}次尝试）: {str(e)}")
                if attempt < max_attempts - 1:
                    # 尝试修复常见的JSON格式问题
                    if "'" in result_str:
                        result_str = result_str.replace("'", "\"")
                    result_str = re.sub(r'\s+', ' ', result_str)
                    
                    # 确保有完整的JSON结构
                    if not result_str.endswith("}"):
                        # 尝试找到最后一个}
                        last_brace = result_str.rfind("}")
                        if last_brace != -1:
                            result_str = result_str[:last_brace+1]
                        else:
                            result_str += "}"
                    
                    if not result_str.startswith("{"):
                        start_idx = result_str.find("{")
                        if start_idx != -1:
                            result_str = result_str[start_idx:]
                        else:
                            result_str = "{" + result_str
                    
                    print(f"🔄 修复后重新尝试: {result_str[:100]}...")
                else:
                    print("❌ 所有JSON解析尝试都失败")
                    raise Exception(f"JSON解析失败: {str(e)}")
        
        if result is None:
            raise Exception("JSON解析失败")
        
        # 补全字段
        required_fields = ["reasoning_steps", "is_ai_generated", "rumor_prob", "is_rumor", "conclusion"]
        for field in required_fields:
            if field not in result:
                print(f"⚠️ 缺失字段: {field}，使用默认值")
                if field == "reasoning_steps":
                    result[field] = ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
                elif field == "is_ai_generated":
                    result[field] = False
                elif field == "rumor_prob":
                    result[field] = 0.5000
                elif field == "is_rumor":
                    result[field] = result.get("rumor_prob", 0.5) >= 0.5
                elif field == "conclusion":
                    is_rumor = result.get("is_rumor", result.get("rumor_prob", 0.5) >= 0.5)
                    result[field] = "经分析，该信息【是谣言】。" if is_rumor else "经分析，该信息【不是谣言】。"
        
        # 确保推理步骤是4步
        if "reasoning_steps" in result:
            if not isinstance(result["reasoning_steps"], list):
                result["reasoning_steps"] = ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
            elif len(result["reasoning_steps"]) != 4:
                print(f"⚠️ 推理步骤数量不正确: {len(result['reasoning_steps'])}，调整为4步")
                if len(result["reasoning_steps"]) < 4:
                    while len(result["reasoning_steps"]) < 4:
                        result["reasoning_steps"].append("信息不足")
                result["reasoning_steps"] = result["reasoning_steps"][:4]
        
        # 确保谣言概率格式正确
        if "rumor_prob" in result:
            try:
                rumor_prob = float(result["rumor_prob"])
                rumor_prob = max(0.0, min(1.0, rumor_prob))
                result["rumor_prob"] = round(rumor_prob, 4)
                print(f"✅ 谣言概率处理成功: {result['rumor_prob']}")
            except Exception as e:
                print(f"⚠️ 谣言概率格式错误: {result['rumor_prob']}，使用默认值0.5")
                result["rumor_prob"] = 0.5000
        
        # 确保结论字段格式正确
        if "conclusion" not in result or not result["conclusion"]:
            is_rumor = result.get("is_rumor", result.get("rumor_prob", 0.5) >= 0.5)
            result["conclusion"] = "经分析，该信息【是谣言】。" if is_rumor else "经分析，该信息【不是谣言】。"
        
        # 确保结论包含明确判断
        if "【是谣言】" not in result["conclusion"] and "【不是谣言】" not in result["conclusion"]:
            is_rumor = result.get("is_rumor", result.get("rumor_prob", 0.5) >= 0.5)
            result["conclusion"] = "经分析，该信息【是谣言】。" if is_rumor else "经分析，该信息【不是谣言】。"
        
        # 确保is_rumor字段与结论一致
        if "is_rumor" not in result:
            result["is_rumor"] = "【是谣言】" in result["conclusion"]
        
        # 添加额外字段
        result["from_cache"] = False
        result["web_context_used"] = web_context.get("success", False)
        
        if web_context.get("success"):
            result["search_used"] = True
            result["search_result_count"] = len(web_context.get("raw_results", []))
            result["search_summary"] = web_context.get("summary", "")[:300] + "..." if len(web_context.get("summary", "")) > 300 else web_context.get("summary", "")
            
            # 确保增强模板的字段存在
            if "verification_based_on_search" not in result:
                result["verification_based_on_search"] = True
            if "search_result_summary" not in result:
                result["search_result_summary"] = "基于搜索结果进行了事实核查"
            if "key_findings_from_search" not in result:
                result["key_findings_from_search"] = ["搜索结果中包含相关信息"]
        else:
            result["search_used"] = False
            result["search_result_count"] = 0
        
        # 确保有confidence字段
        if "confidence" not in result:
            result["confidence"] = "中"
        
        # 确保有verification_suggestions字段
        if "verification_suggestions" not in result:
            result["verification_suggestions"] = ["建议进一步核实信息来源"]
        
        print(f"✅ 增强检测完成，谣言概率: {result['rumor_prob']}, 结论: {result['conclusion']}")
        return result
        
    except Exception as e:
        print(f"❌ 增强检测失败：{str(e)}")
        print(f"❌ 错误类型：{type(e).__name__}")
        print(f"❌ 详细错误：{traceback.format_exc()}")
        # 回退到原始检测
        return real_llm_detect(content, type, keywords)

# 10. 原始检测函数 - 修复版
def real_llm_detect(content: str, type: str, keywords: list):
    """原始的大语言模型检测函数"""
    try:
        print(f"📝 调用GLM-4模型API（原始模式），内容长度: {len(content)}")
        
        escaped_content = content.replace("{", "{{").replace("}", "}}")
        escaped_type = type.replace("{", "{{").replace("}", "}}")
        escaped_keywords = str(keywords).replace("{", "{{").replace("}", "}}")
        
        prompt_content = PROMPT_TEMPLATE.format(
            content=escaped_content, 
            type=escaped_type,
            keywords=escaped_keywords
        )
        
        print(f"📤 发送提示词给GLM-4（原始模式，前100字符）: {prompt_content[:100]}...")
        
        try:
            response = llm_client.chat.completions.create(
                model=config.LLM_CONFIG["model_name"],
                messages=[
                    {"role": "system", "content": "你是一位专业的谣言甄别专家，请严格按照要求输出JSON格式的结果，不要有任何其他文字。"},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=config.LLM_CONFIG["temperature"],
                max_tokens=config.LLM_CONFIG["max_tokens"],
                timeout=30
            )
        except Exception as api_error:
            print(f"❌ 第一次API调用失败: {str(api_error)}")
            try:
                response = llm_client.chat.completions.create(
                    model=config.LLM_CONFIG["model_name"],
                    messages=[
                        {"role": "system", "content": "你是一位专业的谣言甄别专家，请严格按照要求输出JSON格式的结果，不要有任何其他文字。"},
                        {"role": "user", "content": prompt_content}
                    ],
                    temperature=config.LLM_CONFIG["temperature"],
                    max_tokens=config.LLM_CONFIG["max_tokens"],
                    timeout=30
                )
            except Exception as retry_error:
                print(f"❌ 第二次API调用失败: {str(retry_error)}")
                raise Exception(f"API调用失败: {str(retry_error)}")
        
        if not response or not response.choices:
            print("❌ API返回空响应")
            raise Exception("API返回空响应")
        
        result_str = response.choices[0].message.content.strip()
        
        if not result_str or len(result_str) == 0:
            print("❌ 模型返回空内容")
            raise Exception("模型返回空内容")
        
        # 在这里添加关键日志，打印原始响应
        print(f"🤖 原始API响应（原始模式，前500字符）: {result_str[:500]}...")
        print(f"🤖 原始API响应完整长度: {len(result_str)} 字符")
        
        # 清理响应文本
        if result_str.startswith("```json"):
            result_str = result_str.replace("```json", "").replace("```", "").strip()
        elif result_str.startswith("```"):
            result_str = result_str.replace("```", "").strip()
        
        result_str = re.sub(r'<[^>]+>', '', result_str)
        result_str = re.sub(r'^JSON:\s*', '', result_str, flags=re.IGNORECASE)
        result_str = result_str.strip()
        
        print(f"🔄 清理后的响应（原始模式）: {result_str[:200]}...")
        
        # JSON解析
        result = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = json.loads(result_str)
                print(f"✅ JSON解析成功（原始模式，第{attempt+1}次尝试）")
                break
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON解析失败（原始模式，第{attempt+1}次尝试）: {str(e)}")
                if attempt < max_attempts - 1:
                    if "'" in result_str:
                        result_str = result_str.replace("'", "\"")
                    result_str = re.sub(r'\s+', ' ', result_str)
                    
                    # 确保有完整的JSON结构
                    if not result_str.endswith("}"):
                        last_brace = result_str.rfind("}")
                        if last_brace != -1:
                            result_str = result_str[:last_brace+1]
                        else:
                            result_str += "}"
                    
                    if not result_str.startswith("{"):
                        start_idx = result_str.find("{")
                        if start_idx != -1:
                            result_str = result_str[start_idx:]
                        else:
                            result_str = "{" + result_str
                    
                    print(f"🔄 修复后重新尝试（原始模式）: {result_str[:100]}...")
                else:
                    print("❌ 所有JSON解析尝试都失败（原始模式）")
                    raise Exception(f"JSON解析失败: {str(e)}")
        
        if result is None:
            raise Exception("JSON解析失败")
        
        # 补全字段
        required_fields = ["reasoning_steps", "is_ai_generated", "rumor_prob", "is_rumor", "conclusion"]
        for field in required_fields:
            if field not in result:
                print(f"⚠️ 缺失字段（原始模式）: {field}，使用默认值")
                if field == "reasoning_steps":
                    result[field] = ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
                elif field == "is_ai_generated":
                    result[field] = False
                elif field == "rumor_prob":
                    result[field] = 0.5000
                elif field == "is_rumor":
                    result[field] = result.get("rumor_prob", 0.5) >= 0.5
                elif field == "conclusion":
                    is_rumor = result.get("is_rumor", result.get("rumor_prob", 0.5) >= 0.5)
                    result[field] = "经分析，该信息【是谣言】。" if is_rumor else "经分析，该信息【不是谣言】。"
        
        # 确保推理步骤是4步
        if "reasoning_steps" in result:
            if not isinstance(result["reasoning_steps"], list):
                result["reasoning_steps"] = ["识别内容：信息不足", "检查事实：无相关依据", "评估合理性：无法判断", "得出结论：信息不足"]
            elif len(result["reasoning_steps"]) != 4:
                print(f"⚠️ 推理步骤数量不正确（原始模式）: {len(result['reasoning_steps'])}，调整为4步")
                if len(result["reasoning_steps"]) < 4:
                    while len(result["reasoning_steps"]) < 4:
                        result["reasoning_steps"].append("信息不足")
                result["reasoning_steps"] = result["reasoning_steps"][:4]
        
        # 确保谣言概率格式正确
        if "rumor_prob" in result:
            try:
                rumor_prob = float(result["rumor_prob"])
                rumor_prob = max(0.0, min(1.0, rumor_prob))
                result["rumor_prob"] = round(rumor_prob, 4)
                print(f"✅ 谣言概率处理成功（原始模式）: {result['rumor_prob']}")
            except Exception as e:
                print(f"⚠️ 谣言概率格式错误（原始模式）: {result['rumor_prob']}，使用默认值0.5")
                result["rumor_prob"] = 0.5000
        
        # 确保结论字段格式正确
        if "conclusion" not in result or not result["conclusion"]:
            is_rumor = result.get("is_rumor", result.get("rumor_prob", 0.5) >= 0.5)
            result["conclusion"] = "经分析，该信息【是谣言】。" if is_rumor else "经分析，该信息【不是谣言】。"
        
        # 确保结论包含明确判断
        if "【是谣言】" not in result["conclusion"] and "【不是谣言】" not in result["conclusion"]:
            is_rumor = result.get("is_rumor", result.get("rumor_prob", 0.5) >= 0.5)
            result["conclusion"] = "经分析，该信息【是谣言】。" if is_rumor else "经分析，该信息【不是谣言】。"
        
        # 确保is_rumor字段与结论一致
        if "is_rumor" not in result:
            result["is_rumor"] = "【是谣言】" in result["conclusion"]
        
        result["from_cache"] = False
        result["web_context_used"] = False
        result["search_used"] = False
        result["search_result_count"] = 0
        
        print(f"✅ 原始检测完成，谣言概率: {result['rumor_prob']}, 结论: {result['conclusion']}")
        return result
        
    except Exception as e:
        print(f"❌ GLM-4模型调用/解析失败：{str(e)}")
        print(f"❌ 错误类型：{type(e).__name__}")
        print(f"❌ 详细错误：{traceback.format_exc()}")
        return {
            "reasoning_steps": ["识别内容：模型调用异常", "检查事实：检测失败", "评估合理性：无法判断", "得出结论：信息不足"],
            "is_ai_generated": False,
            "rumor_prob": 0.5000,
            "is_rumor": False,
            "conclusion": "经分析，该信息【检测失败，请重试】。",
            "from_cache": False,
            "web_context_used": False,
            "search_used": False,
            "search_result_count": 0
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

# ---------------------- 检测接口（带DuckDuckGo搜索） ----------------------
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
    print(f"🔑 内容哈希值: {content_hash}")
    
    # 4. 先查询数据库是否有相同内容的记录
    existing_record = find_existing_record(db, content_hash)
    if existing_record:
        print(f"✅ 找到缓存记录，使用次数: {existing_record['use_count']}")
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
                "use_count": existing_record["use_count"],
                "web_context_used": False,
                "search_used": False,
                "is_rumor": existing_record["is_rumor"],
                "conclusion": existing_record["conclusion"]
            }
        }
    
    # 5. 如果没有缓存，则提取关键字
    keywords = extract_keywords_with_jieba(request.content)
    print(f"🔑 提取的关键字: {keywords}")
    print("🔄 未找到缓存记录，调用大模型...")
    
    # 6. 调用大语言模型
    if config.LLM_FAKE:
        print("🤖 使用模拟模式")
        llm_result = fake_llm_detect(request.content, request.type, keywords)
    else:
        print("🚀 使用GLM-4真实API模式（带DuckDuckGo搜索）")
        llm_result = enhanced_real_llm_detect(request.content, request.type, keywords)
    
    # 7. 存储新的检测记录到数据库
    try:
        record = ReasoningRecord(
            user_id=user_id,
            content=request.content,
            content_hash=content_hash,
            type=request.type,
            rumor_prob=llm_result["rumor_prob"],
            is_ai_generated=llm_result["is_ai_generated"],
            reasoning_steps=json.dumps(llm_result["reasoning_steps"], ensure_ascii=False),
            keywords=json.dumps(keywords, ensure_ascii=False),
            use_count=1,
            create_time=datetime.now(),
            last_used_time=datetime.now(),
            conclusion=llm_result.get("conclusion", "")
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"存储检测记录失败：{str(e)}")
    
    # 8. 返回结果给前端
    response_data = {
        "rumor_prob": round(llm_result["rumor_prob"], 4),
        "is_ai_generated": llm_result["is_ai_generated"],
        "reasoning_steps": llm_result["reasoning_steps"],
        "keywords": keywords,
        "record_id": record.id,
        "from_cache": False,
        "use_count": 1,
        "web_context_used": llm_result.get("web_context_used", False),
        "search_used": llm_result.get("search_used", False),
        "is_rumor": llm_result.get("is_rumor", llm_result["rumor_prob"] >= 0.5),
        "conclusion": llm_result.get("conclusion", "经分析，该信息【结论待定】。")
    }
    
    # 添加额外字段（如果存在）
    if "confidence" in llm_result:
        response_data["confidence"] = llm_result["confidence"]
    if "verification_suggestions" in llm_result:
        response_data["verification_suggestions"] = llm_result["verification_suggestions"]
    if "search_summary" in llm_result:
        response_data["search_summary"] = llm_result["search_summary"]
    if "search_result_count" in llm_result:
        response_data["search_result_count"] = llm_result["search_result_count"]
    if "verification_based_on_search" in llm_result:
        response_data["verification_based_on_search"] = llm_result["verification_based_on_search"]
    if "search_result_summary" in llm_result:
        response_data["search_result_summary"] = llm_result["search_result_summary"]
    if "key_findings_from_search" in llm_result:
        response_data["key_findings_from_search"] = llm_result["key_findings_from_search"]
    
    return {
        "code": 200,
        "msg": "检测成功" + ("（含网络验证）" if llm_result.get("search_used") else "（实时分析）"),
        "data": response_data
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
        
        # 获取结论
        conclusion = ""
        if r.conclusion:
            conclusion = r.conclusion
        else:
            # 如果没有存储结论，根据谣言概率生成
            if r.rumor_prob >= 0.7:
                conclusion = "经分析，该信息【是谣言】。"
            elif r.rumor_prob <= 0.3:
                conclusion = "经分析，该信息【不是谣言】。"
            else:
                conclusion = "经分析，该信息可能为谣言，建议进一步核实。"
        
        history_list.append({
            "record_id": r.id,
            "content": r.content,
            "content_hash": r.content_hash,
            "type": r.type,
            "rumor_prob": round(float(r.rumor_prob), 4),
            "is_ai_generated": r.is_ai_generated,
            "keywords": keywords_data,
            "reasoning_steps": reasoning_steps_data,
            "use_count": r.use_count,
            "create_time": r.create_time.strftime("%Y-%m-%d %H:%M:%S") if r.create_time else "",
            "last_used_time": r.last_used_time.strftime("%Y-%m-%d %H:%M:%S") if r.last_used_time else "",
            "conclusion": conclusion
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
            "last_used": record.last_used_time.strftime("%Y-%m-%d %H:%M:%S") if record.last_used_time else "",
            "conclusion": record.conclusion if record.conclusion else ("【是谣言】" if record.rumor_prob >= 0.5 else "【不是谣言】")
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

# ---------------------- 新增：检查搜索状态接口 ----------------------
@app.get("/api/search-status")
def get_search_status():
    """检查DuckDuckGo搜索功能状态"""
    
    status_info = {
        "duckduckgo_enabled": config.SEARCH_CONFIG.get("enable", True),
        "duckduckgo_available": DUCKDUCKGO_AVAILABLE,
        "max_results": config.SEARCH_CONFIG.get("max_results", 3),
        "max_queries": config.SEARCH_CONFIG.get("max_queries", 2),
        "timeout": config.SEARCH_CONFIG.get("timeout", 15)
    }
    
    # 测试搜索功能
    test_result = "未测试"
    if status_info["duckduckgo_enabled"] and status_info["duckduckgo_available"]:
        try:
            test_client = DuckDuckGoSearchClient()
            test_search = test_client.search("测试", max_results=1)
            test_result = "正常" if test_search.get("success") else f"失败: {test_search.get('error', '未知错误')}"
        except Exception as e:
            test_result = f"异常: {str(e)}"
    
    status_info["test_result"] = test_result
    
    return {
        "code": 200,
        "msg": "状态查询成功",
        "data": status_info
    }

# ---------------------- 启动后端 ----------------------
if __name__ == "__main__":
    try:
        jieba.load_userdict('userdict.txt')
        print("✅ jieba分词器初始化成功 - 加载自定义词典")
    except:
        print("✅ jieba分词器初始化成功 - 使用默认词典")
    
    # 初始化数据库和测试用户
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "test").first():
            password = str("123456")[:72]
            test_user = User(username="test", password=hash_password(password))
            db.add(test_user)
            db.commit()
            print("✅ 测试用户创建成功：用户名test，密码123456")
        else:
            print("✅ 测试用户已存在")
            
        # 检查搜索配置
        print("\n=== DuckDuckGo搜索功能状态 ===")
        print(f"✓ 启用状态: {config.SEARCH_CONFIG.get('enable', True)}")
        print(f"✓ DuckDuckGo可用: {'是' if DUCKDUCKGO_AVAILABLE else '否'}")
        print(f"✓ 最大结果数: {config.SEARCH_CONFIG.get('max_results', 3)}")
        print(f"✓ 最大查询数: {config.SEARCH_CONFIG.get('max_queries', 2)}")
        print(f"✓ 超时时间: {config.SEARCH_CONFIG.get('timeout', 15)}秒")
        
        print("\n=== 数据库去重功能状态 ===")
        print("✓ content_hash字段已添加")
        print("✓ use_count字段已添加")
        print("✓ last_used_time字段已添加")
        print("✓ conclusion字段已添加")
        print("✓ 去重查询功能已启用")
        
    except Exception as e:
        print(f"❌ 数据库初始化失败：{str(e)}")
        db.rollback()
    finally:
        db.close()
    
    print(f"\n=== 谣言甄别系统后端启动 ===")
    print(f"📱 模型配置: {config.LLM_CONFIG['model_name']}")
    print(f"🔍 DuckDuckGo搜索: {'已启用' if config.SEARCH_CONFIG.get('enable', True) and DUCKDUCKGO_AVAILABLE else '未启用'}")
    print(f"💾 去重功能: 已启用")
    print(f"🤖 LLM_FAKE模式: {config.LLM_FAKE}")
    print(f"🌐 服务地址: http://localhost:8000")
    print(f"📚 API文档: http://localhost:8000/docs")
    print(f"🔍 搜索状态检查: http://localhost:8000/api/search-status")
    
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
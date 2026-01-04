import pandas as pd
import re
from datetime import datetime

# ====================== 替换CSV路径 ======================
input_csv = "D:/weibo_data_v1/SocialNet-Weibo-V1/integrated_news_dataset.csv"  # 你的CSV路径
output_csv = "D:/weibo_data_v1/SocialNet-Weibo-V1/cleaned_news.csv"    # 清洗后保存路径
# ========================================================
# 1. 读取CSV
df = pd.read_csv(input_csv, encoding="utf-8")
print("=== 清洗前数据概况 ===")
print(f"总记录数：{len(df)}，总字段数：{len(df.columns)}")
print(f"假新闻数量（label=0）：{len(df[df['label']==0])}")
print(f"真实新闻数量（label=1）：{len(df[df['label']==1])}")
print(f"缺失值统计：\n{df.isnull().sum()[df.isnull().sum()>0]}")
# 2. 缺失值处理（适配你的字段名）
# 核心字段：新闻正文是context，标签是label
df = df.dropna(subset=["context", "label"], axis=0)
# 非核心字段填充
fill_rules = {
    "name": "未知用户",       # 对应你的name字段
    "local": "未知地区",     # 对应你的local字段
    "context": "无内容",     # 兜底
    "comment_num": 0,       # 对应你的comment_num字段
    "like_num": 0,          # 对应你的like_num字段
    "user": "未知用户ID",    # 对应你的user字段
    "retweet": 0            # 对应你的retweet字段
}
for col, val in fill_rules.items():
    if col in df.columns:
        df[col] = df[col].fillna(val)
# 3. 重复值处理
df = df.drop_duplicates(keep="first")
df = df.drop_duplicates(subset=["context"], keep=False)  # 正文是context
# 4. 文本清洗（清洗context字段）
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http[s]?://[^\s]+', '', text)
    text = re.sub(r'@[^\s]+|#[^\s]+#', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''()（）、]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df["clean_context"] = df["context"].apply(clean_text)  # 正文是context
df = df[df["clean_context"] != ""]
# 5. 保存清洗后的数据
df.to_csv(output_csv, index=False, encoding="utf-8")
# 6. 输出结果
print("\n=== 清洗完成 ===")
print(f"总记录数：{len(df)}")
print(f"假新闻数量（label=0）：{len(df[df['label']==0])}")
print(f"真实新闻数量（label=1）：{len(df[df['label']==1])}")
print(f"清洗后文件已保存至：{output_csv}")
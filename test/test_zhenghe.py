import os
import json
import pandas as pd

# 根目录
root_dir = "D:/weibo_data_v1/SocialNet-Weibo-V1" 
news_data = []

# 遍历假新闻和真实新闻文件夹
for folder, label in [("fake_news", 0), ("real_news", 1)]:
    folder_path = os.path.join(root_dir, folder)
    
    # 检查主文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"警告：文件夹不存在 → {folder_path}")
        continue
    
    # 遍历子文件夹
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # 只处理子文件夹（跳过文件）
        if not os.path.isdir(subfolder_path):
            continue
        
        # 拼接news.json的路径（注意是new.json还是news.json，根据实际文件调整）
        news_json_path = os.path.join(subfolder_path, "new.json")
        
        # 检查JSON文件是否存在
        if not os.path.exists(news_json_path):
            print(f"跳过：未找到new.json → {news_json_path}")
            continue
        
        # 读取JSON文件，处理各种异常
        try:
            # 先尝试UTF-8编码读取
            with open(news_json_path, "r", encoding="utf-8") as f:
                file_content = f.read().strip()
                news_content = None
                
                # 处理空文件
                if not file_content:
                    print(f"警告：空文件 → {news_json_path}")
                    continue
                
                # 尝试解析为单行JSON
                try:
                    news_content = json.loads(file_content)
                except json.JSONDecodeError:
                    # 尝试解析为多行JSON（每行一个JSON对象）
                    lines = [line.strip() for line in file_content.split('\n') if line.strip()]
                    news_content = [json.loads(line) for line in lines]
                
                # 处理单条/多条数据，统一添加标签
                if isinstance(news_content, dict):  # 单条新闻
                    news_content["label"] = label
                    news_data.append(news_content)
                    print(f"成功读取：{news_json_path} → 数据条数：1")
                elif isinstance(news_content, list):  # 多条新闻
                    valid_count = 0
                    for item in news_content:
                        if isinstance(item, dict):
                            item["label"] = label
                            news_data.append(item)
                            valid_count += 1
                    print(f"成功读取：{news_json_path} → 数据条数：{valid_count}")
                else:
                    print(f"警告：非字典/列表格式 → {news_json_path}")
            
        except json.JSONDecodeError:
            print(f"错误：JSON格式解析失败 → {news_json_path}")
        except UnicodeDecodeError:
            # 兼容GBK/GB2312编码
            try:
                with open(news_json_path, "r", encoding="gbk") as f:
                    file_content = f.read().strip()
                    if not file_content:
                        print(f"警告：空文件（GBK编码）→ {news_json_path}")
                        continue
                    
                    news_content = json.loads(file_content)
                    if isinstance(news_content, dict):
                        news_content["label"] = label
                        news_data.append(news_content)
                        print(f"成功读取（GBK编码）：{news_json_path} → 数据条数：1")
                    elif isinstance(news_content, list):
                        valid_count = 0
                        for item in news_content:
                            if isinstance(item, dict):
                                item["label"] = label
                                news_data.append(item)
                                valid_count += 1
                        print(f"成功读取（GBK编码）：{news_json_path} → 数据条数：{valid_count}")
            except Exception as e:
                print(f"错误：GBK编码解析也失败 → {news_json_path}，错误：{str(e)}")
        except Exception as e:
            print(f"未知错误 → {news_json_path}，错误信息：{str(e)}")

# 转换为DataFrame（表格形式）
df = pd.DataFrame(news_data)

# 输出结果验证
print("\n===== 读取结果 =====")
print(f"总读取新闻条数：{len(df)}")
print(f"假新闻（0）数量：{len(df[df['label']==0])}")
print(f"真实新闻（1）数量：{len(df[df['label']==1])}")

# 显示数据前5行（如果有数据）
if len(df) > 0:
    print("\n数据前5行：")
    print(df.head())
else:
    print("\n警告：未读取到任何数据！")

# 保存为CSV文件（方便后续处理）
if len(df) > 0:
    save_path = os.path.join(root_dir, "integrated_news_dataset.csv")
    # 使用utf-8-sig编码，避免Excel打开中文乱码
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n数据集已保存至：{save_path}")
else:
    print("\n警告：无数据可保存！")
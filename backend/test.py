# test_rumor_system_enhanced.py
import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os

class RumorSystemTester:
    """谣言检测系统测试器 - 针对您的数据结构优化"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.user_id = None
        
    def login(self, username="test", password="123456"):
        """登录系统"""
        try:
            response = requests.post(
                f"{self.base_url}/api/login",
                json={"username": username, "password": password}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data["code"] == 200:
                    self.token = data["data"]["token"]
                    self.user_id = data["data"]["user_id"]
                    print(f"✅ 登录成功，用户ID: {self.user_id}")
                    return True
            else:
                print(f"❌ 登录失败: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 登录异常: {str(e)}")
            return False
    
    def analyze_excel_structure(self, excel_path):
        """分析Excel文件结构"""
        try:
            df = pd.read_excel(excel_path, nrows=5)  # 只读取前5行分析结构
            print("📊 Excel文件结构分析:")
            print("="*50)
            
            # 显示所有列名
            print("列名列表:")
            for i, col in enumerate(df.columns, 1):
                print(f"  {i:2d}. {col}")
            
            print("\n前3行数据示例:")
            print(df.head(3).to_string())
            
            # 检查关键字段
            text_columns = []
            label_columns = []
            
            for col in df.columns:
                col_lower = col.lower()
                # 检查文本字段
                if any(keyword in col_lower for keyword in ['context', 'text', 'content', 'clean']):
                    text_columns.append(col)
                # 检查标签字段
                if any(keyword in col_lower for keyword in ['label', 'is_rumor', 'rumor']):
                    label_columns.append(col)
            
            print(f"\n🔍 识别到的文本字段: {text_columns}")
            print(f"🔍 识别到的标签字段: {label_columns}")
            
            return True
            
        except Exception as e:
            print(f"❌ 分析文件结构失败: {str(e)}")
            return False
    
    def test_excel_file(self, excel_path, output_path=None, search_enabled=True, 
                       text_column=None, label_column=None, batch_size=50):
        """
        测试Excel文件中的文本
        
        参数:
            excel_path: Excel文件路径
            output_path: 结果输出路径
            search_enabled: 是否启用搜索
            text_column: 指定文本列名（如不指定自动检测）
            label_column: 指定标签列名（如不指定自动检测）
            batch_size: 批量处理数量
        """
        if not os.path.exists(excel_path):
            print(f"❌ 文件不存在: {excel_path}")
            return None
        
        # 1. 分析文件结构
        print("🔍 分析Excel文件结构...")
        self.analyze_excel_structure(excel_path)
        
        # 2. 读取完整数据
        try:
            df = pd.read_excel(excel_path)
            print(f"📊 读取到 {len(df)} 条数据")
            
            # 3. 确定文本列
            if text_column:
                if text_column not in df.columns:
                    print(f"❌ 指定的文本列 '{text_column}' 不存在")
                    return None
                text_col = text_column
            else:
                # 自动检测文本列
                possible_text_cols = ['clean_context', 'context', 'text', 'content']
                for col in possible_text_cols:
                    if col in df.columns:
                        text_col = col
                        print(f"✅ 使用文本列: {text_col}")
                        break
                else:
                    # 如果标准列名都不存在，让用户选择
                    print("❌ 未找到标准文本列名，请从以下列中选择:")
                    for i, col in enumerate(df.columns, 1):
                        print(f"  {i}. {col}")
                    
                    try:
                        choice = int(input("请输入列号: ")) - 1
                        text_col = df.columns[choice]
                        print(f"✅ 选择文本列: {text_col}")
                    except:
                        print("❌ 选择无效")
                        return None
            
            # 4. 确定标签列
            if label_column:
                if label_column not in df.columns:
                    print(f"❌ 指定的标签列 '{label_column}' 不存在")
                    label_col = None
                else:
                    label_col = label_column
            else:
                # 自动检测标签列
                if 'label' in df.columns:
                    label_col = 'label'
                    print(f"✅ 使用标签列: {label_col}")
                else:
                    label_col = None
                    print("ℹ️ 未找到标签列，将只进行检测不计算准确率")
            
            # 5. 检查数据质量
            print("\n📈 数据质量检查:")
            print(f"  文本列 '{text_col}': {df[text_col].notna().sum()} 个非空值")
            if label_col:
                print(f"  标签列 '{label_col}': 唯一值 {df[label_col].unique()[:5]}")
            
            # 统计文本长度
            df['text_length'] = df[text_col].astype(str).apply(len)
            print(f"  平均文本长度: {df['text_length'].mean():.1f} 字符")
            print(f"  最短文本: {df['text_length'].min()} 字符")
            print(f"  最长文本: {df['text_length'].max()} 字符")
            
        except Exception as e:
            print(f"❌ 读取Excel文件失败: {str(e)}")
            return None
        
        # 6. 登录系统
        if not self.login():
            return None
        
        # 7. 准备结果列表
        results = []
        
        # 8. 批量测试
        print(f"\n🚀 开始批量测试，共 {len(df)} 条数据...")
        
        for i in tqdm(range(min(len(df), batch_size)), desc="测试进度"):
            try:
                # 获取文本
                text = str(df.iloc[i][text_col])
                
                if pd.isna(text) or text.strip() == "":
                    print(f"⚠️ 第 {i+1} 行文本为空，跳过")
                    continue
                
                # 获取标签（如果有）
                label = None
                if label_col and not pd.isna(df.iloc[i][label_col]):
                    label = df.iloc[i][label_col]
                
                # 获取其他有用信息
                other_info = {}
                for col in ['name', 'user', 'time', 'local', 'comment_num', 'like_num']:
                    if col in df.columns and not pd.isna(df.iloc[i][col]):
                        other_info[col] = df.iloc[i][col]
                
                # 测试文本（默认使用"其他"类型，因为您的数据可能没有类型列）
                result = self.test_single_text(text, "其他", search_enabled)
                
                if result and result["code"] == 200:
                    # 提取关键信息
                    data = result["data"]
                    result_dict = {
                        "index": i + 1,
                        "original_text": text[:100] + "..." if len(text) > 100 else text,
                        "full_text_length": len(text),
                        "detected_rumor_prob": data.get("rumor_prob", 0),
                        "detected_is_rumor": data.get("is_rumor", False),
                        "detected_conclusion": data.get("conclusion", ""),
                        "from_cache": data.get("from_cache", False),
                        "search_used": data.get("search_used", False),
                        "use_count": data.get("use_count", 1),
                        "confidence": data.get("confidence", "未知"),
                        "test_time": pd.Timestamp.now()
                    }
                    
                    # 添加原始标签（如果有）
                    if label is not None:
                        result_dict["original_label"] = label
                    
                    # 添加其他信息
                    result_dict.update(other_info)
                    
                    results.append(result_dict)
                    
                    # 添加延迟，避免请求过快
                    time.sleep(0.3)
                    
                else:
                    print(f"❌ 第 {i+1} 行测试失败")
                    
            except Exception as e:
                print(f"❌ 第 {i+1} 行测试异常: {str(e)}")
        
        # 9. 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 10. 保存结果
        if output_path:
            if output_path.endswith('.xlsx'):
                results_df.to_excel(output_path, index=False)
            elif output_path.endswith('.csv'):
                results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            else:
                output_path = output_path + '.xlsx'
                results_df.to_excel(output_path, index=False)
            
            print(f"✅ 测试结果已保存到: {output_path}")
        
        # 11. 生成详细测试报告
        self.generate_detailed_report(results_df, df, text_col, label_col)
        
        return results_df
    
    def test_single_text(self, text, text_type="其他", search_enabled=True):
        """测试单条文本"""
        if not self.token:
            print("❌ 请先登录")
            return None
        
        try:
            headers = {"Authorization": f"Bearer {self.token}"}
            
            response = requests.post(
                f"{self.base_url}/api/detect",
                headers=headers,
                json={
                    "content": text,
                    "type": text_type,
                    "search_enabled": search_enabled
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ 检测失败: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 检测异常: {str(e)}")
            return None
    
    def generate_detailed_report(self, results_df, original_df, text_col, label_col):
        """生成详细测试报告"""
        if results_df.empty:
            print("❌ 没有测试结果")
            return
        
        print("\n" + "="*60)
        print("📊 详细测试报告")
        print("="*60)
        
        total_tests = len(results_df)
        
        # 基础统计
        cache_hits = results_df['from_cache'].sum()
        search_used = results_df['search_used'].sum()
        
        print(f"📈 基础统计:")
        print(f"  总测试数量: {total_tests}")
        print(f"  缓存命中率: {cache_hits/total_tests*100:.1f}% ({cache_hits}/{total_tests})")
        print(f"  搜索使用率: {search_used/total_tests*100:.1f}% ({search_used}/{total_tests})")
        
        # 准确率分析（如果有标签）
        if 'original_label' in results_df.columns:
            correct_predictions = 0
            true_positives = 0  # 正确识别的谣言
            false_positives = 0  # 误报（非谣言被识别为谣言）
            true_negatives = 0  # 正确识别的非谣言
            false_negatives = 0  # 漏报（谣言被识别为非谣言）
            
            for _, row in results_df.iterrows():
                detected = row['detected_is_rumor']
                original = bool(row['original_label'])
                
                if detected == original:
                    correct_predictions += 1
                    if detected:  # 正确识别谣言
                        true_positives += 1
                    else:  # 正确识别非谣言
                        true_negatives += 1
                else:
                    if detected and not original:  # 误报
                        false_positives += 1
                    elif not detected and original:  # 漏报
                        false_negatives += 1
            
            accuracy = correct_predictions / total_tests * 100
            
            print(f"\n🎯 准确率分析:")
            print(f"  准确率: {accuracy:.2f}% ({correct_predictions}/{total_tests})")
            print(f"  真正例(TP): {true_positives} - 正确识别谣言")
            print(f"  真反例(TN): {true_negatives} - 正确识别非谣言")
            print(f"  假正例(FP): {false_positives} - 误报（非谣言->谣言）")
            print(f"  假反例(FN): {false_negatives} - 漏报（谣言->非谣言）")
            
            # 计算指标
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives) * 100
                print(f"  精确率: {precision:.2f}%")
            
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives) * 100
                print(f"  召回率: {recall:.2f}%")
        
        # 谣言概率分布
        print(f"\n📊 谣言概率分布:")
        print(f"  平均概率: {results_df['detected_rumor_prob'].mean():.4f}")
        print(f"  中位数: {results_df['detected_rumor_prob'].median():.4f}")
        print(f"  标准差: {results_df['detected_rumor_prob'].std():.4f}")
        
        # 概率区间分布
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['极不可能', '不太可能', '可能', '很可能', '极可能']
        results_df['prob_category'] = pd.cut(results_df['detected_rumor_prob'], bins=bins, labels=labels)
        
        print(f"\n📊 概率区间分布:")
        for category in labels:
            count = (results_df['prob_category'] == category).sum()
            percentage = count / total_tests * 100
            print(f"  {category}: {count} 条 ({percentage:.1f}%)")
        
        # 置信度分布
        if 'confidence' in results_df.columns:
            print(f"\n🔍 置信度分布:")
            confidence_counts = results_df['confidence'].value_counts()
            for conf, count in confidence_counts.items():
                print(f"  {conf}: {count} 条 ({count/total_tests*100:.1f}%)")
        
        # 文本长度与谣言概率的关系
        if 'full_text_length' in results_df.columns:
            print(f"\n📏 文本长度分析:")
            print(f"  平均长度: {results_df['full_text_length'].mean():.1f} 字符")
            
            # 计算相关性
            correlation = results_df['full_text_length'].corr(results_df['detected_rumor_prob'])
            print(f"  长度与谣言概率相关性: {correlation:.3f}")
        
        # 显示最有代表性的结果
        print(f"\n🔍 代表性结果示例:")
        
        # 最高概率的谣言
        if not results_df.empty:
            max_prob = results_df.loc[results_df['detected_rumor_prob'].idxmax()]
            print(f"  最高谣言概率 ({max_prob['detected_rumor_prob']:.4f}):")
            print(f"    {max_prob['original_text']}")
            
            # 最低概率的谣言
            min_prob = results_df.loc[results_df['detected_rumor_prob'].idxmin()]
            print(f"  最低谣言概率 ({min_prob['detected_rumor_prob']:.4f}):")
            print(f"    {min_prob['original_text']}")
            
            # 缓存命中的例子
            cache_hit_examples = results_df[results_df['from_cache'] == True]
            if not cache_hit_examples.empty:
                example = cache_hit_examples.iloc[0]
                print(f"  缓存命中示例 (使用次数: {example['use_count']}):")
                print(f"    {example['original_text']}")

# 使用示例
def main():
    # 创建测试器
    tester = RumorSystemTester()
    
    # Excel文件路径
    excel_file = "D:/rumor/完整 Excel 文件.xlsx"  # 替换为您的文件路径
    
    if os.path.exists(excel_file):
        # 分析文件结构
        tester.analyze_excel_structure(excel_file)
        
        # 询问用户选择
        print("\n🎯 请选择测试选项:")
        print("1. 使用 clean_context 作为文本")
        print("2. 使用 context 作为文本")
        print("3. 手动指定列名")
        
        choice = input("请输入选项 (1-3): ").strip()
        
        text_column = None
        label_column = 'label' if 'label' in pd.read_excel(excel_file, nrows=1).columns else None
        
        if choice == '1':
            text_column = 'clean_context'
        elif choice == '2':
            text_column = 'context'
        elif choice == '3':
            text_column = input("请输入文本列名: ").strip()
            label_input = input("请输入标签列名 (留空则自动检测): ").strip()
            if label_input:
                label_column = label_input
        
        # 询问是否启用搜索
        search_choice = input("启用搜索功能? (y/n, 默认y): ").strip().lower()
        search_enabled = search_choice != 'n'
        
        # 询问批量大小
        try:
            batch_size = int(input(f"测试数量 (默认50, 最大{len(pd.read_excel(excel_file))}): ") or "50")
        except:
            batch_size = 50
        
        # 运行测试
        results = tester.test_excel_file(
            excel_path=excel_file,
            output_path="test_results.xlsx",
            search_enabled=search_enabled,
            text_column=text_column,
            label_column=label_column,
            batch_size=batch_size
        )
        
        if results is not None:
            print("\n✅ 测试完成！")
    else:
        print(f"❌ 文件不存在: {excel_file}")
        print("请将您的Excel文件放在当前目录下")

if __name__ == "__main__":
    main()
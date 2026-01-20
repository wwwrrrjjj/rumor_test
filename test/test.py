# backend/test_ddg_final.py
from ddgs import DDGS
import time

def test_ddg_search():
    """测试DuckDuckGo搜索功能"""
    print("=== DuckDuckGo详细搜索测试 ===")
    
    test_queries = [
        "疫情谣言 官方辟谣",      # 中文查询
        "apple health benefits",  # 英文查询
        "新冠病毒 最新消息",      # 中文时事
        "weather today"           # 英文简单查询
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}: '{query}'")
        
        try:
            with DDGS() as ddgs:
                results = []
                count = 0
                
                # 尝试获取结果
                for r in ddgs.text(query, max_results=3):
                    results.append(r)
                    count += 1
                
                if results:
                    print(f"✅ 成功找到 {count} 个结果")
                    for j, result in enumerate(results[:2], 1):
                        print(f"  结果 {j}:")
                        print(f"    标题: {result.get('title', '无标题')}")
                        print(f"    内容: {result.get('body', '无内容')[:60]}...")
                        print(f"    链接: {result.get('href', '无链接')[:50]}")
                else:
                    print("⚠ 无搜索结果")
                    
        except Exception as e:
            print(f"❌ 搜索失败: {str(e)}")
        
        # 延迟避免请求过快
        if i < len(test_queries):
            time.sleep(1)
    
    return len(results) > 0

if __name__ == "__main__":
    success = test_ddg_search()
    
    if success:
        print("\n🎉 DuckDuckGo搜索功能完全正常！")
        print("可以在谣言检测系统中使用网络搜索验证")
    else:
        print("\n⚠ DuckDuckGo可能无法返回有效结果")
        print("建议：")
        print("1. 检查网络代理设置")
        print("2. 尝试使用VPN")
        print("3. 或使用纯大模型模式")
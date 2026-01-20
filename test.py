import pymysql
from config import DB_CONFIG  # 导入您的配置

def delete_rumor_record():
    # 连接数据库
    connection = pymysql.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset='utf8mb4'
    )
    
    try:
        with connection.cursor() as cursor:
            # 1. 先查看要删除的记录
            search_sql = """
            SELECT id, user_id, content, rumor_prob, conclusion, use_count, create_time 
            FROM reasoning_records 
            WHERE content LIKE %s
            """
            cursor.execute(search_sql, ('%新冠病毒可以通过5G网络传播%',))
            results = cursor.fetchall()
            
            print("找到以下记录：")
            for row in results:
                print(f"ID: {row[0]}, 内容: {row[2]}, 结论: {row[4]}, 使用次数: {row[5]}")
            
            if not results:
                print("未找到相关记录")
                return
            
            # 2. 确认是否删除
            confirm = input(f"\n确认删除以上 {len(results)} 条记录？(y/n): ")
            if confirm.lower() == 'y':
                # 删除记录
                delete_sql = """
                DELETE FROM reasoning_records 
                WHERE content LIKE %s
                """
                cursor.execute(delete_sql, ('%新冠病毒可以通过5G网络传播%',))
                connection.commit()
                print(f"已删除 {cursor.rowcount} 条记录")
            else:
                print("取消删除")
                
    finally:
        connection.close()

if __name__ == "__main__":
    delete_rumor_record()
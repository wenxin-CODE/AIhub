from vector import get_chroma_vectorstore
from search import search_relevant_info_in_chroma,search_relevant_info

def getInfo(query):
    vectorstore = get_chroma_vectorstore()
    # 从Chroma数据库中搜索与用户问题相关性最高的3条向量
    search_info = search_relevant_info_in_chroma(query, vectorstore, return_scores=True)
    threshold = 1.0
    search_results = []
    
    # 处理search_relevant_info_in_chroma返回的字典列表格式
    if search_info:
        print("========使用本地数据======")
        for item in search_info:
            # 检查item是否为字典且包含必要的键
            if isinstance(item, dict) and 'content' in item and 'similarity_score' in item:
                score = item['similarity_score']
                if score <= threshold:
                    search_results.append((item['content'], score))
    
    if len(search_results) == 0:
        print("========本地数据不符合要求，使用网页数据======")
        search_results = search_relevant_info(query)
    
    return search_results

    
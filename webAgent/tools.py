from vector import get_chroma_vectorstore
from search import search_relevant_info_in_chroma,search_relevant_info

def getInfo(query):
    vectorstore = get_chroma_vectorstore()
    # 从Chroma数据库中搜索与用户问题相关性最高的3条向量
    search_info = search_relevant_info_in_chroma(query, vectorstore, return_scores=True)
    threshold = 1.0
    search_results = [
        (doc, score) for doc, score in search_info
        if score >= threshold
    ]
    if search_results.__len__() == 0:
        search_results = search_relevant_info(query)
    
    return search_results

    
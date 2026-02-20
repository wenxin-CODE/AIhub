from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import dotenv
import os

dotenv.load_dotenv('../.env')
os.environ["TAVILY_API_KEY"] = "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"

# 初始化TavilySearch工具
tavily_search = TavilySearch(k=3)  # k=3表示返回3条结果

def search_relevant_info(user_question):
    """
    根据用户问题使用TavilySearch搜索相关信息
    
    Args:
        user_question: 用户的问题
        
    Returns:
        list: 包含3条关联度最高的搜索结果
    """
    try:
        # 执行搜索
        search_results = tavily_search.run(user_question)
        # print(search_results)
        return search_results
    except Exception as e:
        print(f"搜索过程中出错: {e}")
        return []

# 使用max_marginal_relevance_search函数在chroma向量数据库中搜索与query相关性最高的3条向量
def search_relevant_info_in_chroma(query, vectorstore, k=3, return_scores=False):
    """
    在Chroma向量数据库中搜索与query相关性最高的k条向量
    
    Args:
        query: 搜索查询
        vectorstore: Chroma向量数据库实例
        k: 返回的向量数量，默认3条
        return_scores: 是否返回相似度分数，默认False
        
    Returns:
        list: 包含k条相关性最高的向量文档
              如果return_scores为True，返回格式为[{"content": str, "metadata": dict, "similarity_score": float}, ...]
    """
    try:
        if return_scores:
            # 执行搜索并返回相似度分数
            search_results = vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            # 转换为更清晰的格式
            formatted_results = []
            for doc, score in search_results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
            return formatted_results
        else:
            # 保持向后兼容，只返回文档
            search_results = vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=20  # 先获取20条向量，再筛选出k条
            )
            return search_results
    except Exception as e:
        print(f"在Chroma数据库中搜索时出错: {e}")
        return []

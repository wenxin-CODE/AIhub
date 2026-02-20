import os
import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 加载环境变量
dotenv.load_dotenv('../.env')


def read_pdf_and_split(pdf_path):
    """
    读取指定路径下的所有PDF文件并完成文本切分
    
    Args:
        pdf_path: PDF文件路径或包含PDF文件的文件夹路径
        
    Returns:
        list: 所有PDF文件切分后的文本片段列表
    """
    combined_chunks = []
    
    try:
        # 检查路径是否存在
        if not os.path.exists(pdf_path):
            print(f"路径不存在: {pdf_path}")
            return []
        
        # 确定要处理的PDF文件列表
        pdf_files = []
        if os.path.isdir(pdf_path):
            # 如果是目录，获取所有.pdf文件
            print(f"正在处理目录: {pdf_path}")
            for file in os.listdir(pdf_path):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(pdf_path, file))
            
            if not pdf_files:
                print(f"目录中没有找到PDF文件: {pdf_path}")
                return []
            print(f"找到 {len(pdf_files)} 个PDF文件")
        else:
            # 如果是文件，检查是否为.pdf文件
            if pdf_path.lower().endswith('.pdf'):
                pdf_files.append(pdf_path)
                print(f"正在处理单个PDF文件: {pdf_path}")
            else:
                print(f"文件不是PDF格式: {pdf_path}")
                return []
        
        # 配置文本切分器
        # chunk_size: 每个文本片段的大小
        # chunk_overlap: 片段之间的重叠部分大小，有助于保持上下文连续性
        # length_function: 用于计算文本长度的函数
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # 每个片段1000个字符
            chunk_overlap=200,      # 重叠200个字符
            length_function=len     # 使用Python内置的len函数计算长度
        )
        
        # 处理每个PDF文件
        total_files_processed = 0
        for pdf_file in pdf_files:
            try:
                print(f"\n正在处理文件: {os.path.basename(pdf_file)}")
                # 读取PDF文件
                loader = PyPDFLoader(pdf_file)
                
                # 提取所有页面的文本
                docs = loader.load()
                print(f"  提取到 {len(docs)} 页文本")
                
                # 文本切分
                chunks = text_splitter.split_documents(docs)
                print(f"  切分完成，得到 {len(chunks)} 个文本片段")
                
                # 将切分结果添加到总列表
                combined_chunks.extend(chunks)
                total_files_processed += 1
                
            except Exception as e:
                print(f"  处理文件 {os.path.basename(pdf_file)} 时出错: {e}")
                # 继续处理下一个文件
                continue
        
        # 输出处理结果
        print(f"\nPDF文件处理完成:")
        print(f"- 成功处理 {total_files_processed} 个文件")
        print(f"- 总共得到 {len(combined_chunks)} 个文本片段")
        
        return combined_chunks
        
    except Exception as e:
        print(f"读取PDF文件时出错: {e}")
        return []


def vectorize_text(text_chunks):
    """
    将切分后的文本向量化
    
    Args:
        text_chunks: 切分后的文本片段列表
        
    Returns:
        list: 向量化后的文本片段列表
    """
    try:
        # 初始化嵌入模型
        embeddings = OllamaEmbeddings(
            model="qwen3-embedding:0.6b",       # 确保这个模型已在 Ollama 中 pull 过
            base_url="http://localhost:11434",  # 正确写法：不要加 /v1
            # api_key="ollama",  # 如需要兼容某些接口可保留
        )
        
        print("文本向量化中...")
        # 注意：这里我们不直接计算嵌入，而是返回文本片段，
        # 因为Chroma会在存储时自动计算嵌入
        return text_chunks
    except Exception as e:
        print(f"文本向量化时出错: {e}")
        return []


def store_vectors_to_chroma(text_chunks, collection_name="pdf_documents"):
    """
    将向量化后的文本存储到Chroma向量数据库中
    
    Args:
        text_chunks: 切分后的文本片段列表
        collection_name: 集合名称，默认为"pdf_documents"
        
    Returns:
        Chroma: Chroma向量数据库实例
    """
    try:
        # 初始化嵌入模型
        embeddings = OllamaEmbeddings(
            model="qwen3-embedding:0.6b",       # 确保这个模型已在 Ollama 中 pull 过
            base_url="http://localhost:11434",  # 正确写法：不要加 /v1
            # api_key="ollama",  # 如需要兼容某些接口可保留
        )
        
        # 创建Chroma向量数据库并存储文本
        vectorstore = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory="./chroma_db"
        )
        
        # 持久化存储
        # vectorstore.persist()
        print(f"文本已成功存储到Chroma向量数据库，集合名称: {collection_name}")
        print(f"共存储了{len(text_chunks)}个文本片段")
        return vectorstore
    except Exception as e:
        print(f"存储向量到Chroma数据库时出错: {e}")
        return None


def get_chroma_vectorstore(collection_name="pdf_documents", persist_directory="./chroma_db"):
    """
    获取本地保存的Chroma向量数据库
    
    Args:
        collection_name: 集合名称，默认为"pdf_documents"
        persist_directory: 存储路径，默认为"./chroma_db"
        
    Returns:
        Chroma: Chroma向量数据库实例
    """
    try:
        # 初始化嵌入模型（与存储时使用的模型相同）
        embeddings = OllamaEmbeddings(
            model="qwen3-embedding:0.6b",       # 确保这个模型已在 Ollama 中 pull 过
            base_url="http://localhost:11434",  # 正确写法：不要加 /v1
            # api_key="ollama",  # 如需要兼容某些接口可保留
        )
        
        # 加载本地存储的Chroma向量数据库
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # 获取存储的文档数量
        count = vectorstore._collection.count()
        print(f"成功加载Chroma向量数据库，集合名称: {collection_name}")
        print(f"数据库中存储了{count}个文本片段")
        return vectorstore
    except Exception as e:
        print(f"加载Chroma向量数据库时出错: {e}")
        return None


# 示例用法
if __name__ == "__main__":
    # 示例PDF文件路径
    # pdf_path = "./rawData/"
    
    # # 读取PDF并切分文本
    # text_chunks = read_pdf_and_split(pdf_path)
    
    # if text_chunks:
    #     # 存储到Chroma数据库
    #     store_vectors_to_chroma(text_chunks)
    # else:
    #     print("\n提示: 请将PDF文件放在正确的位置，或修改pdf_path变量指向你的PDF文件路径。")
    from search import search_relevant_info,search_relevant_info_in_chroma
    # 示例：加载本地存储的向量数据库
    print("\n尝试加载本地存储的向量数据库...")
    vectorstore = get_chroma_vectorstore()
    if vectorstore:
        # 执行相似度搜索
        query = "MySQL的事务有哪几种隔离级别"
        print(f"\n执行相似度搜索: {query}")
        results = search_relevant_info_in_chroma(query, vectorstore, return_scores=True)
        print(f"搜索结果数量: {len(results)}")
        if results:
            print("\n搜索结果预览:")
            for i, result in enumerate(results, 1):
                # content_preview = result.page_content[:100] + "..." if len(result.page_content) > 100 else result.page_content
                print(f"{i}. {result}")
    else:
        print("\n提示: 向量数据库不存在或无法加载。请先成功处理PDF文件后再尝试。")

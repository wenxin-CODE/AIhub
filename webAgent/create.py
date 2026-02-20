import dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate

def create_chat():
    dotenv.load_dotenv('../.env')

    os.environ['ZHIPUAI_API_KEY'] = os.getenv('GML_API_KEY')
    os.environ['ZHIPUAI_BASE_URL'] = os.getenv('GML_BASE_URL')

    chat = ChatZhipuAI(
        model="GLM-4-Flash",
    )

    return chat

def create_prompt(role,user_question):
    prePrompt = SystemMessage(
        content=f"你是一个专业的{role}，你的任务是回答用户下面的问题。"
    )
    # 生成一段提示词chat_prompt，其中system_message是系统消息，除此之外，还包含用户的问题
    chat_prompt = ChatPromptTemplate.from_messages([
        prePrompt,
        HumanMessage(content=user_question)
    ])
    return chat_prompt

def create_search_prompt(role, user_question, search_results):
    """
    将用户问题和搜索结果合并成一条标准的ChatPromptTemplate大语言模型提示词
    
    Args:
        role: 用户需要的职业人士
        user_question: 用户的问题
        search_results: 搜索结果列表
        
    Returns:
        ChatPromptTemplate: 合并后的提示词模板
    """
    # 构建系统消息
    system_message = SystemMessagePromptTemplate.from_template(
        f"你是一个专业的{role}，需要根据用户的问题和提供的搜索结果来生成详细、准确的回答。\n"
        "请参考以下搜索结果：\n"
        "{search_results}\n"
        "回答时要：\n"
        "1. 基于搜索结果提供准确的信息\n"
        "2. 保持回答的连贯性和逻辑性\n"
        "3. 语言自然，易于理解\n"
        "4. 直接回答用户的问题，不要有引言或开场白\n"
        "5. 参考文献部分必须另起一行展示，与正文之间有明显分隔\n"
        "6. 每条参考文献必须独占一行，格式为：序号. 标题 - 来源 \n"
        "7. 确保参考文献之间有清晰的换行分隔"
    )
    
    # 构建人类消息
    human_message = HumanMessagePromptTemplate.from_template(
        "用户问题：{user_question}"
    )
    
    # 创建并返回ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        human_message
    ])
    
    return chat_prompt
    
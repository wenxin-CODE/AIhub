import os
import json
import dotenv
from datetime import datetime
from flask import request
from langgraph.graph import StateGraph, START
# from langgraph.runtime import LangGraphRunnable
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from create import create_chat,create_prompt,create_search_prompt
from tools import getInfo

# 加载环境变量
dotenv.load_dotenv('../.env')


class SessionManager:
    """
    会话管理类，基于langgraph框架实现
    
    功能：
    1. 集成GLM-4-Flash大模型API
    2. 管理会话历史记录
    3. 支持上下文传递
    4. 会话状态管理
    5. 会话持久化
    """
    
    def __init__(self, model_name="GLM-4-Flash", session_dir="./sessions", system_prompt=None):
        """
        初始化会话管理器
        
        Args:
            model_name: 模型名称，默认为"GLM-4-Flash"
            session_dir: 会话存储目录，默认为"./sessions"
            system_prompt: 系统提示词，默认为None
        """
        self.model_name = model_name
        self.session_dir = session_dir
        self.system_prompt = system_prompt or "你是一个专业的AI助手，能够友好、准确地回答用户的问题。"
        
        # 创建会话存储目录
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 初始化大模型
        self.llm = self._initialize_llm()
        
        # 存储活跃会话
        self.active_sessions = {}
        
    def _initialize_llm(self):
        """
        初始化大模型
        
        Returns:
            ChatZhipuAI: 初始化后的大模型实例
        """
        try:
            os.environ['ZHIPUAI_API_KEY'] = os.getenv('GML_API_KEY')
            os.environ['ZHIPUAI_BASE_URL'] = os.getenv('GML_BASE_URL')
            
            llm = ChatZhipuAI(
                model=self.model_name,
                temperature=0.7,
                max_tokens=1024
            )
            return llm
        except Exception as e:
            print(f"初始化大模型时出错: {e}")
            raise
    
    def _create_graph(self):
        """
        创建langgraph状态图
        
        Returns:
            LangGraphRunnable: 初始化后的状态图实例
        """
        # 定义处理节点
        def process_message(state):
            """
            处理消息节点
            """
            try:
                # 确保state是字典格式
                if not isinstance(state, dict):
                    # 如果不是字典，尝试转换为字典
                    state = dict(state)
                
                # 构建消息列表，包括系统提示和历史消息
                messages = []
                
                # 添加系统提示
                if self.system_prompt:
                    messages.append(SystemMessage(content=self.system_prompt))
                
                # 添加历史消息
                messages.extend(state.get("messages", []))
                
                # 调用大模型生成回复
                response = self.llm.invoke(messages)
                
                # 更新状态
                new_state = state.copy() if hasattr(state, 'copy') else dict(state)
                new_state["messages"] = state.get("messages", []) + [response]
                
                # 保存会话
                session_id = state.get("session_id", "")
                if session_id:
                    self._save_session(session_id, new_state["messages"])
                
                return new_state
            except Exception as e:
                print(f"处理消息时出错: {e}")
                # 创建错误状态
                error_state = state.copy() if hasattr(state, 'copy') else dict(state)
                error_state["messages"] = state.get("messages", []) + [AIMessage(content=f"抱歉，处理您的请求时出错: {e}")]
                return error_state
        
        # 创建状态图 - 使用字典作为状态类型
        # 对于langgraph，我们可以直接使用dict作为状态类型
        graph = StateGraph(dict)
        graph.add_node("process_message", process_message)
        graph.add_edge(START, "process_message")
        
        return graph.compile()
    
    def create_session(self, session_id=None):
        """
        创建新会话
        
        Args:
            session_id: 会话ID，默认为None（自动生成）
            
        Returns:
            str: 会话ID
        """
        if not session_id:
            # 自动生成会话ID
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 初始化会话状态
        self.active_sessions[session_id] = {
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "graph": self._create_graph()
        }
        
        # 保存空会话
        self._save_session(session_id, [])
        
        print(f"创建新会话: {session_id}")
        return session_id
    
    def send_message(self, session_id, message):
        """
        发送消息到指定会话
        
        Args:
            session_id: 会话ID
            message: 用户消息内容
            
        Returns:
            str: 大模型回复内容
        """
        try:
            # 检查会话是否存在
            if session_id not in self.active_sessions:
                # 尝试从文件加载会话
                session_data = self._load_session(session_id)
                if session_data:
                    self.active_sessions[session_id] = {
                        "messages": session_data,
                        "created_at": datetime.now().isoformat(),
                        "graph": self._create_graph()
                    }
                else:
                    # 创建新会话
                    self.create_session(session_id)
            
            # 获取会话状态
            session = self.active_sessions[session_id]
            
            # 添加用户消息
            # user_message = HumanMessage(content=message)
            chat = create_chat()
            role = chat.invoke(f"请根据用户问题，推断用户可能需要什么职业的人士来进行解答：{message}。回答结果只包含职业即可").content
            search_results = getInfo(message)
            chat_prompt = create_search_prompt(role,message,search_results)
            # 正确处理chat_prompt.format_messages()的返回值
            # format_messages返回的是消息对象列表，我们需要获取其内容
            formatted_messages = chat_prompt.format_messages(user_question=message, search_results=search_results)
            # 构建一个包含所有消息内容的字符串
            prompt_content = ""
            for msg in formatted_messages:
                if hasattr(msg, 'content'):
                    prompt_content += msg.content + "\n"
            # 使用字符串作为HumanMessage的content
            user_message = HumanMessage(content=prompt_content)
            session["messages"].append(user_message)
            
            # 执行状态图
            state = {
                "messages": session["messages"],
                "session_id": session_id
            }
            
            result = session["graph"].invoke(state)
            
            # 更新会话状态
            self.active_sessions[session_id]["messages"] = result["messages"]
            
            # 返回最新的AI回复
            if result["messages"] and isinstance(result["messages"][-1], AIMessage):
                return result["messages"][-1].content
            else:
                return "抱歉，未能生成回复。"
                
        except Exception as e:
            print(f"发送消息时出错: {e}")
            return f"抱歉，处理您的请求时出错: {e}"
    
    def get_history(self, session_id):
        """
        获取会话历史记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            list: 会话历史消息列表
        """
        try:
            # 检查会话是否在活跃会话中
            if session_id in self.active_sessions:
                return self.active_sessions[session_id]["messages"]
            else:
                # 尝试从文件加载
                return self._load_session(session_id)
        except Exception as e:
            print(f"获取历史记录时出错: {e}")
            return []
    
    def _save_session(self, session_id, messages):
        """
        保存会话到文件
        
        Args:
            session_id: 会话ID
            messages: 消息列表
        """
        try:
            # 转换消息为可序列化格式
            serializable_messages = []
            for msg in messages:
                msg_dict = {
                    "type": msg.__class__.__name__,
                    "content": msg.content,
                    "additional_kwargs": msg.additional_kwargs,
                    "response_metadata": msg.response_metadata
                }
                serializable_messages.append(msg_dict)
            
            # 保存到文件
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_messages, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存会话时出错: {e}")
    
    def _load_session(self, session_id):
        """
        从文件加载会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            list: 消息列表
        """
        try:
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            if not os.path.exists(session_file):
                return []
            
            # 从文件加载
            with open(session_file, 'r', encoding='utf-8') as f:
                serializable_messages = json.load(f)
            
            # 转换为消息对象
            messages = []
            for msg_dict in serializable_messages:
                msg_type = msg_dict["type"]
                content = msg_dict["content"]
                additional_kwargs = msg_dict.get("additional_kwargs", {})
                response_metadata = msg_dict.get("response_metadata", {})
                
                if msg_type == "HumanMessage":
                    msg = HumanMessage(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        response_metadata=response_metadata
                    )
                elif msg_type == "AIMessage":
                    msg = AIMessage(
                        content=content,
                        additional_kwargs=additional_kwargs,
                        response_metadata=response_metadata
                    )
                elif msg_type == "SystemMessage":
                    msg = SystemMessage(
                        content=content,
                        additional_kwargs=additional_kwargs
                    )
                else:
                    continue
                
                messages.append(msg)
            
            return messages
            
        except Exception as e:
            print(f"加载会话时出错: {e}")
            return []
    
    def list_sessions(self):
        """
        列出所有会话
        
        Returns:
            list: 会话ID列表
        """
        try:
            sessions = []
            # 从活跃会话中获取
            sessions.extend(self.active_sessions.keys())
            
            # 从文件中获取
            for file in os.listdir(self.session_dir):
                if file.endswith('.json'):
                    session_id = file[:-5]  # 移除.json后缀
                    if session_id not in sessions:
                        sessions.append(session_id)
            
            return sessions
        except Exception as e:
            print(f"列出会话时出错: {e}")
            return []
    
    def delete_session(self, session_id):
        """
        删除会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 是否删除成功
        """
        try:
            # 从活跃会话中删除
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # 从文件中删除
            session_file = os.path.join(self.session_dir, f"{session_id}.json")
            if os.path.exists(session_file):
                os.remove(session_file)
            
            print(f"删除会话: {session_id}")
            return True
        except Exception as e:
            print(f"删除会话时出错: {e}")
            return False


# 示例用法
if __name__ == "__main__":
    try:
        # 初始化会话管理器
        session_manager = SessionManager()
        
        # 创建新会话
        session_id = session_manager.create_session()
        print(f"创建会话: {session_id}")
        
        # 发送消息
        response = session_manager.send_message(session_id, "软件开发常用的mysql数据库，其事务有几种隔离级别？")
        print(f"AI回复: {response}")
        
        # 继续对话
        response = session_manager.send_message(session_id, "这几种隔离级别分别是如何实现的")
        print(f"AI回复: {response}")
        
        # 获取历史记录
        history = session_manager.get_history(session_id)
        print(f"\n会话历史:")
        for msg in history:
            print(f"{msg.__class__.__name__}: {msg.content[:50]}..." if len(msg.content) > 50 else f"{msg.__class__.__name__}: {msg.content}")
        
        # 列出所有会话
        sessions = session_manager.list_sessions()
        print(f"\n所有会话: {sessions}")
        
    except Exception as e:
        print(f"示例运行出错: {e}")

import os
import json
import hashlib
import dotenv
from datetime import datetime
from flask import request
from langgraph.graph import StateGraph, START
# from langgraph.runtime import LangGraphRunnable
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
        
        # 存储客户端到会话的映射
        self.client_session_map = {}
        
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
    
    def _get_client_identifier(self):
        """
        从请求对象生成客户端标识符
        
        Returns:
            str: 客户端标识符
        """
        try:
            # 尝试从请求中获取客户端信息
            client_info = []
            
            # 获取用户代理
            user_agent = request.headers.get('User-Agent', '')
            client_info.append(user_agent)
            
            # 获取客户端IP
            client_ip = request.remote_addr or ''
            client_info.append(client_ip)
            
            # 获取Cookie信息（如果有）
            session_cookie = request.cookies.get('session_id', '')
            if session_cookie:
                client_info.append(session_cookie)
            
            # 生成唯一标识符
            client_string = '|'.join(client_info)
            client_hash = hashlib.md5(client_string.encode()).hexdigest()
            
            return client_hash
        except Exception as e:
            print(f"生成客户端标识符时出错: {e}")
            # 如果无法获取请求信息，返回一个临时标识符
            return f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
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
        创建新会话或重用现有会话
        
        根据请求参数判断是否来自同一个客户端，如果是同一个客户端直接使用之前生成的session，否则生成新的session
        
        Args:
            session_id: 会话ID，默认为None（自动生成）
            
        Returns:
            str: 会话ID
        """
        # 获取客户端标识符
        client_id = self._get_client_identifier()
        
        # 检查客户端是否已有会话
        if client_id in self.client_session_map:
            # 重用现有会话
            existing_session_id = self.client_session_map[client_id]
            print(f"重用现有会话: {existing_session_id} 对应客户端: {client_id}")
            
            # 检查会话是否仍然存在
            if existing_session_id not in self.active_sessions:
                # 尝试从文件加载会话
                session_data = self._load_session(existing_session_id)
                if session_data:
                    self.active_sessions[existing_session_id] = {
                        "messages": session_data,
                        "created_at": datetime.now().isoformat(),
                        "graph": self._create_graph()
                    }
                else:
                    # 如果会话不存在，创建新会话
                    if not session_id:
                        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    
                    # 初始化会话状态
                    self.active_sessions[session_id] = {
                        "messages": [],
                        "created_at": datetime.now().isoformat(),
                        "graph": self._create_graph()
                    }
                    
                    # 更新客户端会话映射
                    self.client_session_map[client_id] = session_id
                    
                    # 保存空会话
                    self._save_session(session_id, [])
                    
                    print(f"创建新会话: {session_id} 对应客户端: {client_id}")
                    return session_id
            
            return existing_session_id
        
        # 如果客户端没有现有会话，创建新会话
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 初始化会话状态
        self.active_sessions[session_id] = {
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "graph": self._create_graph()
        }
        
        # 更新客户端会话映射
        self.client_session_map[client_id] = session_id
        
        # 保存空会话
        self._save_session(session_id, [])
        
        print(f"创建新会话: {session_id} 对应客户端: {client_id}")
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
            user_message = HumanMessage(content=message)
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
            
            # 从客户端会话映射中删除
            # 查找并删除对应的客户端映射
            clients_to_remove = []
            for client_id, sid in self.client_session_map.items():
                if sid == session_id:
                    clients_to_remove.append(client_id)
            
            for client_id in clients_to_remove:
                del self.client_session_map[client_id]
                print(f"删除客户端映射: {client_id}")
            
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
        response = session_manager.send_message(session_id, "《哈姆雷特》的作者是谁")
        print(f"AI回复: {response}")
        
        # 继续对话
        response = session_manager.send_message(session_id, "他的喜剧代表作品有哪些")
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

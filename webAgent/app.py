from flask import Flask, request, jsonify
from flask_cors import CORS
from create import create_chat,create_prompt,create_search_prompt
from search import search_relevant_info,search_relevant_info_in_chroma
from vector import get_chroma_vectorstore
from tools import getInfo
from chat import SessionManager
import threading
import uuid
import logging
import time
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局变量
# 存储IP地址与session_id的映射
ip_session_map = {}
# 线程锁，确保线程安全
ip_session_lock = threading.RLock()
# SessionManager实例
session_manager = None

# 初始化SessionManager
try:
    session_manager = SessionManager()
    logger.info("SessionManager初始化成功")
except Exception as e:
    logger.error(f"SessionManager初始化失败: {e}")
    session_manager = None

app = Flask(__name__)
CORS(app)  # 允许跨域请求，以便Vue前端可以访问

# 测试路由
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({"message": "Hello from Flask!"})

# 接收前端POST请求的路由
@app.route('/vueflasks',methods=['POST','GET'])
def vueflasks():
    chat = create_chat()
    if request.method == 'POST':
        # 获取vue中传递的值
        GetMSG = request.get_data(as_text=True)
        role = chat.invoke(f"请根据用户问题，推断用户可能需要什么职业的人士来进行解答：{GetMSG}。回答结果只包含职业即可").content
        # print(role)
        search_results = getInfo(GetMSG)
        chat_prompt = create_search_prompt(role,GetMSG,search_results)
        msg = chat.invoke(chat_prompt.format_messages(user_question=GetMSG, search_results=search_results)).content
        return jsonify(msg)
    else:
        return 'defeat'

@app.route('/vueflask',methods=['POST','GET'])
def vueflask():
    """
    处理前端Vue项目的请求
    - 基于IP地址管理会话
    - 使用SessionManager处理消息
    """
    try:
        if request.method == 'POST':
            # 获取vue中传递的值
            GetMSG = request.get_data(as_text=True)
            logger.info(f"接收到消息: {GetMSG[:100]}..." if len(GetMSG) > 100 else f"接收到消息: {GetMSG}")
            
            # 检查SessionManager是否初始化成功
            if session_manager is None:
                logger.error("SessionManager未初始化，无法处理请求")
                return jsonify({"error": "服务未初始化，请稍后再试"})
            
            # 从request中提取客户端IP地址
            client_ip = request.remote_addr
            logger.info(f"客户端IP地址: {client_ip}")
            
            # 线程安全地获取或生成session_id
            with ip_session_lock:
                if client_ip in ip_session_map:
                    session_id = ip_session_map[client_ip]
                    logger.info(f"IP {client_ip} 已存在会话: {session_id}")
                else:
                    # 生成新的session_id
                    # 使用UUID结合时间戳和IP地址生成唯一且安全的session_id
                    # session_id = f"session_{uuid.uuid4()}_{int(time.time())}_{hash(client_ip) % 10000}"
                    session_id = session_manager.create_session()
                    ip_session_map[client_ip] = session_id
                    logger.info(f"为IP {client_ip} 创建新会话: {session_id}")
            
            # 使用获取或生成的session_id调用SessionManager的send_message方法
            try:
                response = session_manager.send_message(session_id, GetMSG)
                logger.info(f"消息处理成功，会话ID: {session_id}")
                return jsonify(response)
            except Exception as e:
                logger.error(f"SessionManager处理消息时出错: {e}")
                return jsonify({"error": f"处理消息时出错: {str(e)}"})
        else:
            logger.info("接收到GET请求")
            return 'defeat'
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"})

if __name__ == '__main__':
    # 运行Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000)
    # 以下是测试代码，如需测试可取消注释
    # vectorstore = get_chroma_vectorstore()
    # search_results = search_relevant_info_in_chroma("精排模型的转化率提升了多少？", vectorstore, return_scores=True)
    # print(search_results)

# 邮箱导出接口
# @app.route('/api/export_session', methods=['POST'])
# def export_session():

#     """
#     导出会话历史到邮箱
#     请求体: {"session_id": "xxx", "email": "user@example.com"}
#     """
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"status": "fail", "message": "请求数据为空"})
        
#         session_id = data.get('session_id')
#         email = data.get('email')
        
#         if not session_id or not email:
#             return jsonify({"status": "fail", "message": "缺少session_id或email参数"})
        
#         # 验证邮箱格式
#         if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
#             return jsonify({"status": "fail", "message": "邮箱地址格式无效"})
        
#         # 调用SessionManager的导出功能
#         result = session_manager.export_session_to_email(session_id, email)
        
#         return jsonify(result)
        
#     except Exception as e:
#         logger.error(f"导出会话到邮箱时出错: {e}")
#         return jsonify({"status": "fail", "message": f"服务器内部错误: {str(e)}"})

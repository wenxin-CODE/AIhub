from vector import get_chroma_vectorstore
from search import search_relevant_info_in_chroma,search_relevant_info
import os
import json
import logging
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from typing import Dict, List, Tuple, Optional
# import fcntl
import threading
from datetime import datetime
from flask import Request

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

# 配置日志
logger = logging.getLogger(__name__)

# 全局锁，用于文件并发控制
file_lock = threading.Lock()

def extract_client_ip(request: Request) -> str:
    """
    从请求中提取客户端真实IP，支持代理转发场景
    """
    x_forwarded_for = request.headers.get('X-Forwarded-For')
    if x_forwarded_for:
        # 取第一个IP
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.remote_addr or ''
    return ip


def query_session_id_by_ip(client_ip: str, mapping_file: str = 'ip_session_mapping.json') -> Optional[str]:
    """
    根据IP查询session_id，支持多IP映射及无映射情况
    """
    if not os.path.exists(mapping_file):
        logger.warning("IP-Session映射文件不存在: %s", mapping_file)
        return None

    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
    except (json.JSONDecodeError, PermissionError) as e:
        logger.error("读取IP-Session映射文件失败: %s", e)
        return None

    # 支持一个IP对应多个session，这里取最新一个
    sessions = mappings.get(client_ip, [])
    if isinstance(sessions, list) and sessions:
        return sessions[-1]
    elif isinstance(sessions, str):
        return sessions
    return None


# def read_chat_history(session_id: str, chat_dir: str = 'chats') -> Optional[List[Dict]]:
#     """
#     安全读取指定session_id的聊天记录文件，含异常处理
#     """
#     chat_file = os.path.join(chat_dir, f"{session_id}.json")
#     if not os.path.exists(chat_file):
#         logger.warning("聊天记录文件不存在: %s", chat_file)
#         return None

#     try:
#         # 使用文件锁防止并发读写冲突
#         with file_lock:
#             with open(chat_file, 'r', encoding='utf-8') as f:
#                 # 加共享锁（非阻塞）
#                 fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
#                 try:
#                     history = json.load(f)
#                 finally:
#                     fcntl.flock(f.fileno(), fcntl.LOCK_UN)
#     except (PermissionError, json.JSONDecodeError, OSError) as e:
#         logger.error("读取聊天记录失败: %s", e)
#         return None

#     # 格式校验
#     if not isinstance(history, list):
#         logger.error("聊天记录格式错误，非列表")
#         return None
#     for msg in history:
#         if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
#             logger.error("聊天记录消息格式错误: %s", msg)
#             return None
#     return history


def send_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    timeout: int = 10
) -> Tuple[bool, str]:
    """
    发送邮件，含超时与异常处理
    """
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = Header(subject, 'utf-8').encode()
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        return True, "发送成功"
    except (smtplib.SMTPException, OSError) as e:
        logger.error("邮件发送失败: %s", e)
        return False, str(e)


def export_chat_history_via_email(
    request: Request,
    to_email: str,
    smtp_config: Dict[str, str],
    ip_mapping_file: str = 'ip_session_mapping.json',
    chat_dir: str = 'chats'
) -> Dict[str, str]:
    """
    主函数：导出聊天记录并通过邮件发送
    返回格式: {"status": "success|fail", "message": "详细信息"}
    """
    start_time = time.time()
    client_ip = extract_client_ip(request)
    logger.info("开始处理聊天记录导出请求, client_ip=%s, target_email=%s", client_ip, to_email)

    # 1. 查询session_id
    session_id = query_session_id_by_ip(client_ip, mapping_file=ip_mapping_file)
    if not session_id:
        msg = "未找到该IP对应的会话"
        logger.warning(msg)
        return {"status": "fail", "message": msg}

    # 2. 读取聊天记录
    history = read_chat_history(session_id, chat_dir=chat_dir)
    if history is None:
        msg = "读取聊天记录失败或记录为空"
        logger.error(msg)
        return {"status": "fail", "message": msg}

    # 3. 构造邮件内容（脱敏处理）
    body_lines = []
    for msg_item in history:
        role = msg_item.get('role', 'unknown')
        content = msg_item.get('content', '')
        # 简单脱敏：截断过长内容
        if len(content) > 500:
            content = content[:500] + "..."
        body_lines.append(f"{role}: {content}")
    body = "\n".join(body_lines)

    subject = f"Chat History Export - {session_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # 4. 发送邮件
    ok, send_msg = send_email(
        to_email=to_email,
        subject=subject,
        body=body,
        smtp_host=smtp_config['host'],
        smtp_port=int(smtp_config['port']),
        smtp_user=smtp_config['user'],
        smtp_password=smtp_config['password'],
        timeout=10
    )

    if ok:
        logger.info("聊天记录邮件发送成功, session_id=%s, elapsed=%.2fs", session_id, time.time() - start_time)
        return {"status": "success", "message": "聊天记录已成功发送到指定邮箱"}
    else:
        logger.error("聊天记录邮件发送失败, session_id=%s, reason=%s", session_id, send_msg)
        return {"status": "fail", "message": f"邮件发送失败: {send_msg}"}

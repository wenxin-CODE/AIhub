from flask import Flask, request, jsonify
from flask_cors import CORS
from create import create_chat,create_prompt,create_search_prompt
from search import search_relevant_info,search_relevant_info_in_chroma
from vector import get_chroma_vectorstore
from tools import getInfo

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
    
    pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    # vectorstore = get_chroma_vectorstore()
    # search_results = search_relevant_info_in_chroma("精排模型的转化率提升了多少？", vectorstore, return_scores=True)
    # print(search_results)
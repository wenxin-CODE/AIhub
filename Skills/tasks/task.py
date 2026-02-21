"""
美股七大科技巨头每日投资分析任务
使用 Gemini 模型执行实时数据检索和投资报告生成
"""

import os
import sys
import glob
import dotenv
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

# 设置控制台编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 加载环境变量
# load_dotenv()

# 配置API密钥
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# 初始化 Gemini 模型
dotenv.load_dotenv()

os.environ['ZHIPUAI_API_KEY'] = os.getenv('GML_API_KEY')
os.environ['ZHIPUAI_BASE_URL'] = os.getenv('GML_BASE_URL')

llm = ChatZhipuAI(
    model="GLM-4-Flash",
)

def prompt_generator(name):
    # 读取技能文件内容
    with open(name, "r", encoding="utf-8") as f:
        skill_content = f.read()

    # 定义角色
    role = llm.invoke(f"请根据用户问题:{skill_content}，推断用户可能需要什么职业的人士来进行解答。回答结果只包含职业即可").content
    print(role)
    # 创建系统提示词
    system_prompt = f"""你是一名{role}，专注于美股七大科技巨头的投资分析。

    你的任务是按照以下技能要求执行分析工作：

    {skill_content}

    请严格按照技能文件中的操作流程执行：
    1. 使用搜索工具针对每家公司执行实时数据检索
    2. 提取核心维度信息
    3. 按照报告模板生成专业的投资分析报告
    """

    return system_prompt

# 创建 ReAct Agent
# os.environ["TAVILY_API_KEY"] = "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"

def run_analysis(name):
    # 获取tavily实例（TavilySearch 继承 BaseTool，可直接作为工具使用，无需 StructuredTool 包装）
    search = TavilySearch(max_results=3) #内置工具名称、描述等信息

    # 获取提示词
    system_prompt = prompt_generator(name)

    # 获取agent Executor实例，执行invoke
    agent = create_agent(model=llm, tools=[search], system_prompt=system_prompt, debug=True)

    """执行美股分析任务"""
    print("=" * 80)
    print("开始执行任务")
    print("=" * 80)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # 执行 Agent
        result = agent.invoke({
            "messages": [("human", f"请开始执行分析任务。当前日期：{current_date}")]
        })
        
        print("\n" + "=" * 80)
        print("分析报告生成完成")
        print("=" * 80)
        
        # 提取最后一条消息作为输出
        output = result["messages"][-1].content
        
        # 如果输出是列表，提取文本内容
        if isinstance(output, list):
            output_text = ""
            for item in output:
                if isinstance(item, dict) and 'text' in item:
                    output_text += item['text']
                elif isinstance(item, str):
                    output_text += item
            output = output_text
        
        print(output)
        
        # 保存报告到文件
        result = Path(name).name

        print(result)  # 输出: politics.md
        report_filename = f"investment_report_{current_date}_{result}"
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(output)
        
        print(f"\n报告已保存至: {report_filename}")
        
        return output
        
    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # 获取当前文件夹下所有md文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_files = glob.glob(os.path.join(current_dir, "*.md"))
    
    if not md_files:
        print("未找到任何md文件")
        sys.exit(0)
    
    print(f"找到 {len(md_files)} 个md文件:")
    for i, md_file in enumerate(md_files, 1):
        print(f"  {i}. {os.path.basename(md_file)}")
    
    print("\n" + "="*80)
    
    # 依次处理每个md文件
    for md_file in md_files:
        print(f"\n开始处理: {os.path.basename(md_file)}")
        print("="*80)
        try:
            run_analysis(md_file)
            print(f"\n✓ 完成: {os.path.basename(md_file)}")
        except Exception as e:
            print(f"\n✗ 失败: {os.path.basename(md_file)}")
            print(f"错误: {str(e)}")
        print("="*80 + "\n")

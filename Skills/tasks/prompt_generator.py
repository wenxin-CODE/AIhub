"""
智能系统提示词生成器
根据技能文件内容自动判断需要的专业人士角色，并生成对应的system_prompt
"""

import os
import re
import sys
from typing import Dict, List, Tuple

# 设置控制台编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class SystemPromptGenerator:
    """系统提示词生成器类"""
    
    # 专业角色关键词映射
    ROLE_KEYWORDS = {
        "美股分析师": {
            "keywords": ["美股", "股票", "投资", "财报", "证券", "纳斯达克", "标普", "道琼斯", "市值", "股价"],
            "description": "资深美股分析师，专注于美股市场的投资分析与研究"
        },
        "AI工程师": {
            "keywords": ["人工智能", "AI", "机器学习", "深度学习", "神经网络", "模型训练", "算法"],
            "description": "资深AI工程师，专注于人工智能算法研发与应用"
        },
        "数据分析师": {
            "keywords": ["数据分析", "数据挖掘", "统计分析", "数据可视化", "报表", "指标"],
            "description": "资深数据分析师，擅长数据挖掘与商业智能分析"
        },
        "产品经理": {
            "keywords": ["产品", "用户需求", "功能设计", "产品规划", "用户体验", "迭代"],
            "description": "资深产品经理，专注于产品规划与用户体验优化"
        },
        "技术架构师": {
            "keywords": ["架构", "系统设计", "技术选型", "微服务", "分布式", "高可用"],
            "description": "资深技术架构师，专注于系统架构设计与技术决策"
        },
        "金融分析师": {
            "keywords": ["金融", "理财", "基金", "债券", "期货", "外汇", "风险管理"],
            "description": "资深金融分析师，专注于金融市场分析与风险管理"
        },
        "法律顾问": {
            "keywords": ["法律", "合规", "法规", "合同", "知识产权", "诉讼"],
            "description": "资深法律顾问，专注于法律合规与风险防控"
        },
        "市场营销专家": {
            "keywords": ["营销", "推广", "品牌", "广告", "市场", "获客", "转化"],
            "description": "资深市场营销专家，专注于品牌建设与市场推广"
        },
        "内容创作者": {
            "keywords": ["内容", "写作", "文案", "编辑", "创作", "文章", "博客"],
            "description": "资深内容创作者，擅长高质量内容创作与编辑"
        },
        "项目经理": {
            "keywords": ["项目", "进度", "资源", "协调", "交付", "里程碑", "风险"],
            "description": "资深项目经理，专注于项目规划与执行管理"
        }
    }
    
    # 任务类型关键词映射
    TASK_KEYWORDS = {
        "分析报告": ["分析", "报告", "研究", "评估", "洞察"],
        "数据检索": ["搜索", "检索", "查询", "获取", "搜集"],
        "内容生成": ["生成", "创建", "编写", "撰写", "输出"],
        "决策建议": ["建议", "决策", "推荐", "策略", "方案"],
        "监控预警": ["监控", "预警", "跟踪", "追踪", "检测"]
    }
    
    def __init__(self):
        """初始化生成器"""
        pass
    
    def read_skill_file(self, file_path: str) -> str:
        """
        读取技能文件内容
        
        Args:
            file_path: 技能文件路径
            
        Returns:
            str: 文件内容
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def extract_metadata(self, content: str) -> Dict[str, str]:
        """
        从技能文件中提取元数据
        
        Args:
            content: 文件内容
            
        Returns:
            Dict: 元数据字典
        """
        metadata = {}
        
        # 提取YAML前置元数据
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_content = parts[1]
                # 提取name
                name_match = re.search(r"name:\s*(.+)", yaml_content)
                if name_match:
                    metadata["name"] = name_match.group(1).strip()
                
                # 提取description
                desc_match = re.search(r"description:\s*(.+)", yaml_content)
                if desc_match:
                    metadata["description"] = desc_match.group(1).strip()
                
                # 提取tags
                tags_match = re.search(r"tags:\s*\[(.+)\]", yaml_content)
                if tags_match:
                    metadata["tags"] = [tag.strip() for tag in tags_match.group(1).split(",")]
        
        return metadata
    
    def identify_role(self, content: str, metadata: Dict) -> Tuple[str, str]:
        """
        识别需要的专业角色
        
        Args:
            content: 文件内容
            metadata: 元数据
            
        Returns:
            Tuple[str, str]: (角色名称, 角色描述)
        """
        # 合并内容和元数据进行关键词匹配
        search_text = content.lower()
        if "tags" in metadata:
            search_text += " " + " ".join(metadata["tags"]).lower()
        if "description" in metadata:
            search_text += " " + metadata["description"].lower()
        
        # 统计每个角色的关键词匹配次数
        role_scores = {}
        for role, role_info in self.ROLE_KEYWORDS.items():
            score = 0
            for keyword in role_info["keywords"]:
                # 计算关键词出现次数
                count = len(re.findall(keyword, search_text))
                score += count
            role_scores[role] = score
        
        # 选择得分最高的角色
        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            if role_scores[best_role] > 0:
                return best_role, self.ROLE_KEYWORDS[best_role]["description"]
        
        # 默认角色
        return "专业分析师", "资深专业分析师，专注于提供高质量的分析服务"
    
    def identify_task_type(self, content: str) -> str:
        """
        识别任务类型
        
        Args:
            content: 文件内容
            
        Returns:
            str: 任务类型
        """
        content_lower = content.lower()
        task_scores = {}
        
        for task_type, keywords in self.TASK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            task_scores[task_type] = score
        
        if task_scores:
            best_task = max(task_scores, key=task_scores.get)
            if task_scores[best_task] > 0:
                return best_task
        
        return "任务执行"
    
    def extract_workflow_steps(self, content: str) -> List[str]:
        """
        从技能文件中提取工作流程步骤
        
        Args:
            content: 文件内容
            
        Returns:
            List[str]: 步骤列表
        """
        steps = []
        
        # 查找操作流程部分
        workflow_patterns = [
            r"操作流程.*?\n(.*?)(?=\n##|\n---|\Z)",
            r"执行步骤.*?\n(.*?)(?=\n##|\n---|\Z)",
            r"工作流程.*?\n(.*?)(?=\n##|\n---|\Z)"
        ]
        
        for pattern in workflow_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                workflow_content = match.group(1)
                # 提取步骤
                step_matches = re.findall(r"(?:第[一二三四五六七八九十]+步|Step\s*\d+|步骤\d*|^\d+\.)\s*[:：]?\s*(.+)", 
                                         workflow_content, re.MULTILINE)
                if step_matches:
                    steps = [s.strip() for s in step_matches]
                    break
        
        return steps
    
    def generate_system_prompt(self, skill_file_path: str) -> str:
        """
        根据技能文件生成系统提示词
        
        Args:
            skill_file_path: 技能文件路径
            
        Returns:
            str: 生成的系统提示词
        """
        # 读取文件内容
        content = self.read_skill_file(skill_file_path)
        
        # 提取元数据
        metadata = self.extract_metadata(content)
        
        # 识别角色
        role_name, role_description = self.identify_role(content, metadata)
        
        # 识别任务类型
        task_type = self.identify_task_type(content)
        
        # 提取工作流程步骤
        workflow_steps = self.extract_workflow_steps(content)
        
        # 生成系统提示词
        system_prompt = f"你是一名{role_description}。\n\n"
        
        # 添加任务描述
        if "description" in metadata:
            system_prompt += f"你的任务是{metadata['description']}\n\n"
        else:
            system_prompt += f"你的任务是执行{task_type}工作。\n\n"
        
        # 添加技能要求
        system_prompt += "你需要按照以下技能要求执行工作：\n\n"
        system_prompt += f"{content}\n\n"
        
        # 添加执行步骤
        if workflow_steps:
            system_prompt += "请严格按照以下步骤执行：\n"
            for i, step in enumerate(workflow_steps, 1):
                system_prompt += f"{i}. {step}\n"
        else:
            system_prompt += "请严格按照技能文件中的操作流程执行任务。\n"
        
        return system_prompt


def generate_system_prompt_from_skill(skill_file_path: str) -> str:
    """
    便捷函数：根据技能文件生成系统提示词
    
    Args:
        skill_file_path: 技能文件路径
        
    Returns:
        str: 生成的系统提示词
    """
    generator = SystemPromptGenerator()
    return generator.generate_system_prompt(skill_file_path)


# 示例用法
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        skill_file = sys.argv[1]
    else:
        skill_file = "skill.md"
    
    if os.path.exists(skill_file):
        prompt = generate_system_prompt_from_skill(skill_file)
        print("="*80)
        print("生成的系统提示词：")
        print("="*80)
        print(prompt)
    else:
        print(f"错误：文件不存在 {skill_file}")

"""
任务调度器
读取tasks文件夹中的py文件并依次执行，将生成的md文件保存到result文件夹下的日期文件夹中
"""

import os
import sys
import glob
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# 设置控制台编码为UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class TaskScheduler:
    """任务调度器类"""
    
    def __init__(self, base_dir=None):
        """
        初始化任务调度器
        
        Args:
            base_dir: 基础目录，默认为当前文件所在目录
        """
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.base_dir = Path(base_dir)
        self.tasks_dir = self.base_dir / "tasks"
        self.result_dir = self.base_dir / "result"
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.original_md_files = set()  # 记录原有的md文件
        
    def get_task_files(self):
        """
        获取tasks文件夹中的所有py文件
        
        Returns:
            list: py文件路径列表
        """
        if not self.tasks_dir.exists():
            print(f"错误: tasks文件夹不存在: {self.tasks_dir}")
            return []
        
        # 记录原有的md文件
        self.original_md_files = set(self.tasks_dir.glob("*.md"))
        if self.original_md_files:
            print(f"原有md文件: {len(self.original_md_files)} 个")
            for md_file in self.original_md_files:
                print(f"  - {md_file.name}")
        
        # 获取所有py文件（排除__init__.py）
        task_files = list(self.tasks_dir.glob("*.py"))
        task_files = [f for f in task_files if f.name != "__init__.py"]
        
        print(f"找到 {len(task_files)} 个任务文件:")
        for i, task_file in enumerate(task_files, 1):
            print(f"  {i}. {task_file.name}")
        
        return task_files
    
    def create_result_folder(self):
        """
        在result文件夹下创建以当天日期命名的文件夹
        
        Returns:
            Path: 新创建的文件夹路径
        """
        # 确保result文件夹存在
        self.result_dir.mkdir(exist_ok=True)
        
        # 创建日期文件夹
        date_folder = self.result_dir / self.current_date
        date_folder.mkdir(exist_ok=True)
        
        print(f"结果文件夹: {date_folder}")
        return date_folder
    
    def execute_task(self, task_file):
        """
        执行单个任务文件
        
        Args:
            task_file: 任务文件路径
            
        Returns:
            bool: 执行是否成功
        """
        print(f"\n{'='*80}")
        print(f"开始执行任务: {task_file.name}")
        print(f"{'='*80}")
        
        try:
            # 切换到tasks目录执行
            original_dir = os.getcwd()
            os.chdir(self.tasks_dir)
            
            # 执行py文件
            result = subprocess.run(
                [sys.executable, task_file.name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=300  # 5分钟超时
            )
            
            # 恢复原目录
            os.chdir(original_dir)
            
            if result.returncode == 0:
                print(f"✓ 任务执行成功: {task_file.name}")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"✗ 任务执行失败: {task_file.name}")
                if result.stderr:
                    print(f"错误信息: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ 任务超时: {task_file.name}")
            os.chdir(original_dir)
            return False
        except Exception as e:
            print(f"✗ 任务执行异常: {task_file.name}")
            print(f"异常信息: {str(e)}")
            os.chdir(original_dir)
            return False
    
    def find_generated_md_files(self):
        """
        查找tasks文件夹中新生成的md文件（排除原有文件）
        
        Returns:
            list: 新生成的md文件路径列表
        """
        # 获取当前所有md文件
        current_md_files = set(self.tasks_dir.glob("*.md"))
        
        # 找出新生成的md文件（当前文件 - 原有文件）
        new_md_files = current_md_files - self.original_md_files
        
        return list(new_md_files)
    
    def move_md_files(self, target_folder):
        """
        将新生成的md文件移动到目标文件夹（不移动原有md文件）
        
        Args:
            target_folder: 目标文件夹路径
        """
        md_files = self.find_generated_md_files()
        
        if not md_files:
            print("未找到新生成的md文件")
            return
        
        print(f"\n找到 {len(md_files)} 个新生成的md文件，正在移动...")
        
        for md_file in md_files:
            target_path = target_folder / md_file.name
            
            # 如果目标文件已存在，添加时间戳
            if target_path.exists():
                timestamp = datetime.now().strftime("%H%M%S")
                stem = target_path.stem
                new_name = f"{stem}_{timestamp}.md"
                target_path = target_folder / new_name
            
            shutil.move(str(md_file), str(target_path))
            print(f"  ✓ {md_file.name} -> {target_path.relative_to(self.base_dir)}")
    
    def run(self):
        """
        运行任务调度器
        """
        print("\n" + "="*80)
        print("任务调度器启动")
        print("="*80)
        print(f"当前日期: {self.current_date}")
        print(f"基础目录: {self.base_dir}")
        print(f"任务目录: {self.tasks_dir}")
        print(f"结果目录: {self.result_dir}")
        
        # 1. 获取任务文件
        task_files = self.get_task_files()
        if not task_files:
            print("\n没有找到任务文件，退出")
            return
        
        # 2. 创建结果文件夹
        result_folder = self.create_result_folder()
        
        # 3. 依次执行任务
        success_count = 0
        fail_count = 0
        
        for task_file in task_files:
            if self.execute_task(task_file):
                success_count += 1
            else:
                fail_count += 1
        
        # 4. 移动生成的md文件
        self.move_md_files(result_folder)
        
        # 5. 输出统计信息
        print("\n" + "="*80)
        print("任务调度完成")
        print("="*80)
        print(f"总任务数: {len(task_files)}")
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        print(f"结果保存在: {result_folder.relative_to(self.base_dir)}")
        print("="*80 + "\n")


def main():
    """主函数"""
    scheduler = TaskScheduler()
    scheduler.run()


if __name__ == "__main__":
    main()

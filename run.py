# run_all.py

import os
import sys
import importlib.util

# 确保根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

def run_script(script_path):
    """动态导入并运行一个 Python 脚本"""
    if not os.path.exists(script_path):
        print(f"[警告] 文件不存在: {script_path}")
        return

    # 生成模块名
    module_name = os.path.splitext(os.path.basename(script_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)

    # 插入到 sys.modules 避免重复导入
    sys.modules[module_name] = module

    # 执行脚本（相当于 __name__ == "__main__"）
    print(f"\n{'='*50}")
    print(f"运行: {script_path}")
    print(f"{'='*50}")
    spec.loader.exec_module(module)

def main():
    # 运行 pre_train 下的所有脚本
    pre_train_dir = os.path.join(PROJECT_ROOT, 'pre_train')
    for file in os.listdir(pre_train_dir):
        if file.endswith('.py') and not file.startswith('__'):
            script_path = os.path.join(pre_train_dir, file)
            run_script(script_path)

    # 运行 supervised_finetuning 下的所有脚本
    sft_dir = os.path.join(PROJECT_ROOT, 'supervised_finetuning')
    for file in os.listdir(sft_dir):
        if file.endswith('.py') and not file.startswith('__'):
            script_path = os.path.join(sft_dir, file)
            run_script(script_path)

if __name__ == "__main__":
    main()
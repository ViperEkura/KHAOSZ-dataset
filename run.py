import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

def run_script(script_path):
    if not os.path.exists(script_path):
        print(f"[Warning] File does not exist: {script_path}")
        return

    print(f"\n{'='*50}")
    print(f"Running: {script_path}")
    print(f"{'='*50}")

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = PROJECT_ROOT
        subprocess.run(
            [sys.executable, script_path], 
            check=True, 
            cwd=PROJECT_ROOT, 
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"[Error] Script execution failed: {script_path}, Error code: {e.returncode}")
        
def run_scripts(project_root: str, directory: str):
    pre_train_dir = os.path.join(project_root, directory)
    for file in os.listdir(pre_train_dir):
        if file.endswith('.py'):
            script_path = os.path.join(pre_train_dir, file)
            run_script(script_path)

def main():
    run_scripts(PROJECT_ROOT, 'pre_train')
    run_scripts(PROJECT_ROOT, 'supervised_finetuning')

if __name__ == "__main__":
    main()
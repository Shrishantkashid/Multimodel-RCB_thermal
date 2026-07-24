import subprocess
import sys
import os

def main():
    config_path = "stage2_baseline/configs/rgb_baseline.py"
    work_dir = "stage2_baseline/work_dirs/rgb_baseline"
    
    if not os.path.exists(config_path):
        print(f"Config {config_path} not found.")
        sys.exit(1)
        
    print(f"Starting RGB Baseline Training...")
    cmd = f"mim train mmdet {config_path} --work-dir {work_dir}"
    print(f"Running command: {cmd}")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("Training failed or was interrupted.")
        sys.exit(result.returncode)
        
    print("Training finished!")

if __name__ == "__main__":
    main()

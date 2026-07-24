import subprocess
import sys
import os

def run_evaluation(modality, config_path, checkpoint, out_json):
    print(f"\nEvaluating {modality.upper()} Baseline...")
    
    if not os.path.exists(config_path):
        print(f"Config {config_path} not found.")
        return False
        
    if not os.path.exists(checkpoint):
        print(f"Checkpoint {checkpoint} not found. Did you finish training?")
        return False
        
    cmd = f"mim test mmdet {config_path} --checkpoint {checkpoint} --out {out_json} --eval bbox"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    os.makedirs("stage2_baseline/outputs/metrics", exist_ok=True)
    
    rgb_cfg = "stage2_baseline/configs/rgb_baseline.py"
    rgb_ckpt = "stage2_baseline/work_dirs/rgb_baseline/epoch_12.pth"
    rgb_out = "stage2_baseline/outputs/metrics/rgb_metrics.json"
    
    thermal_cfg = "stage2_baseline/configs/thermal_baseline.py"
    thermal_ckpt = "stage2_baseline/work_dirs/thermal_baseline/epoch_12.pth"
    thermal_out = "stage2_baseline/outputs/metrics/thermal_metrics.json"
    
    run_evaluation("rgb", rgb_cfg, rgb_ckpt, rgb_out)
    run_evaluation("thermal", thermal_cfg, thermal_ckpt, thermal_out)

if __name__ == "__main__":
    main()

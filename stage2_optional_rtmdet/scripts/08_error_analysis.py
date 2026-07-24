import subprocess
import os

def analyze_errors(modality, config_path, prediction_pkl, out_dir):
    print(f"Running Error Analysis for {modality}...")
    os.makedirs(out_dir, exist_ok=True)
    
    # In MMDetection, you typically run:
    # python tools/analysis_tools/analyze_results.py config prediction.pkl out_dir --show-score-thr 0.3
    # Since we are using mim, we can use mim run mmdet analyze_results
    
    cmd = f"mim run mmdet analyze_results {config_path} {prediction_pkl} {out_dir} --show-score-thr 0.3"
    print(f"Command: {cmd}")
    print("Ensure you have generated the prediction.pkl using mim test --out prediction.pkl first!")

def main():
    rgb_cfg = "stage2_baseline/configs/rgb_baseline.py"
    rgb_pkl = "stage2_baseline/outputs/metrics/rgb_predictions.pkl"
    rgb_out = "stage2_baseline/outputs/visualizations/error_analysis/rgb"
    
    thermal_cfg = "stage2_baseline/configs/thermal_baseline.py"
    thermal_pkl = "stage2_baseline/outputs/metrics/thermal_predictions.pkl"
    thermal_out = "stage2_baseline/outputs/visualizations/error_analysis/thermal"
    
    analyze_errors("rgb", rgb_cfg, rgb_pkl, rgb_out)
    analyze_errors("thermal", thermal_cfg, thermal_pkl, thermal_out)

if __name__ == "__main__":
    main()

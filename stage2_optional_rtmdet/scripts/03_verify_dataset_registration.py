import os
from mmengine.config import Config
from mmdet.registry import DATASETS

def verify_dataset(config_path):
    print(f"\nVerifying dataset registration for {config_path}...")
    try:
        cfg = Config.fromfile(config_path)
        
        # Build the training dataset
        dataset_cfg = cfg.train_dataloader.dataset
        dataset = DATASETS.build(dataset_cfg)
        
        print(f"Success! Dataset registered.")
        print(f"Number of instances in dataset: {len(dataset)}")
        print(f"Sample data info: {dataset[0]}")
    except Exception as e:
        print(f"Error verifying dataset: {e}")

def main():
    rgb_cfg = 'stage2_baseline/configs/rgb_baseline.py'
    thermal_cfg = 'stage2_baseline/configs/thermal_baseline.py'
    
    if not os.path.exists(rgb_cfg):
        print("Configs not found. Run 02_prepare_configs.py first.")
        return
        
    verify_dataset(rgb_cfg)
    verify_dataset(thermal_cfg)

if __name__ == "__main__":
    main()

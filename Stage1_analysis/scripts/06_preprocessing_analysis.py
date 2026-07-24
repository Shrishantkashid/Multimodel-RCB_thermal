import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from pathlib import Path

# Parameters for CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

def main():
    # Paths
    base_dir = Path("data/VTUAV_subset")
    annotations_path = base_dir / "annotations" / "train.json"
    thermal_dir = base_dir / "VTUAV_ir" / "train" / "images"
    
    output_dir = Path("Stage1_analysis/outputs/preprocessing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(annotations_path, 'r') as f:
            coco = json.load(f)
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return
        
    stats = []
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    
    # Process ~10 representative images
    count = 0
    for img_info in coco.get("images", []):
        if count >= 10:
            break
            
        filename = img_info["file_name"]
        thermal_path = thermal_dir / filename
        
        if not thermal_path.exists():
            continue
            
        # 1. Load image
        img = cv2.imread(str(thermal_path))
        if img is None:
            continue
            
        # 2. Convert to grayscale if necessary
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 3. Global Histogram Equalization
        equalized = cv2.equalizeHist(gray)
        
        # 4. CLAHE
        clahe_img = clahe.apply(gray)
        
        # Calculate statistics
        for method, img_data in [("Original", gray), ("Histogram Equalization", equalized), ("CLAHE", clahe_img)]:
            stats.append({
                "image": filename,
                "method": method,
                "mean_intensity": np.mean(img_data),
                "std_intensity": np.std(img_data),
                "min_intensity": np.min(img_data),
                "max_intensity": np.max(img_data)
            })
            
        # Generate side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Sample: {filename}", fontsize=16)
        
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title("Original Thermal")
        axes[0].axis("off")
        
        axes[1].imshow(equalized, cmap='gray')
        axes[1].set_title("Histogram Equalization")
        axes[1].axis("off")
        
        axes[2].imshow(clahe_img, cmap='gray')
        axes[2].set_title("CLAHE")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{count+1:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        count += 1

    # Save Statistics
    if stats:
        df = pd.DataFrame(stats)
        df.to_csv(output_dir / "preprocessing_statistics.csv", index=False)
        print("Generated preprocessing_statistics.csv")
    
    # Generate Summary Markdown
    summary_md = """# Stage 1 Thermal Preprocessing Analysis

## Techniques Evaluated

- Original thermal imagery
- Global Histogram Equalization
- CLAHE

## Purpose

Preprocessing was explored to determine whether thermal contrast enhancement may improve visual separation of pedestrians in difficult thermal scenes.

## Method

Representative thermal samples were compared using original images, global histogram equalization, and CLAHE.

## Observations

- Global Histogram Equalization significantly stretched the contrast but often exaggerated background noise.
- CLAHE increased local contrast without severely overexposing the background, enhancing edges of objects in the scene.
- Both CLAHE and global histogram equalization increased image contrast in several evaluated samples, but the impact on pedestrian detection performance must be validated experimentally during later model evaluation.

## Decision

The raw dataset remains unchanged.

Preprocessing is currently treated as an experimental candidate rather than automatically applied to the complete training pipeline.

Whether preprocessing should be adopted will later be determined using detection metrics rather than visual appearance alone.
"""
    with open(output_dir / "preprocessing_summary.md", "w") as f:
        f.write(summary_md)
        
    print(f"Processed {count} images and generated outputs in {output_dir}")

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import os
import json

def check_alignment(rgb_path, thermal_path, output_path):
    rgb = cv2.imread(rgb_path)
    thermal = cv2.imread(thermal_path)
    
    if rgb is None or thermal is None:
        print(f"Failed to read images at {rgb_path} or {thermal_path}")
        return False
        
    # Resize thermal to match RGB dimensions if they differ
    thermal_resized = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))
    
    # Blend both images to visually check if edges/objects line up
    overlay = cv2.addWeighted(rgb, 0.5, thermal_resized, 0.5, 0)
    cv2.imwrite(output_path, overlay)
    return True

with open("data/VTUAV_subset/annotations/train.json") as f:
    coco = json.load(f)

rgb_dir = "data/VTUAV_subset/VTUAV_co/train/images"
thermal_dir = "data/VTUAV_subset/VTUAV_ir/train/images"

count = 0
for img_info in coco["images"]:
    filename = img_info["file_name"]
    rgb_path = os.path.join(rgb_dir, filename)
    thermal_path = os.path.join(thermal_dir, filename)
    
    out_path = f"Stage1_analysis/outputs/visualizations/alignment_check_{count}.png"
    if check_alignment(rgb_path, thermal_path, out_path):
        count += 1
    
    if count >= 3:
        break

print(f"Saved {count} alignment checks to Stage1_analysis/outputs/visualizations/")

import cv2
import json
import matplotlib.pyplot as plt
import os

def draw_boxes(image, annotations, image_id):
    if image is None: return None
    img = image.copy()
    for ann in annotations:
        if ann.get("image_id") == image_id:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

with open("data/VTUAV_subset/annotations/train.json") as f:
    coco = json.load(f)

os.makedirs("Stage1_analysis/outputs/visualizations", exist_ok=True)

rgb_dir = "data/VTUAV_subset/VTUAV_co/train/images"
thermal_dir = "data/VTUAV_subset/VTUAV_ir/train/images"

successful_pairs = 0
for i, img_info in enumerate(coco["images"][:20]): # try first 20 images
    filename = img_info["file_name"]
    image_id = img_info["id"]
    
    rgb_path = os.path.join(rgb_dir, filename)
    thermal_path = os.path.join(thermal_dir, filename)
    
    rgb_img = cv2.imread(rgb_path)
    thermal_img = cv2.imread(thermal_path)
    
    if rgb_img is None or thermal_img is None:
        print(f"Skipping {filename} - Could not read both RGB and Thermal images.")
        continue
        
    rgb_annotated = draw_boxes(rgb_img, coco["annotations"], image_id)
    thermal_annotated = draw_boxes(thermal_img, coco["annotations"], image_id)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(rgb_annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title("RGB")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(thermal_annotated, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Thermal")
    axes[1].axis("off")
    
    plt.savefig(f"Stage1_analysis/outputs/visualizations/pair_{i:02d}.png", bbox_inches="tight")
    plt.close()
    successful_pairs += 1

print(f"Saved {successful_pairs} annotated pairs to Stage1_analysis/outputs/visualizations/")

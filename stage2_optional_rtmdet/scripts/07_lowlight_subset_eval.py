import subprocess
import sys
import os
import json

def generate_lowlight_subset(full_ann_path, out_ann_path):
    # This is a conceptual implementation. 
    # In a real scenario, you would filter images based on average RGB brightness or an explicit metadata tag.
    print(f"Generating low-light subset from {full_ann_path}...")
    try:
        with open(full_ann_path, 'r') as f:
            coco = json.load(f)
            
        # Example filter: In the hackathon, you might hardcode the image IDs of known nighttime scenes 
        # or compute the mean intensity of the RGB images and select those < threshold.
        # For this skeleton, we'll assume we take a random 20% for demonstration if no metadata exists.
        subset_images = coco['images'][:len(coco['images'])//5]
        subset_img_ids = set([img['id'] for img in subset_images])
        subset_anns = [ann for ann in coco['annotations'] if ann['image_id'] in subset_img_ids]
        
        subset_coco = {
            'images': subset_images,
            'annotations': subset_anns,
            'categories': coco['categories']
        }
        
        with open(out_ann_path, 'w') as f:
            json.dump(subset_coco, f)
            
        print(f"Saved {len(subset_images)} low-light images to {out_ann_path}")
        return True
    except Exception as e:
        print(f"Error generating subset: {e}")
        return False

def main():
    os.makedirs("stage2_baseline/outputs/metrics", exist_ok=True)
    
    test_ann = "data/VTUAV_subset/annotations/test.json"
    lowlight_ann = "data/VTUAV_subset/annotations/test_lowlight.json"
    
    if not generate_lowlight_subset(test_ann, lowlight_ann):
        return
        
    print("Evaluate RGB and Thermal on the Low-Light Subset using mim test (similar to 06_evaluate.py)...")
    # You would dynamically create a config here pointing to test_lowlight.json and evaluate

if __name__ == "__main__":
    main()

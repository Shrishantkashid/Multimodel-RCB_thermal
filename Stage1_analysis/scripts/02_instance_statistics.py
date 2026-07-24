import json
import pandas as pd
import os

def analyze_split(json_path, split_name):
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} does not exist.")
        return None

    with open(json_path) as f:
        data = json.load(f)
    
    num_images = len(data["images"])
    num_instances = len(data["annotations"])
    avg_per_image = round(num_instances / num_images, 2) if num_images else 0
    
    print(f"\n=== {split_name} ===")
    print(f"Images: {num_images}")
    print(f"Pedestrian instances: {num_instances}")
    print(f"Avg pedestrians per image: {avg_per_image}")
    
    return {
        "split": split_name,
        "images": num_images,
        "instances": num_instances,
        "avg_per_image": avg_per_image
    }

results = []
results.append(analyze_split("data/VTUAV_subset/annotations/train.json", "train"))
results.append(analyze_split("data/VTUAV_subset/annotations/val.json", "val"))
results.append(analyze_split("data/VTUAV_subset/annotations/test.json", "test"))

results = [r for r in results if r is not None]

if results:
    df = pd.DataFrame(results)
    os.makedirs("Stage1_analysis/outputs", exist_ok=True)
    df.to_csv("Stage1_analysis/outputs/instance_statistics.csv", index=False)
    print("\nSaved instance_statistics.csv:")
    print(df)

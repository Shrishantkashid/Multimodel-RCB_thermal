import os
import json

data_root = "data/VTUAV_subset"

print("--- Inspecting Folder Structure ---")
if os.path.exists(data_root):
    print(f"\n=== {data_root} ===")
    for item in os.listdir(data_root):
        print("  ", item)
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            for f in files[:3]:
                print("      ", f)
            if len(files) > 3:
                print(f"       ... and {len(files) - 3} more")

print("\n--- Inspecting Annotations (COCO format) ---")
try:
    with open(os.path.join(data_root, "annotations", "train.json")) as f:
        coco_data = json.load(f)
        print("Keys:", coco_data.keys())
        if "categories" in coco_data:
            print("Categories:", coco_data["categories"])
        if "annotations" in coco_data and len(coco_data["annotations"]) > 0:
            print("First annotation example:", coco_data["annotations"][0])
except FileNotFoundError as e:
    print(f"Error: {e}")

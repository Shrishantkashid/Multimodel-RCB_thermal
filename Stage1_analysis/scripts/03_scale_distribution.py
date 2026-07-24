import json

def scale_distribution(json_path, split_name):
    with open(json_path) as f:
        data = json.load(f)
    
    small, medium, large = 0, 0, 0
    for ann in data["annotations"]:
        w, h = ann["bbox"][2], ann["bbox"][3] # COCO bbox format: [x, y, width, height]
        area = w * h
        if area < 32**2:
            small += 1
        elif area < 96**2:
            medium += 1
        else:
            large += 1
            
    total = small + medium + large
    if total == 0:
        total = 1 # avoid division by zero if empty
        
    print(f"\n=== {split_name} scale distribution ===")
    print(f"Small: {small} ({round(100*small/total,1)}%)")
    print(f"Medium: {medium} ({round(100*medium/total,1)}%)")
    print(f"Large: {large} ({round(100*large/total,1)}%)")
    
    return {"split": split_name, "small": small, "medium": medium, "large": large}

for split in ["train", "val", "test"]:
    try:
        scale_distribution(f"data/VTUAV_subset/annotations/{split}.json", split)
    except FileNotFoundError:
        print(f"Error: {split}.json not found.")

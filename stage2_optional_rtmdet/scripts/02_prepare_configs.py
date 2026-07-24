import os

def create_config(modality, config_path):
    # Set the appropriate directory based on modality
    img_dir = 'VTUAV_co' if modality == 'rgb' else 'VTUAV_ir'
    
    # We use a base Faster R-CNN config template and override the dataset paths
    config_content = f"""
# Inherit from a standard MMDetection baseline
_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# Modifying the model to output 1 class instead of 80
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)
    )
)

# Dataset settings
data_root = 'data/VTUAV_subset/'
metainfo = dict(classes=('person',), palette=[(220, 20, 60)])

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='{img_dir}/train/images/')
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='{img_dir}/val/images/')
    )
)
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='{img_dir}/test/images/')
    )
)

val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/test.json')

# Customize training schedule for fine-tuning
train_cfg = dict(max_epochs=12, val_interval=1)
optim_wrapper = dict(optimizer=dict(lr=0.01))

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"Created config for {modality.upper()} baseline at {config_path}")

def main():
    print("Preparing baseline configurations...")
    create_config('rgb', 'stage2_baseline/configs/rgb_baseline.py')
    create_config('thermal', 'stage2_baseline/configs/thermal_baseline.py')
    print("Configuration generation complete!")

if __name__ == "__main__":
    main()

# Proposed Standard Fair Baseline Recipe

本文采用标准公平 baseline，而非极限调参 baseline。所有消融模型均沿用 baseline 的相同训练配置，确保性能差异主要来自模型结构变化。

| 项目 | 配置 |
|---|---|
| model | YOLOv8m |
| data | /root/autodl-tmp/yolov8/data/dental_lesion_3cls_large.yaml |
| dataset path | /root/autodl-tmp/dental_lesion_3cls_large |
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| device | 0 |
| workers | 8 |
| seed | 42 |
| patience | 100 |
| optimizer | auto |
| cos_lr | False |
| amp | True |
| cache | False |
| close_mosaic | 10 |
| augmentation | 当前 YOLOv8 默认增强参数 |
| 选择理由 | 该配置与当前 Ultralytics YOLO 默认训练设置接近，适合 RTX 3090 24GB 和 8118/1690/1715 的 train/val/test 数据规模；它不是故意弱化 baseline，也不是针对牙病数据集的极限调参。 |

若 batch=16 发生 OOM，所有实验必须统一改为 batch=8，并从 baseline 重新开始。

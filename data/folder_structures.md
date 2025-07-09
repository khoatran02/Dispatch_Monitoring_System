Dataset/
├── Detection/  # YOLO format (bounding boxes)
│   ├── train/
│   │   ├── images/  # All dish/tray images
│   │   └── labels/  # YOLO labels (0: dish, 1: tray)
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
└── Classification/  # Separate classes
    ├── dish/
    │   ├── empty/
    │   ├── kakigori/
    │   └── not_empty/
    └── tray/
        ├── empty/
        ├── kakigori/
        └── not_empty/
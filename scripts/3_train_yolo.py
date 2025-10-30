from ultralytics import YOLO
import os

print("🚀 Training YOLOv8 model on annotated dataset...")

model = YOLO("yolov8n.pt")

model.train(
    data=r"C:\Desktop\vision_agent\data\processed\yolo_car_dataset\dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="car_detector_v2"
)

print("✅ Training complete!")

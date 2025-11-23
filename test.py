
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import ultralytics.nn.tasks
model = YOLO("/home/YOLOv8/m3fd/train4/weights/best.pt")
results = model.val(data="/home/YOLOv8/data/M3FDtest.yaml",batch=16,imgsz=640,device=0,project="/home/YOLOv8/runswuyong")
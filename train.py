#训练
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
import ultralytics.nn.tasks
model = YOLO("/home/YOLOv8/yaml/yolov8l-hd-changehead.yaml")
results = model.train(data="/home/YOLOv8/data/M3FD.yaml",batch=8,epochs=300,imgsz=640,device=1,project="/home/YOLOv8/runsyanzheng/m3fd/")

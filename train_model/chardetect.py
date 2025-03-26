from ultralytics import YOLO

import os
os.environ['WANDB_MODE'] = 'disabled'

model = YOLO("yolov11n.pt")
results = model.train(data=r"C:\Users\Downloads\final_project\dataset\LP_detection\data1.yaml", epochs=45, imgsz=640 ,device=0)
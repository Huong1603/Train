from ultralytics import YOLO
from PIL import Image
from helper import *
from number_classifier import get_number_classifier
import os
file_path = os.path.abspath(__file__)
weights_dir = os.path.join(os.path.dirname(file_path), 'weights')
print(f"Weight dir : {weights_dir}")
yolov11 = YOLO(os.path.join(weights_dir,"best_25k.pt"))
yolo_number_detection = YOLO(os.path.join(weights_dir,"best_uc3_val.pt"))
number_classifier = get_number_classifier(device = 'cpu',pretrained_path = os.path.join(weights_dir,"inception_classifier.pt") )


def get_plate(img):
	result = yolov11(img,verbose = False,conf = 0.35)[0]
	if len(result.boxes) == 0:
		return None,img
	all_plate = []
	for index in range(len(result.boxes.xyxy)):
		
		plate = result.boxes.xyxy[index].tolist()
		all_plate.append(plate)
 
	return all_plate


def get_text_from_plate(crop_img):
	
		all_box = []
		all_number = []
		all_label_cls = []
		all_number = []
		all_box = []
  
		result = yolo_number_detection(crop_img,conf = 0.35,imgsz = 640,verbose=False)[0]
		
		for _,box in enumerate(result.boxes.xyxy):
			padding_h = 0
			padding_x = 0
			x1,y1,x2,y2 = list(map(int, box.tolist()))

			y1 = max(2,y1-padding_h)
			x1 = max(2,x1-padding_x)
			y2 = min(crop_img.shape[0],y2+padding_h)
			x2 = min(crop_img.shape[1],x2+padding_x)
			
			number = crop_img[y1-2:y2+2,x1-2:x2+2,:]
			all_number.append(number)
			box = [x1,y1,x2,y2]
			all_box.append(box)
		
			
		if len(all_number) == 0 :
			return None,None,None,None

		top_k = 5 
		_,cls = number_classifier.predict_from_list_image(all_number,top_k=top_k)
		
		top1 = [c[0] for c in cls]
		

		
		for i,b in enumerate(all_box):

			b.append(cls[i])
			all_box[i] = b
		
		
		lp,crop_img = get_final_plate(all_box,crop_img)
		lp = lp.replace("-","")
  
		return lp,crop_img,all_box,[all_label_cls,top1]
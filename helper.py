import math
import cv2
import numpy as np

def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result
def compute_skew(src_img, center_thres):
	if len(src_img.shape) == 3:
		h, w, _ = src_img.shape
	elif len(src_img.shape) == 2:
		h, w = src_img.shape
	else:
		print('upsupported image type')
	img = cv2.medianBlur(src_img, 3)
	edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
	lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 1.5, maxLineGap=h/3.0)
	if lines is None:
		return 1

	min_line = 100
	min_line_pos = 0
	for i in range (len(lines)):
		for x1, y1, x2, y2 in lines[i]:
			center_point = [((x1+x2)/2), ((y1+y2)/2)]
			if center_thres == 1:
				if center_point[1] < 7:
					continue
			if center_point[1] < min_line:
				min_line = center_point[1]
				min_line_pos = i

	angle = 0.0
	nlines = lines.size
	cnt = 0
	for x1, y1, x2, y2 in lines[min_line_pos]:
		ang = np.arctan2(y2 - y1, x2 - x1)
		if math.fabs(ang) <= 30: # excluding extreme rotations
			angle += ang
			cnt += 1
	if cnt == 0:
		return 0.0
	return (angle / cnt)*180/math.pi

def changeContrast(img):
	lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l_channel, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
	cl = clahe.apply(l_channel)
	limg = cv2.merge((cl,a,b))
	enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	return enhanced_img

def deskew(src_img, change_cons, center_thres):
	if change_cons == 1:
		return rotate_image(src_img, compute_skew(changeContrast(src_img), center_thres))
	else:
		return rotate_image(src_img, compute_skew(src_img, center_thres))
	
# license plate type classification helper function
def linear_equation(x1, y1, x2, y2):
	b = y1 - (y2 - y1) * x1 / (x2 - x1)
	a = (y1 - b) / x1
	return a, b
def predict_point(x,point1,point2):
	x1,y1= point1
	x2,y2 = point2
	a,b = linear_equation(x1,y1,x2,y2)
	return a*x+b

def check_point_linear(x, y, x1, y1, x2, y2,threshold = 20):
	a, b = linear_equation(x1, y1, x2, y2)
	y_pred = a*x+b
	return(math.isclose(y_pred, y, abs_tol = threshold))

def get_final_plate(bb_list,image,real_label = None):
	center_list = []
	y_mean = 0
	y_sum = 0
	names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','S','T','U','V','X','Y','Z','0']
	names = sorted(names)
 
	for i,bb in enumerate(bb_list):
	   
		x_c = (bb[0]+bb[2])/2
		y_c = (bb[1]+bb[3])/2
		y_sum += y_c
		
		class_name =bb[-1]
		
		center_list.append([x_c,y_c,class_name,bb])
	LP_type = 1
	# find 2 point to draw line
	l_point = center_list[0]
	r_point = center_list[0]
	for cp in center_list:
		if cp[0] < l_point[0]:
			l_point = cp
		if cp[0] > r_point[0]:
			r_point = cp

	for ct in center_list:
		if l_point[0] != r_point[0]:
			if (check_point_linear(ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]) == False):
				LP_type = "2"

	y_mean = int(int(y_sum) / len(bb_list))
   

	# 1 line plates and 2 line plates
	line_1 = []
	line_2 = []
	license_plate = ""
	
	if LP_type == "2":
		
		for c in center_list:
			if int(c[1]) > y_mean:
				line_2.append(c)
			else:
				line_1.append(c)
 
		i = 0 
		oto = True if  len(line_1) == 3 else False
  
		for l1 in sorted(line_1, key = lambda x: x[0]):
			
			class_index = l1[2]
			class_names_index  = [names[index] for index in class_index]
			if i == 2:
				class_names = [name for name in class_names_index if not name.isdigit()]

			elif i == 3:
				if not oto:
					class_names = [name for name in class_names_index]
				else:
					class_names = [name for name in class_names_index if name.isdigit()]
			else:
				
				
				class_names = [name for name in class_names_index if name.isdigit()]
			  

			try:
				class_name = class_names[0]
			except:
				class_name = class_names_index[0]
    
			i+=1
			license_plate += str(class_name)
			
		license_plate += "-"
		for l2 in sorted(line_2, key = lambda x: x[0]):
		   
			class_index = l2[2]
			class_names_index  = [names[index] for index in class_index]
			
			class_names = [name for name in class_names_index if name.isdigit()]
	
			try:
				class_name = class_names[0]
			except:
				class_name = class_names_index[0]
		 
			license_plate += str(class_name)
			
	else:
		
		for i,l in enumerate(sorted(center_list, key = lambda x: x[0])):
			
			class_index = l[2]
			class_names_index  = [names[index] for index in class_index]
			if i!= 2:
				class_names = [name for name in class_names_index if name.isdigit()]
			else:
				class_names = [name for name in class_names_index if not name.isdigit()]
			try:
				class_name = class_names[0]
			except:
				class_name = class_names_index[0]
    
			license_plate += str(class_name)
		
	
	return license_plate,image
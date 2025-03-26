
import cv2
from model import  get_text_from_plate,get_plate
from utils import ProcessImageV2, convert_float_to_int, crop_img
from PIL import Image
import numpy as np
import gradio as gr

# Initialize the image processor
resize = ProcessImageV2(size=1240, color='black')

# Set the GRADIO_TEMP_DIR environment variable if not already set
import os 
# if "GRADIO_TEMP_DIR" not in os.environ:
# 	# Default to a user-writable directory if not specified
# 	default_temp_dir = os.path.join(os.path.expanduser("~"), "gradio_temp")
# 	os.makedirs(default_temp_dir, exist_ok=True)
# 	os.environ["GRADIO_TEMP_DIR"] = default_temp_dir
 
def get_plate_text(img):
	try:
		# Convert numpy array to PIL image
		pil_img = Image.fromarray(img)
		# Resize image to 1240
		pil_img = resize(pil_img)
		all_location,all_predict_text = [],[]
		# Predict plate location and get processed image
		for plate_location in  get_plate(img):
		
			# Convert float coordinates to integer
			plate_location = convert_float_to_int(plate_location)
			all_location.append(plate_location)
			x1, y1, x2, y2 = plate_location
			
			# Crop the plate image
			plate = crop_img(img, plate_location)
			
			# Predict text from license plate
			predict_text, _, _, _ = get_text_from_plate(plate)
			all_predict_text.append(predict_text)
		for plate_location,predict_text in zip(all_location,all_predict_text):
			x1,y1,x2,y2 = plate_location
			# Draw rectangle around plate
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			
			# Add text at top-left corner (x1, y1)
			font = cv2.FONT_HERSHEY_SIMPLEX
			font_scale = 1
			color = (0, 255, 0)  # Green text
			thickness = 2
			text_position = (x1, y1 - 10)  # Slightly above the top-left corner
			if text_position[1] < 0:  # Ensure text stays within image bounds
				text_position = (x1, y1 + 20)
			
			cv2.putText(
				img, 
				predict_text, 
				text_position, 
				font, 
				font_scale, 
				color, 
				thickness
			)
		
		return img
		
	except Exception as e:
		# Create error text on image
		error_img = img.copy()
		cv2.putText(
			error_img,
			f"Error: {str(e)}",
			(10, 50),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,
			(0, 0, 255),  # Red text for error
			2
		)
		return error_img

# Updated Gradio interface to handle the image output
def gradio_interface(image):
	# Convert Gradio image (PIL format) to numpy array in BGR format
	img = np.array(image)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	# Get processed image with visualization
	result_img = get_plate_text(img)
	# Convert back to RGB for Gradio display
	result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
	return result_img

# Define Gradio interface with image output
interface = gr.Interface(
	fn=gradio_interface,
	inputs=gr.Image(type="pil", label="Upload Vehicle Image"),
	outputs=gr.Image(type="numpy", label="Processed Image with Plate Text"),
	title="License Plate Text Recognition",
	description="Upload an image of a vehicle to see the license plate location and text",
	examples=["/work/21013187/phuoc/Image_Captionning_Transformer/data/bien_f/example1.jpg"]
)

if __name__ == "__main__":
	
	interface.launch(
		server_name="0.0.0.0",
		server_port=7861
	)
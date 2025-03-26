import os 
from inception import INCEPTION
from PIL import Image
import torch
import cv2
from torchvision import transforms
from utils import ProcessImageV2

names = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','K','L','M','N','P','S','T','U','V','X','Y','Z','0']
names = sorted(names)

class NumberClassifier(object):
	def __init__(self, device,
              			pretrained_path ,
                 		image_size = 180):
		
   
		self.device = device
		self.model = INCEPTION().to(device)
		self.pretrained_path = pretrained_path
		print(f"Device: {self.device}")
		self.image_size = image_size
  
		self.load_pretrained_model()
		self.model.eval()
		self.transform = transforms.Compose([ProcessImageV2(self.image_size),
										transforms.ToTensor()])
  
		
	def load_pretrained_model(self):
		pretrained_path = self.pretrained_path
	  
		state_dict = torch.load(pretrained_path, map_location=self.device)
		model_state_dict = state_dict['model']
		self.model.load_state_dict(state_dict=model_state_dict)
		# print(f"Load pretrained model from {pretrained_path}")

	
	def preprocessing_image(self,image_pil):
		return self.transform(image_pil).unsqueeze(0)

	def predict(self, image_tensor, top_k):
		with torch.no_grad():
			image_tensor = image_tensor.to(self.device)
			predict = self.model(image_tensor).squeeze()
			predict = torch.softmax(predict, dim=-1)
			
		# Get the top-k predictions and their respective probabilities
		conf, prediction = torch.topk(predict, top_k)

		return conf, prediction


	def preprocess_image_cv2(self,image):
		image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).convert('RGB')
		image_tensor = self.preprocessing_image(image)
		image_tensor = image_tensor.to(self.device)
		return image_tensor
  
	def predict_from_list_image(self,image_list,top_k ):
		image_list = [self.preprocess_image_cv2(image) for image in image_list ]
		with torch.no_grad():
			conf,predict = self.predict(torch.cat(image_list, dim=0).to(self.device),top_k=top_k)
		conf = conf.squeeze()
		predict = predict.squeeze()
		return conf.tolist(), predict.tolist()
	


def get_number_classifier( **kwargs):
	return NumberClassifier(**kwargs)
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import numpy as np
import torch
import PIL
from torchvision import datasets, transforms
import os
import json
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Model():
	def __init__(self):
		self.mtcnn =  MTCNN(
			image_size=160, margin=0, min_face_size=20,
			thresholds=[0.4, 0.5, 0.5], factor=0.709, post_process=True,
			select_largest = True,
			device=device)

		self.CNN = InceptionResnetV1(
			classify=False,
			pretrained='vggface2'
			).to(device)

		self.preprocessing = transforms.Compose([
			transforms.Resize((160, 160)),
			np.float32,
			transforms.ToTensor(),
			fixed_image_standardization
			])

		self.id=None

	def detect(self, img, save_img=False, save_path = './Face/'):
		box,_ = self.mtcnn(img)
		detect_im = img.crop(box[0])
		if save_img==True:
			save_img.save(save_path+self.id+'.jpg')
		return box[0], detect_im


	def create_vector(self, img):
		embed_vector = self.CNN(self.preprocessing(img).unsqueeze(0).to(device))[0]
		return embed_vector

	def save_embed_vector(self, vector, save_path = './Embed_Vectors/'):
		while True:
			name = str(random.randint(0, 1000))
			name = (4-len(name))*'0' + name + '.pt'
			try:
				f = open(save_path+name)
				continue
			except:
				torch.save(vector, save_path+name)
				break
		return



from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import numpy as np
import torch
import PIL
from torchvision import datasets, transforms
import os
import json
import random
import cv2


class Model:
	def __init__(self):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.mtcnn =  MTCNN(
			image_size=160, margin=0, min_face_size=20,
			thresholds=[0.4, 0.5, 0.5], factor=0.709, post_process=True,
			select_largest = True,
			device=self.device)

		self.CNN = InceptionResnetV1(
			classify=False,
			pretrained='vggface2'
			).to(self.device)

		self.preprocessing = transforms.Compose([
			transforms.Resize((160, 160)),
			np.float32,
			transforms.ToTensor(),
			fixed_image_standardization
			])

	def detect(self, img, save_img=False, save_path = './Face/', down_sample=2):
		w, h = img.shape[1], img.shape[0]
		new_w, new_h= w//down_sample, h//down_sample
		sample = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR )
		try:
			box, _ = self.mtcnn.detect(sample)
			box = [int(ele) for ele in box[0]]

			ori_box= [ele*down_sample for ele in box]
			detect_im = img[box[1]: box[3]-box[1], box[0]: box[2]- box[0]]
			if save_img==True:
				save_img.save(save_path+self.id+'.jpg')
			return ori_box, detect_im, None
		except Exception as e:
			return None, None, e


	def create_vector(self, img):
		embed_vector = self.CNN(self.preprocessing(img).unsqueeze(0).to(self.device))[0]
		return embed_vector

	def save_embed_vector(self, vector, name, save_path = './Embed_vector.json'):
		f= open(save_path)
		file = json.loads(f)
		n_line = len(f.readlines())
		while True:
			id=n_line-1
			id = (4-len(id))*'0' + id
			try:
				f = open(save_path)
				continue
			except:
				vector_list = vector.tolist()
				dic = {"id":id, "name":name, "vector":vector_list}
				file.update(dic)
				json.dumps(file, f)
				break

		f.close()
		return



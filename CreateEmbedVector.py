from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import numpy as np
import torch
import PIL
from torchvision import datasets, transforms
import os
import json
import cv2
import os, sys

def convert_i_to_id(x):
	#string is return, be aware
	return (4 - len(str(x))) * '0' + str(x)

def convert_id_to_i(id):
	return int(id)

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
		try:
			w, h = img.shape[1], img.shape[0]
		except:
			(w, h) = img.size
		new_w, new_h= w//down_sample, h//down_sample
		sample = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_LINEAR )
		try:
			box, _ = self.mtcnn.detect(sample)
			box = [int(ele) for ele in box[0]]

			ori_box= [ele*down_sample for ele in box]
			detect_im = img[box[1]: box[3], box[0]: box[2]]

			#convert to PIL for easy to handle
			detect_im = PIL.Image.fromarray(detect_im.astype('uint8'), 'RGB')
			return ori_box, detect_im, None
		except Exception as e:
			return None, None, e


	def create_vector(self, img):
		self.CNN.eval()
		embed_vector = self.CNN(self.preprocessing(img).unsqueeze(0).to(self.device))
		return embed_vector

	def save_embed_vector(self, list_id, vector=None, id=None, name="Test name", save_path = 'Embed_vector.json', img_path="Face"):
		#Expect id to be string, example: "0001"
		if getattr(sys, 'frozen', False):
			application_path = os.path.dirname(sys.executable)
		elif __file__:
			application_path = os.path.dirname(__file__)

		#Save vector and save the attendance status of person
		save_path = os.path.join(application_path, save_path)
		f= open(save_path, 'w')

		json.dump(list_id, f)

		#Save vector only
		if vector!=None:
			tensor_file_name = "Embed_Vectors/" + id + "_" + name + ".pt"

			tensor_file_name = os.path.join(application_path, tensor_file_name)
			torch.save(vector, tensor_file_name)

		f.close()
		return



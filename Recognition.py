import json
import torch
import os
from torch.nn import CosineSimilarity
import numpy as np

class Recognition:
	def __init__(self):
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.score_fn = CosineSimilarity(dim=0, eps = 1e-5)
		self.list_vector = []
		self.list_id = []

	def update_List(self, vector_path="./Embed_Vectors/"):
		self.list_vector = []
		self.list_id = []
		for filename in os.listdir(vector_path):
			self.list_vector.append(torch.load(vector_path + filename, map_location=self.device))
		f = open("./Embed_vector.json")
		self.list_id = json.load(f)
		print("Num vec:", len(self.list_vector))
		return

	def Best_match(self, vector):

		#read the json file
		scores = []
		for ele in self.list_vector:
			scores.append(self.score_fn(vector, ele[0]).detach().numpy())

		match = np.argmax(scores)
		conf_score = scores[match]
		#print("score: ", scores)
		if conf_score<0.6:
			#label unknow
			return -1
		else:
			#return the best match
			return match



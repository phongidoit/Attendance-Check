import json
import torch
from torch.nn import CosineSimilarity

class Recognition():
	def __init__(self):
		self.score_fn = CosineSimilarity(dim=0, eps = 1e-5)
		#self.
		pass

	def Best_match(self, vector, embed_vector_path = "./", id_name_path="./"):
		f = open(embed_vector_path)
		file = json.load(s)

		#read the json file
		scores = []
		for ele in embed_vectors:
			scores.append(self.score_fn(vector, ele))

		match = np.argmax(scores)
		conf_score = score[match]
		if conf_score<0.6:
			#label unknow
			pass
		else:
			#top 5 highest score
			top_5_id = score.argsort()[-5:][::-1]


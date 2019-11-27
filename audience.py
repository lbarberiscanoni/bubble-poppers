import numpy as np
from scipy import stats
import networkx as nx
from user import *
import pickle
import json
from tqdm import tqdm
from tensorforce.agents import PPOAgent
import numpy as np
import itertools

class Audience:

	def __init__(self, population_size, distribution):

		self.content = {}

		index = 0

		#841 items
		for x in range(1, distribution):
			for y in range(1, distribution):
				item = [x, y]
				self.content[index] = item
				index += 1

		#graph of users 
		self.user_graph = np.full((population_size, population_size), 0)

		#graph of items
		self.item_graph = np.full((len(self.content), len(self.content)), 0)

		self.graph = np.full((population_size, len(self.content)), 0)
		self.user_base = {}
		self.item_dimensions = len(self.content[list(self.content.keys())[0]])

		for i in range(population_size):
			local_user = User(i, self.item_dimensions)
			self.user_base[i] = local_user

	def recommendation(self, user_id, item_id):
		if self.graph[user_id][item_id] == 1:
			reward = -5
		else:
			reward = self.user_base[user_id].evaluate(self.content[item_id])
			self.graph[user_id][item_id] = 1
			# update user graph
			occurrences = [i for i,val in enumerate(self.graph[:, item_id]) if val==1]
			for permutation in list(itertools.permutations(occurrences)):
				# print(permutation)
				if len(permutation) < 2:
					self.user_graph[permutation[0], permutation[0]] += 1
				else:
					self.user_graph[permutation[0], permutation[1]] += 1

			#update item graph
			# print(self.graph[:, item_id])

		return reward

	def clustering(self):
		# a = np.reshape(np.random.random_integers(0,0,size=900),(30,30))
		D = nx.DiGraph(self.user_graph)
		return nx.average_clustering(D)

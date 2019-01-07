import numpy as np
from scipy import stats
import networkx as nx
from user import *
import pickle
import json
from tqdm import tqdm
from tensorforce.agents import PPOAgent
import numpy as np

class Audience:

	def __init__(self, population_size):

		self.content = {}

		distribution = range(10)

		index = 0
		for x in distribution:
			for y in distribution:
				for z in distribution:
					for w in distribution:
						item = [x, y, z, w]
						self.content[index] = item
						index += 1

		self.graph = np.full((population_size, len(self.content)), 0)
		self.user_base = {}
		self.item_dimensions = len(self.content[self.content.keys()[0]])

		for i in range(population_size):
			local_user = User(i, self.item_dimensions)
			self.user_base[i] = local_user

	def recommendation(self, user_id, item_id):
		reward = self.user_base[user_id].evaluate(self.content[item_id])
		self.graph[user_id][item_id] = 1

		return reward
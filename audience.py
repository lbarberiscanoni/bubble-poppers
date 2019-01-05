import numpy as np
from scipy import stats
import networkx as nx
from user import *
import pickle
import json
from tqdm import tqdm
from tensorforce.agents import PPOAgent

class Audience:

	def __init__(self, population_size, content):
		self.graph = nx.Graph()
		self.user_base = {}
		self.item_dimensions = len(content[content.keys()[0]])

		for i in range(population_size):
			local_user = User(i, self.item_dimensions)
			self.user_base[i] = local_user
			self.graph.add_node(i)

		self.content = content

		for content_index in content.keys():
			self.graph.add_node(content_index)

distribution = range(10)

content = {}

index = 100
for x in tqdm(distribution):
	for y in distribution:
		for z in distribution:
			for w in distribution:
				item = [x, y, z, w]
				content[index] = item
				index += 1

G = Audience(20, content)

matrix = nx.to_numpy_matrix(G.graph)

print("matrix made")

agent = PPOAgent(
    states={"type":'float', "shape": matrix.shape },
    actions={
    	str(1): dict(type="int", num_actions=3)
    },
    network=[
	    dict(type='flatten'),
	    dict(type="dense", size=32),
	   	dict(type="dense", size=32),
	   	dict(type="dense", size=32)
    ],
)

print("agent ready")

action = agent.act(matrix)
print(action)




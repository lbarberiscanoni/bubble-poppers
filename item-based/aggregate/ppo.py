import numpy as np
from scipy import stats
import networkx as nx
from user import *
import pickle
import json
from tqdm import tqdm
from tensorforce.agents import PPOAgent
import numpy as np
from audience import *

G = Audience(20)

print(G.graph.shape)

agent = PPOAgent(
    states={"type":'float', "shape": G.graph.shape },
    actions={
    "user": dict(type="int", num_actions=G.graph.shape[0]),
    "item": dict(type="int", num_actions=G.graph.shape[1])
    },
    network=[
	    dict(type='flatten'),
	    dict(type="dense", size=32),
    ],
)

print("agent ready")

for i in range(1000):
	action = agent.act(G.graph)

	reward = G.recommendation(action["user"], action["item"])

	print(i, action, reward)

	agent.observe(reward=reward, terminal=False)
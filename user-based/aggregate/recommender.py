import numpy as np
from scipy import stats
import networkx as nx
from user import *
import pickle
import json
from tqdm import tqdm
from tensorforce.agents import PPOAgent, DQNAgent, VPGAgent
import numpy as np
from audience import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--agent", help="select an agent type [ppo, vpg, dqn]")
parser.add_argument("--process", help="select process [train, test]")

args = parser.parse_args()

G = Audience(20, 10)

print("graph shape", G.graph.shape)

if args.agent == "ppo":
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
elif args.agent == "dqn":
    agent = DQNAgent(
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

elif args.agent == "vpg":
    agent = VPGAgent(
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

print("agent ready", agent)

if args.process == "train":
    epochs = 1
    for epoch in range(epochs):
        #100 reccomendations for every user
        training_size = G.graph.shape[0] * 100
        for step in range(training_size):
            action = agent.act(G.graph)

            print(G.graph)

            reward = G.recommendation(action["user"], action["item"])

            print(epoch, step, action, reward)

            if step < training_size:
                agent.observe(reward=reward, terminal=False)
            else:
                agent.observe(reward=reward, terminal=True)   
import numpy as np
from scipy import stats
import networkx as nx
from user import *
import pickle
import json
from tqdm import tqdm
from tensorforce.agents import PPOAgent, DQNAgent, VPGAgent, TRPOAgent, DRLAgent, DDPGAgent, NAFAgent, DQFDAgent
import numpy as np
from audience import *
import argparse
import copy
from tensorforce.agents import VPGAgent

parser = argparse.ArgumentParser()
parser.add_argument("--agent", help="select an agent type [ppo, vpg, dqn]")
parser.add_argument("--process", help="select process [train, test]")

args = parser.parse_args()

G = Audience(30, 10)

# print("graph shape", G.graph.shape)




if args.agent == "ppo":
    agent = PPOAgent(
        states={"type":'float', "shape": G.graph.shape },
        actions={
        "user": dict(type="int", num_values=G.graph.shape[0]),
        "item": dict(type="int", num_values=G.graph.shape[1])
        },
        network=[
    	    dict(type='flatten'),
    	    dict(type="dense", size=32),
        ],
        memory=10000,
    )
elif args.agent == "dqn":
    agent = DQNAgent(
        states={"type":'float', "shape": G.graph.shape },
        actions={
        "user": dict(type="int", num_values=G.graph.shape[0]),
        "item": dict(type="int", num_values=G.graph.shape[1])
        },
        network=[
            dict(type='flatten'),
            dict(type="dense", size=32),
        ],
        memory=10000,
    )

elif args.agent == "vpg":
    agent = VPGAgent(
        states={"type":'float', "shape": G.graph.shape },
        actions={
        "user": dict(type="int", num_values=G.graph.shape[0]),
        "item": dict(type="int", num_values=G.graph.shape[1])
        },
        network=[
            dict(type='flatten'),
            dict(type="dense", size=32),
        ],
        memory=10000,
    )
elif args.agent == "trpo":
    agent = TRPOAgent(
        states={"type":'float', "shape": G.graph.shape },
        actions={
        "user": dict(type="int", num_values=G.graph.shape[0]),
        "item": dict(type="int", num_values=G.graph.shape[1])
        },
        network=[
            dict(type='flatten'),
            dict(type="dense", size=32),
        ],
        memory=10000,
    )



print("agent ready", agent)
new_agent = copy.deepcopy(agent)
agent.initialize()


if args.process == "train":
    epochs = 1000000
    for epoch in tqdm(range(epochs)):
        #20 reccomendations for every user
        training_size = G.graph.shape[0] * 20
        for step in range(training_size):
            action = agent.act(G.graph)

            #print(G.graph)

            reward = G.recommendation(action["user"], action["item"])

            #print(epoch, step, action, reward)

            if step < training_size:
                agent.observe(reward=reward, terminal=False)
            else:
                agent.observe(reward=reward, terminal=True)   

    # agent.save(directory="/Users/lbarberiscanoni/Lorenzo/Github/bubble-poppers/user-based/aggregate/saved", filename=None)
    # print("agent saved")
    # agent.close()
    # new_agent.restore(directory="/Users/lbarberiscanoni/Lorenzo/Github/bubble-poppers/user-based/aggregate/saved", filename=None)
    # print("restored")
    agent.save(directory="saved", filename=None)
    print("agent saved")
    agent.close()
    new_agent.restore(directory="saved", filename=None)
    print("restored")
if args.process == "test":
    agent.restore(directory="/Users/lbarberiscanoni/Lorenzo/Github/bubble-poppers/user-based/aggregate/saved", filename=None)


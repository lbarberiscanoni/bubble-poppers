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
import os
import subprocess

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

try:
    lastEpoch = int(os.listdir("saved/" + args.agent)[2].split("-")[0])

    agent.restore(directory="saved/" + args.agent)
    print("restored")
except:
    lastEpoch = 0
    

if args.process == "train":
    
    epochs = 100000
    for epoch in tqdm(range(lastEpoch, epochs)):
        G = Audience(30, 10)

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

        if (epoch % 10) == 0:
            subprocess.call("rm saved/" + args.agent + "/*", shell=True)
            fName = str(epoch) + "-agent"
            agent.save(directory="saved/" + args.agent, filename=fName)
            print("agent saved")
            # agent.close()

if args.process == "test":
    agent.restore(directory="saved/" + args.agent)

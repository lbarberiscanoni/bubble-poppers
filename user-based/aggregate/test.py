import copy
import os
import sys
import unittest

from tensorforce.agents import VPGAgent
from tensorforce.tests.unittest_base import UnittestBase

from audience import *

G = Audience(20, 10)
print(G.graph.shape)
class TestSaving(UnittestBase, unittest.TestCase):


	def saving_prepare(self, name, **kwargs):
		states = dict(type='float', shape=(1,))
		actions = dict(type='int', shape=(), num_values=3)
		agent, environment = self.prepare(name=name, states=states, actions=actions, **kwargs)

		return agent, environment

	def run(self):
		agent = VPGAgent(
			states = dict(type='float', shape=(30,1000,)),
		    #states={"type":'float', "shape": (20, 6561,) },
		    actions={
		    "user": dict(type="int", num_values=G.graph.shape[0]),
		    "item": dict(type="int", num_values=G.graph.shape[1])
		    },
		    network=[
		        dict(type='flatten'),
		        dict(type="dense", size=32),
		    ],
		)
		# #agent, environment = self.saving_prepare(name='explicit-default')
		# restored_agent = copy.deepcopy(agent)
		# agent.initialize()
		# agent.save(directory='/Users/lbarberiscanoni/Lorenzo/Github/bubble-poppers/user-based/aggregate/saved',filename=None)
		agent.restore(directory = '/Users/lbarberiscanoni/Lorenzo/Github/bubble-poppers/user-based/aggregate/saved', filename=None)
		print("Restored -------")

TestSaving().run()


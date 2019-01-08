import numpy as np
from scipy import stats
import networkx

class User:

	def __init__(self, index, item_dimensions):
		#taste coefficients initialized at .25
		self.index = index
		self.taste_coefficients = { x: {"value": .25, "history":[.25, .25, .25, .25, .25]} for x in range(item_dimensions) }

		self.satisfaction_history = [ x for x in np.random.normal(5, 1, 50) ]

	def z_score(self, distribution, value):
		mean = np.mean(distribution)
		std = np.std(distribution)

		return (value - mean) / float(std)

	def update_preferences(self, item):

		total = sum(item)
		print(item, total)
		for i in range(len(item)):
			self.taste_coefficients[i]["history"].append(item[i] / float(total))
			self.taste_coefficients[i]["value"] = np.mean(self.taste_coefficients[i]["history"])


	def evaluate(self, item):
		satisfaction_score = 0
		for i in range(len(item)):
			satisfaction_score += (item[i] * self.taste_coefficients[i]["value"])
		
		z_score = self.z_score(self.satisfaction_history, satisfaction_score)

		self.satisfaction_history.append(satisfaction_score)

		self.update_preferences(item)

		return z_score
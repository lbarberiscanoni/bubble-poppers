import numpy as np
from tqdm import tqdm
import pickle
import json

distribution = range(100)

content = {}

index = 100
for x in tqdm(distribution):
	for y in distribution:
		for z in distribution:
			for w in distribution:
				item = [x, y, z, w]
				content[index] = item
				index += 1

with open('content.pkl','w') as f:
    pickle.dump(json.dumps(content), f)

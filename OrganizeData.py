from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

data = {}
data['name'] = []
data['activity'] = []
data['descriptors'] = np.ndarray(shape = (37241, 9491), dtype = np.float32)
with open('data/TrainingSet/ACT1_competition_training.csv') as csvfile:
	rows = csvfile.read().splitlines()
	for i in range(1, len(rows)):
		row = rows[i]
		line = row.split(',')
		data['name'].append(line[0])
		data['activity'].append(float(line[1]))
		data['descriptors'][i-1]=np.array(line[2:]).astype(np.float32)
		print (i)
data['activity'] = np.asarray(data['activity'])
data['name'] = np.asarray(data['name'])
with open('training_1.pickle', 'wb') as f:
	pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
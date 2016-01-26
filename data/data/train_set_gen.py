#!/usr/bin/python
import numpy as np
from os import listdir
from os.path import isfile, join
import re
from copy import deepcopy
from itertools import chain
import cPickle as p

# /afs/inf.ed.ac.uk/user/s12/s1235260/.local/lib/python2.7/site-packages/sklearn/neural_network/
# model3.pkl is the current benchmark with 0.93475877193 acc
class ExerciseDataProvider:
	"""
	The following data provider is here to parse
	the tree-folder styled data set in to a dictionary
	which can be then used as a proper training set.

	# yoloswags
	"""

	class_dict = {
					'a01': 'sitting',
					'a02': 'standing',
					'a03': 'lying_on_back',
					'a04': 'lying_on_right_side',
					'a05': 'ascending_stairs',
					'a06': 'descending_stairs',
					'a07': 'standing_in_an_elevator_still',
					'a08': 'moving_around_in_an_elevator',
					'a09': 'walking_in_a_parking_lot',
					'a10': 'walking_on_a_treadmill_with_a_speed_of_4km/h_in_flat',
					'a11': 'deg_inclined_positions',
					'a12': 'running_on_a_treadmill_with_a_speed_of_8km/h',
					'a13': 'exercising_on_a_stepper',
					'a14': 'exercising_on_a_cross_trainer',
					'a15': 'cycling_on_an_exercise_bike_horizontal',
					'a16': 'cycling_on_an_exercise_bike_vertical',
					'a17': 'rowing',
					'a18': 'jumping',
					'a19': 'playing_basketball'
	}


	def __init__(self,
				 base_path="."):
		self.base_path = base_path
		self.data_dict = self.get_data_dict()
		self.x , self.t, self.xt, self.tt = self.p()
		# ((self.x , self.t),(self.xt , self.tt)) = self.prepare_train_set()

	def p(self):
		X = np.array(map(lambda x: x[0],
		             self.data_dict.values()))
		X = X.reshape(-1,X.shape[-1])

		T = np.array(map(lambda x: x[1],
		             self.data_dict.values())).\
					 reshape(-1)
		pr = np.random.permutation(len(X))
		T = T[pr]
		X = X[pr]

		percent = int(len(X)*0.8)
		Xt = X[percent:, :]
		X = X[:percent, :]
		Tt = T[percent:]
		T = T[:percent]

		return X, T, Xt, Tt


	def get_data_dict(self):
		mypath_on = self.base_path
		folds = [f
				 for f
				 in listdir(mypath_on)
				 if re.match('a*',f).group(0)]
		files_dict = dict()
		for fold in folds:
			people_path = mypath_on + "/" + fold
			people =[f for f in listdir(people_path)]
			files = list()
			for p in people:
				file_path = join(people_path, p)
				files.append([join(file_path,f)
							  for f in listdir(file_path)
							  if isfile(join(file_path,f))])
			files = list(chain(*files))
			parsed_files = list()
			for f in files:
				with open(f,'rb') as fo:
					ford = np.array(map(lambda x:map(float, x.split(",")),
									fo.read().split("\n")[:-1]))
					# print ford.shape
					parsed_files.append(deepcopy(np.array(ford).T.ravel() ))
			assert len(parsed_files) == len(files)
			labl =  int(''.join((x for x in fold if x.isdigit())))
			# print self.class_dict[fold]
			# print np.array(parsed_files).shape
			# hkjfdhfdkjhfds
			# print (np.array(parsed_files),
			# 	   np.array([labl]*len(files)) )
			files_dict[fold] = (np.array(parsed_files),
							    np.array([labl]*len(files)))
		return files_dict





	def randomize(self):
	   raise BaseException("TODO")






if  __name__ == "__main__":
	from sklearn.neural_network.multilayer_perceptron import MLPClassifier
	import sys
	import warnings
	warnings.filterwarnings("ignore", category=DeprecationWarning)
	# bbbb
	mode = sys.argv[1]
	library = 'mine'
	if library != 'mine':
		if mode == 'train':
			print "training"
			obj = ExerciseDataProvider(".")
			X = obj.x[:,0:125]
			y = obj.t
			Xt = obj.xt[:,0:125]
			yt = obj.tt
			print "input vec shape: ", X.shape
			# print y.shape
			# print X.shape[-1]
			clf_t = MLPClassifier(algorithm='l-bfgs',
			                      alpha=1e-5,
				     			  hidden_layer_sizes=(X.shape[-1], 19),
								  random_state=1,
								  spectral_mode='fft')
			clf_t.fit(X, y)

			with open('/afs/inf.ed.ac.uk/user/s12/s1235260/model_spec3.pkl', 'wb') as m:
				p.dump((clf_t, Xt, yt) , m)

		else:
			with open('/afs/inf.ed.ac.uk/user/s12/s1235260/model_spec3.pkl', 'rb') as m:
				clf, Xt, yt = p.load(m)
			y2 = clf.predict(Xt)
			print clf.coefs_[0].shape #.shape
			print y2, yt
			print len(y2), len(yt)
			acc = sum(y2==yt) / float(len(y2))
			print acc
	    #"""
	else:
		if mode == 'train':
			print "training"
			obj = ExerciseDataProvider(".")
			X = obj.x[:,0:125]
			y = obj.t
			Xt = obj.xt[:,0:125]
			yt = obj.tt
		else:
			pass

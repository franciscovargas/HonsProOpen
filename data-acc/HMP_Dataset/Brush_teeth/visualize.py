import numpy as np
from numpy import dot
from re import split
import matplotlib.pyplot as plt

def euclidean(ndarray):
	return np.sqrt(dot(ndarray, ndarray.T))[:,0]

if __name__ == "__main__":
	fil = filter( lambda x: x != '',
		         split("[\r\n]",
		         	   open("Accelerometer-2011-04-11-13-28-18-brush_teeth-f1.txt",
		         	   	    'r').read())[:-1])

	fil = np.array(map(lambda x: map(int,split("\s", x)), fil))
	euclid = euclidean(fil)
	# print(fil.T)
	plt.plot(range(len(euclid)), euclid)
	plt.show()
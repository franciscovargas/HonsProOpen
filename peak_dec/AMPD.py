import numpy as np
from numpy import array
import numpy.linalg as la
import math as m
import matplotlib.pyplot as plt

def least_squares(t, x):
	"""
	x = mt + c fit
	"""
	design_mat = np.vstack([t, np.ones(len(t))]).T
	m, c = la.lstsq(design_mat, x)[0]
	return m*t + c

def LMS_calc(t, x, alpha=1):
	assert len(t) == len(x)
	n = len(x)
	remove = x - least_squares(t, x)
	k_max = m.ceil(0.5 * n) - 1
	# read in to numpy boleean masks

def main():
	# sinc func signal
	x = np.arange(-15, 15, 0.0005)
	noise = 0.1*np.random.randn(len(x))
	y = np.sinc(0.5 * x) + 0*noise
	plt.plot(x, y, 'b')
	plt.plot(x , least_squares(x, y), 'r')
	remove_line = y - least_squares(x, y)
	plt.plot(x, remove_line, 'y')
	plt.show()

if __name__ == "__main__":
	main()
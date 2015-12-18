import numpy as np
from numpy import array
import numpy.linalg as la
import math as m
import matplotlib.pyplot as plt
import random

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
	x_prime = x - least_squares(t, x)
	k_max = int(m.ceil(0.5 * n) - 1)
	# read in to numpy boleean masks
	print(k_max, n)
	M = np.zeros((k_max, n))

	for k in range(k_max):
		#print k
		for i in range(k+1, n - k):
			r = random.randint(0,1000) / 1000.0
			i_tilde =  i
			#print(i_tilde  ,i_tilde - k -1 , i_tilde-1 , i_tilde+ k -2)
			if(not x[i_tilde]  > x[i_tilde - k -1] and
			  not x[i_tilde-1] > x[i_tilde+ k -1]):
			  M[k, i_tilde] = r + alpha
	gamma = np.sum(M, axis = 1)
	print len(gamma)
	# exit()
	print gamma
	# exit()
	print M
	print gamma
	print max(gamma)
	print min(gamma)
	print np.min(gamma)
	lambd = np.argmin(gamma)
	print lambd
	M_r = M[0:lambd+1:,]
	print M_r
	# exit()
	std_M_r = np.std(M_r, axis =0)
	print std_M_r
	print len(std_M_r)
	print min(std_M_r)





def main():
	# sinc func signal
	x = np.arange(-15, 15, 0.05)
	print len(x)
	# exit()
	noise = 0.1*np.random.randn(len(x))
	y = np.sinc(0.5 * x) + 0.0*noise
	plt.plot(x, y, 'b')
	plt.plot(x , least_squares(x, y), 'r')
	remove_line = y - least_squares(x, y)
	LMS_calc(x, y)
	plt.plot(x, remove_line, 'y')
	plt.show()

if __name__ == "__main__":
	main()

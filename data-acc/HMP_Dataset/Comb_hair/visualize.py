import numpy as np
from numpy import dot
from re import split
import matplotlib.pyplot as plt

# from wave import *
# import struct

def euclidean(ndarray):
	# print ndarray[:,0]
	s1 = np.sum(ndarray**2,axis=1)
	# print s1
	# exit()
	return np.sqrt(s1)
if __name__ == "__main__":
	fil = filter( lambda x: x != '',
		         split("[\r\n]",
		         	   open("Accelerometer-2011-03-24-09-44-34-comb_hair-f1.txt",
		         	   	    'r').read())[:-1])

	fil = np.array(map(lambda x: map(int,split("\s", x)), fil))
	g = 9.8
	# print fil
	fil = -1.5*g + (fil/63.0)*3*g
	# print fil
	# exit()
	euclid = euclidean(fil)
	# print(fil.T)
	# plt.plot(range(len(euclid)), euclid)
	# x = s1 + s2 + nse # the signal
	dt = (1/32.0)
	t = np.arange(0, 26.03125, dt)
	# print len(t)
	# exit()
	# from scipy import *
	from pylab import *
	NFFT = 64     # the length of the windowing segments
	Fs = int(1.0/dt)  # the sampling frequency
	# exit()
	ax1 = subplot(211)
	plot(t - 128/32, euclid)
	subplot(212, sharex=ax1)

	Pxx, freqs, bins, im = specgram(euclid, NFFT=NFFT, Fs=32, cmap=cm.gist_heat, noverlap=10)
	print(Pxx)
	print(bins)
	print(im)
	print(freqs)
	show()

import numpy as np
%config InlineBackend.figure_format='svg'
from scipy.sparse import coo_matrix
from numpy import arange,pi,exp,sin,cos,ones,inf
from numpy.linalg import norm
from matplotlib.pyplot import figure,loglog,semilogy,text,grid,xlabel,ylabel,title

# This file will contain all three of the different types of solvers I will want to use: Dormand-Prince solver for 2nd order diff eqs, spectral method (pythonized from gr-resonance-tools), and the Prufer shooting method. I will likely have calls to other parts of the code what use the solvers/get the coefficients, but for now, I'm just going to build. Test functions will come later (or maybe in here? Not entirely sure):


class SpectralMethod:
	def __init__(self):

	def example():
		Nvec = 2**arange(3,13)
		for N in Nvec:
			h = 2*pi/N
			


class PruferShooter:
	def __init__(self):

	def mohammad_test_functions(self):
		# (x^3 * y')' + lambda * x * y = 0
		# y'' + 2(lambda-3)/x^2 * y' = 0
		# (e^x * y')' + lambda * e^x * y = 0

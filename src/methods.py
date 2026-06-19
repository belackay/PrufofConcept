import numpy as np

# This file will contain all three of the different types of solvers I will want to use: Dormand-Prince solver for 2nd order diff eqs, spectral method (pythonized from gr-resonance-tools), and the Prufer shooting method. I will likely have calls to other parts of the code what use the solvers/get the coefficients, but for now, I'm just going to build. Test functions will come later (or maybe in here? Not entirely sure):


class PruferShooter:
	def __init__(self):

	def mohammad_test_functions(self):
		# (x^3 * y')' + lambda * x * y = 0
		# y'' + 2(lambda-3)/x^2 * y' = 0
		# (e^x * y')' + lambda * e^x * y = 0

# In this exercise you are asked to write a generally applicable (i.e., different values for applied loads, 
# different boundary conditions, different number of elements) code to solve the problem of the 
# tapered bar under an end load. As an input, your code should take the 
# number of elements,
# material elastic modulus,
# total length of the bar, 
# magnitude of the force applied at the end, 
# and boundary conditions. 
# The shape of the bar is not a variable, the load is always applied on the final node, and each element in a single
# model has the same length. Your code should calculate the 
# displacement of each node, 
# and stress 
# and strains in each element.
# The input file for the code (with the variables that can be changed) is attached to this assignment. Please
# code in either Python or Matlab, for both an input file is provided. You do have to fill in the correct
# numbers for this case, but do not change the form of the input file (i.e., keep the names as given, follow
# the format described for each variable and do NOT add additional variables). Only the functionality of
# the code will be graded, not whether ‘good coding practises’ have been used.

import numpy as np


class Shape:
	def __init__(self, length: float, n: int, w_1: float, w_end: float, t: float):
		self.length = length
		self.n = n
		self.w_1 = w_1
		self.w_end = w_end
		self.t = t
		self.sections = self.compute_sections()

	def compute_sections(self):
		l = self.length / self.n
		h = np.array([self.w_1 - (self.w_1 - self.w_end)/self.n * (i+1) for i in range(self.n)])
		areas = h * self.t
		return areas


def calc_equivalent_stifness(sections: np.array, E: float, L: float, n: int):
	return sections * E / (L/n)


def calcStrain(u: np.array, L: float, n: int):
	eps = np.zeros(len(u)-1)
	for i in range(len(u)-1):
		eps[i] = (u[i+1]-u[i]) / (L/n)
	return eps


def calcStress(eps: np.array, E: float):
	return eps*E


def apply_boundary_conditions(K, F, BC):
	# Convert node numbers to zero-based indices
	node_indices = BC[0] - 1  # Convert to zero-based indexing
	displacements = BC[1]  # Corresponding displacements

	# Make copies of the original stiffness matrix and load vector
	K_mod = np.copy(K)
	F_mod = np.copy(F)

	# Process each boundary condition
	for idx, displacement in sorted(zip(node_indices, displacements), key=lambda x: x[0], reverse=True):
		if displacement != 0:  # Non-zero displacement boundary condition
			# Subtract the column times the applied displacement from the load vector
			F_mod -= K_mod[:, idx] * displacement

		# Remove the row from the stiffness matrix and the corresponding entry from the load vector
		K_mod = np.delete(K_mod, idx, axis=0)
		F_mod = np.delete(F_mod, idx)

		# Remove the column from the stiffness matrix
		K_mod = np.delete(K_mod, idx, axis=1)

	return K_mod, F_mod


def calcDisp(L, Nelem, F, E, BC):
	bar = Shape(L, Nelem, 50, 25, 3.125)
	k = calc_equivalent_stifness(bar.sections, E, L, Nelem)
	# Stiffness matrix
	K = np.zeros((Nelem+1, Nelem+1))
	for i in range(Nelem):
		K[i, i] += k[i]
		K[i, i+1] -= k[i]
		K[i+1, i] -= k[i]
		K[i+1, i+1] += k[i]
	# Load vector
	load = np.zeros(Nelem+1)
	load[0] = -F
	load[-1] = F
	# Boundary conditions
	nodes_modified = [BC[0][i] for i in range(len(BC[0]))]
	K, load = apply_boundary_conditions(K, load, BC)
	u_partial = np.linalg.solve(K, load)
	u = np.zeros(Nelem+1)
	c_mod = 0
	c_not_mod = 0
	for i in range(Nelem+1):
		if i+1 in nodes_modified:
			u[i] = BC[1][c_mod]
			c_mod += 1
		else:
			u[i] = u_partial[c_not_mod]
			c_not_mod += 1
	strain = calcStrain(u, L, Nelem)
	stress = calcStress(strain, E)

	return u, strain, stress
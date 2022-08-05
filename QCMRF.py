"""A QCMRF implementation."""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from scipy.linalg import expm
import itertools

import copy

from qiskit.opflow import I, X, Z, MatrixOp
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ParameterVector

# begin QCMRF

class QCMRF(QuantumCircuit):
	"""Quantum circuit Markov random field."""

	def __init__(
		self,
		cliques=None,
		theta=None,
		gamma=None,
		beta : float = 1,
		name: str = "QCMRF",
		qubit_hack=True,
		with_measurements=True,
		with_barriers=True
	):
		r"""
		Args:
			cliques (List[List[int]]): List of integer lists, representing the clique structure of the
				Markov random field. For a n-dimensional random field, variable indices 0..n-1 are required.
			theta (List[float], optional): The native parameter of the Markov random field.
			gamma (List[float], optional): The alternative circuit parameters of the Markov random field.
			beta (float, optional): inverse temperature, default 1
			name (str): A name of the circuit, default 'QCMRF'
		"""
		self._cliques = cliques
		self._theta = theta
		self._gamma = gamma
		self._beta = beta
		self._name = name
		self._qubit_hack = qubit_hack
		self._with_measurements = with_measurements
		self._with_barriers = with_barriers
		
		if type(self._cliques) != list or type(self._cliques[0]) != list or type(self._cliques[0][0]) != int:
			raise ValueError(
				"The set of clique is not set properly. Type must be list of list of int."
			)
		
		self._num_cliques = len(self._cliques)
		
		self._n = self._cliques[0][0] # first vertex of first clique
		for C in cliques:
			for v in C:
				if v > self._n:
					self._n = v
		self._n += 1
		
		self._dim = 0
		self._c_max = 0
		for C in self._cliques:
			m = len(C)
			if m > self._c_max:
				self._c_max = m
			self._dim += (2**m)
			
			
		if self._theta is not None and len(self._theta) != self._dim:
			raise ValueError(
				"The parameter vector has an incorrect dimension. Expected: " + str(self._dim)
			)

		if self._gamma is not None and len(self._gamma) != self._dim:
			raise ValueError(
				"The QCMRF parameter vector has an incorrect dimension. Expected: " + str(self._dim)
			)

		embedding_aux = 1 if not self._qubit_hack else 0

		super().__init__(self._n + self._num_cliques + embedding_aux, self._n + self._num_cliques, name=name)
		
		self._build()

	@property
	def dimension(self):
		"""The parameter dimension of the Markov random field.

		Returns:
			 int: number of parameters.
		"""
		return self._dim

	@property
	def cliques(self):
		"""Returns the clique set of the Markov random field.

		Returns:
			List[List[int]]: cliques.
		"""
		return self._cliques
		
	@property
	def num_vertices(self):
		"""Returns the number of variables of the Markov random field.

		Returns:
			int: number of vertices.
		"""
		return self._n
		
	num_nodes = num_vertices # alias
		
	@property
	def num_cliques(self):
		"""Returns the number of cliques of the Markov random field.

		Returns:
			int: number of cliques.
		"""
		return self._num_cliques

	@property
	def max_clique(self):
		"""Returns the size of the larges clique of the Markov random field.

		Returns:
			int: size of largest clique.
		"""
		return self._c_max
		
	@property
	def theta(self):
		"""Returns the parameters of the Markov random field.

		Returns:
			List[float]: List of real-valued parameters.
		"""
		if self._theta is None:
			self._theta = []
			for g in self._gamma:
				w = 2*np.log(np.cos(2*g))/self._beta
				self._theta.append(w)

		return self._theta

	@property
	def gamma(self):
		"""Returns the parameters of the Circuit, computed from theta and beta.

		Returns:
			List[float]: List of real-valued parameters.
		"""
		if self._gamma is None:
			self._gamma = []
			for w in self._theta:
				g = 0.5 * np.arccos(np.exp(self._beta*0.5*w))
				self._gamma.append(g)

		return self._gamma

	def sufficient_statistic(self,C,y):
		"""Computes the Pauli-Markov sufficient statistic for clique-state pair (C,y)

		Returns:
			PauliSumOp: opflow representation of Pauli-Markov sufficient statistic
		"""
		result = 1

		plus  = [v for i,v in enumerate(C) if not y[i]] # 0
		minus = [v for i,v in enumerate(C) if     y[i]] # 1)

		for i in range(self._n):
			f = I
			if i in minus:
				f = (I-Z)/2
			elif i in plus:
				f = (I+Z)/2

			result = result^f

		return result
		
	def Hamiltonian(self):
		"""Computes the Hamiltonian

		Returns:
			PauliSumOp: opflow representation of the Hamiltonian
		"""
		H = 0
		i = 0
		for C in self._cliques:
			for y in list(itertools.product([0, 1], repeat=len(C))):
				H += -self.theta[i] * self.sufficient_statistic(C,y)
				i += 1
		return H

	def _conjugateBlocks(self,A):
		"""Returns a block unitary with A and ~A on its diagonal."""
		return (((I+Z)/2)^A) + (((I-Z)/2)^(~A))

	def _build(self):
		"""Return the actual QCMRF."""
		
		num_main_qubits = self._n
		if not self._qubit_hack:
			num_main_qubits += 1

		for i in range(num_main_qubits):
			self.h(i)
		if self._with_barriers:
			self.barrier()

		# init parameters uniformly if none are provided
		if self._theta is None and self._gamma is None:
			self._theta = []
			for i in range(self._dim):
				self._theta.append(np.random.uniform(low=-5.0,high=-0.001))

		i = 0
		for ii,C in enumerate(self._cliques):

			# Construct U^C(gamma(theta_C))
			factor = I^(self._n+1)
			for y in list(itertools.product([0, 1], repeat=len(C))):
				Phi = self.sufficient_statistic(C,y)
				U = (X^((I^(self._n))-Phi)) + (Z^Phi)
				RZ = (-self.gamma[i] * Z).exp_i() ^ (I^self._n)
				Ugam = (RZ @ U)**2
				factor = Ugam @ factor
				i += 1

			# The "qubit hack" allows us to numerically remove the qubit required for
			# the unitary embedding. This does neither alter the output distribution
			# nor does it affect the success probability. As can be seen from extract_probs,
			# both values for that qubit are accepted.
			if self._qubit_hack:
				M = factor.to_matrix()[:2**(self._n),:2**(self._n)]
				factor = MatrixOp(M)

			u = self._conjugateBlocks(factor).to_circuit().to_instruction(label='U^C('+str(ii)+')')

			# intermediate transpilation
			#u = transpile(u, basis_gates=['cx', 'id', 'rx', 'ry', 'rz', 'sx', 'x', 'y', 'z'], optimization_level=3).to_instruction(label='U^C('+str(ii)+')')

			# RUS for real part extraction
			self.h(num_main_qubits + ii)
			self.append(u, [j for j in range(num_main_qubits)]+[num_main_qubits + ii])
			self.h(num_main_qubits + ii)
			if self._with_measurements:
				self.measure(num_main_qubits + ii, num_main_qubits + ii if self._qubit_hack else num_main_qubits - 1 + ii) # real part extraction successful when measure 0
			if self._with_barriers:
				self.barrier()
		if self._with_measurements:
			self.measure(range(self._n),range(self._n))
			
# end QCMRF

def fidelity(P,Q):
	"""Returns fidelity between probability mass functions, given by P and Q."""
	F = 0
	for i in range(len(P)):
		F += np.sqrt(P[i] * Q[i])
	return F**2

def KL(P,Q):
	"""Returns Kullback-Leibler divergence between probability mass functions, given by P and Q."""
	kl = 0
	for i in range(len(P)):
		if Q[i] > 0 and P[i] > 0:
			kl += P[i] * np.log(P[i] / Q[i])
	return kl

def extract_probs(R,n,a):
	Y = list(itertools.product([0, 1], repeat=n))
	P = np.zeros(2**n)
	for i,y in enumerate(Y):
		s = ''
		for b in y:
			s += str(b)

		s0 = '0'*a + s

		if s0 in R:
			P[i] += R[s0]		
	z = np.sum(P)
	
	return P/z, z


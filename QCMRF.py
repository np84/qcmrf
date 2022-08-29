"""A QCMRF implementation."""

import numpy as np
import itertools

from qiskit.opflow import I, Z
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_gate
from qiskit.circuit.library import AND

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
		with_measurements=True,
		with_barriers=False
	):
		"""
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

		super().__init__(self._n + self._num_cliques + 1, self._n + self._num_cliques + 1, name=name)
		
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
		"""Returns the parameters of the circuit, computed from theta and beta.

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
		minus = [v for i,v in enumerate(C) if     y[i]] # 1

		for i in range(self._n):
			f = I
			if i in minus:
				f = (I-Z)/2 # |1><1|
			elif i in plus:
				f = (I+Z)/2 # |0><0|

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
				H += self.sufficient_statistic(C,y) * (-self.theta[i])
				i += 1
		return H

	def _conjugateBlocks(self,A):
		"""Returns a block unitary with A and ~A on its diagonal."""
		return (((I+Z)/2)^A) + (((I-Z)/2)^(~A))

	def _build(self):
		"""Return the actual QCMRF."""
		
		num_main_qubits = self._n + 1

		for i in range(self._n):
			self.h(i)
		if self._with_barriers:
			self.barrier()

		# init parameters uniformly if none are provided
		if self._theta is None and self._gamma is None:
			self._theta = []
			for i in range(self._dim):
				self._theta.append(np.random.uniform(low=-5.0,high=0))

		i = 0
		for ii,C in enumerate(self._cliques):
			UCC = QuantumCircuit(num_main_qubits, name='U_C'+str(ii))
			var = [(self._n-1)-v for v in C] + [self._n]

			# Construct U^C(gamma(theta_C))
			for y in list(itertools.product([0, 1], repeat=len(C))):
				flags = (np.array(y)*2-1).tolist()

				UCC.append(AND(len(C),flags),var)
				UCC.p(2*self.gamma[i],self._n)
				UCC.append(AND(len(C),flags),var) # UCC.reset(var)
				i = i + 1

			UCC = transpile(UCC, basis_gates=['cx', 'id', 'rz', 'sx', 'x'], optimization_level=0)

			UC    = circuit_to_gate(UCC)
			CUC   = UC.control(1)
			CUCdg = UC.inverse().control(1)

			# RUS for real part extraction
			self.h(num_main_qubits + ii)
			self.append(CUC, [num_main_qubits + ii] + list(range(num_main_qubits)))
			self.x([num_main_qubits + ii])
			self.append(CUCdg, [num_main_qubits + ii] + list(range(num_main_qubits)))
			self.x([num_main_qubits + ii])
			self.h(num_main_qubits + ii)
			if self._with_measurements:
				self.measure(num_main_qubits + ii, num_main_qubits + ii) # real part extraction successful when measure 0
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


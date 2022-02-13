#
# (C) Copyright Nico Piatkowski and Christa Zoufal 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A self-contained QCMRF."""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.linalg import expm
import itertools

import time
import copy

from qiskit.opflow import I, X, Z, MatrixOp
from qiskit.compiler import transpile
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend
from qiskit.circuit import QuantumRegister
from qiskit.transpiler import Layout
from qiskit.circuit import ParameterVector

from qiskit.compiler import assemble
from qiskit.assembler.run_config import RunConfig
from qiskit.utils.mitigation import CompleteMeasFitter, TensoredMeasFitter

from qiskit.utils.run_circuits import run_qobj, run_circuits
from qiskit.utils.measurement_error_mitigation import (
	get_measured_qubits_from_qobj,
	get_measured_qubits,
	build_measurement_error_mitigation_circuits,
	build_measurement_error_mitigation_qobj,
)

from qiskit.providers.aer.backends.qasm_simulator import QasmSimulator

class Publisher:
	"""Class used to publish interim results."""

	def __init__(self, messenger):
		self._messenger = messenger

	def callback(self, *args, **kwargs):
		text = list(args)
		for k, v in kwargs.items():
			text.append({k: v})
		self._messenger.publish(text)


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
		qubit_hack=True
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
				"The MRF parameter vector has an incorrect dimension. Expected: " + str(self._dim)
			)

		if self._gamma is not None and len(self._gamma) != self._dim:
			raise ValueError(
				"The QCMRF parameter vector has an incorrect dimension. Expected: " + str(self._dim)
			)

		self._mrfqubits = self._n + self._num_cliques + 1
		if self._qubit_hack:
			self._mrfqubits -= 1

		super().__init__(self._mrfqubits, self._mrfqubits if self._qubit_hack else self._mrfqubits - 1, name=name)
		
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
		"""Computes the Hamiltonian of the MRF

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

			u = self._conjugateBlocks(factor).to_circuit()

			# intermediate transpilation
			u = transpile(u, basis_gates=['cx', 'id', 'rx', 'ry', 'rz', 'sx', 'x', 'y', 'z'], optimization_level=3).to_instruction(label='U^C('+str(ii)+')')

			# RUS for real part extraction
			self.h(num_main_qubits + ii)
			self.append(u, [j for j in range(num_main_qubits)]+[num_main_qubits + ii])
			self.h(num_main_qubits + ii)
			self.measure(num_main_qubits + ii, num_main_qubits + ii if self._qubit_hack else num_main_qubits - 1 + ii) # real part extraction successful when measure 0

			self.barrier()
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

def run(backend,graphs,thetas,gammas,betas,repetitions,shots,layout=None,callback=None,measurement_error_mitigation=0,optimization_level=3):
	np.random.seed(1984)
	for i in range(len(graphs)):
		for j in range(repetitions):

			beta  =  betas[i] if ( betas is not None) else 1
			theta = thetas[i] if (thetas is not None) else None
			gamma = gammas[i] if (gammas is not None) else None

			s2 = time.time()
			C = QCMRF(graphs[i], theta=theta, gamma=gamma, beta=beta)

			if layout is not None:
				nvars = C.num_vertices+C.num_cliques
				
				if len(layout) < nvars:
					raise ValueError(
						"Layout has not enough qubits. Expected: " + str(C.num_vertices+C.num_cliques) + " (at least)"
					)
				LAY = layout[:nvars]
			else:
				LAY = None
			s2 = time.time() - s2

			s1 = time.time()
			T = transpile(C, backend, optimization_level=optimization_level, seed_transpiler=42, initial_layout=LAY)
			s1 = time.time() - s1		
			
			s3 = time.time()
			if not measurement_error_mitigation:
				job = backend.run(T, shots=shots)
				result = job.result()
				
			else:
				cls = None
				if measurement_error_mitigation == 1:
					cls = CompleteMeasFitter

				elif measurement_error_mitigation == 2:
					cls = TensoredMeasFitter
					
				result = run_mitigated(T, shots=shots, backend=backend, error_mitigation_cls=cls, optimization_level=optimization_level, seed_transpiler=42, initial_layout=LAY)

			P, c = extract_probs(result.get_counts(), C.num_vertices, C.num_cliques)
			s3 = time.time() - s3
			
			H = C.Hamiltonian()
			RHO = expm(-beta*H.to_matrix())
			z = np.trace(RHO)
			Q = np.diag(RHO)/z
			
			callback(C.num_vertices, C.dimension, C.num_cliques, C.max_clique, np.real(fidelity(P,Q)), np.real(KL(Q,P)), c/shots, len(T), T.depth(), shots, s1, s2, s3, C.theta, result.get_counts())


def train(backend,graph,mu,data,shots,iterations,layout=None,callback=None,measurement_error_mitigation=0,optimization_level=3):
	np.random.seed(1984)

	dim = len(mu)
	theta = np.zeros(dim)

	for ii in range(iterations):

		CC = QCMRF(graph, theta=theta)

		if layout is not None:
			nvars = CC.num_vertices+CC.num_cliques			
			if len(layout) < nvars:
				raise ValueError(
					"Layout has not enough qubits. Expected: " + str(CC.num_vertices+CC.num_cliques) + " (at least)"
				)
				LAY = layout[:nvars]
		else:
			LAY = None

		T = transpile(CC, backend, optimization_level=optimization_level, seed_transpiler=42, initial_layout=LAY)

		if not measurement_error_mitigation:
			job = backend.run(T, shots=shots)
			result = job.result()
				
		else:
			cls = None
			if measurement_error_mitigation == 1:
				cls = CompleteMeasFitter
			elif measurement_error_mitigation == 2:
				cls = TensoredMeasFitter
					
			result = run_mitigated(T, shots=shots, backend=backend, error_mitigation_cls=cls, optimization_level=optimization_level, seed_transpiler=42, initial_layout=LAY)

		P, c = extract_probs(result.get_counts(), CC.num_vertices, CC.num_cliques)

		L = 0
		for x in data:
			L -= np.log(P[int(x, 2)])
		L *= 1/len(data)
		
		mu_hat = np.zeros(dim)
		Y = list(itertools.product([0, 1], repeat=CC.num_vertices))
		for i,x in enumerate(Y):
			temp = np.zeros(dim)
			off = 0
			for C in graph:
				s = ''
				for v in C:
					s += str(x[v])
				yn = int(s, 2)
				temp[off+yn] = 1
				off += 2**len(C)
			mu_hat += P[i] * temp
			
		grad = np.array(mu) - mu_hat
		
		theta += 0.1 * grad

		theta -= np.max(theta)

		callback(CC.num_vertices, CC.dimension, CC.num_cliques, CC.max_clique, c/shots, len(T), T.depth(), shots, ii, L)

def main(backend, user_messenger, **kwargs):
	"""Entry function."""
	# parse inputs
	mandatory = {"graphs"}
	missing = mandatory - set(kwargs.keys())
	if len(missing) > 0:
		raise ValueError(f"The following mandatory arguments are missing: {missing}.")

	train_mode = kwargs.get("train",False)

	# Extract the input form the kwargs and build serializable kwargs for book keeping.
	serialized_inputs = {}
	serialized_inputs["graphs"] = kwargs["graphs"]
	serialized_inputs["thetas"] = kwargs.get("thetas",None)
	serialized_inputs["gammas"] = kwargs.get("gammas",None)
	serialized_inputs["betas"] = kwargs.get("betas",None)
	serialized_inputs["mu"] = kwargs.get("mu",None)
	serialized_inputs["data"] = kwargs.get("data",[])
	serialized_inputs["repetitions"] = kwargs.get("repetitions", 1)
	serialized_inputs["iterations"] = kwargs.get("iterations", 100)
	serialized_inputs["shots"] = kwargs.get("shots", 8192)
	serialized_inputs["measurement_error_mitigation"] = kwargs.get("measurement_error_mitigation", 0)
	serialized_inputs["optimization_level"] = kwargs.get("optimization_level", 3)
	serialized_inputs["layout"] = kwargs.get("layout", None)

	publisher = Publisher(user_messenger)
	
	history = {"n": [], "d": [], "num_cliques": [], "max_clique": [], "fidelity": [], "KL": [], "success_rate": [], "gates": [], "depth": [], "shots": [], "transpile_time": [], "prepare_time": [], "exec_time": [], "theta": [], "counts": []}

	history_train = {"n": [], "d": [], "num_cliques": [], "max_clique": [], "success_rate": [], "gates": [], "depth": [], "shots": [], "iteration": [], "loss": []}

	def store_history_and_forward(n, d, num_cliques, max_clique, fidelity, KL, success_rate, gates, depth, shots, transpile_time, prepare_time, exec_time, theta, counts):
		# store information
		history["n"].append(n)
		history["d"].append(d)
		history["num_cliques"].append(num_cliques)
		history["max_clique"].append(max_clique)
		history["fidelity"].append(fidelity)
		history["KL"].append(KL)
		history["success_rate"].append(success_rate)
		history["gates"].append(gates)
		history["depth"].append(depth)
		history["shots"].append(shots)
		history["transpile_time"].append(transpile_time)
		history["prepare_time"].append(prepare_time)
		history["exec_time"].append(exec_time)
		history["theta"].append(theta)
		history["counts"].append(counts)
		# and forward information to users callback
		publisher.callback(n, d, num_cliques, max_clique, fidelity, KL, success_rate, gates, depth, shots, transpile_time, prepare_time, exec_time, theta)

	def store_history_and_forward_train(n, d, num_cliques, max_clique, success_rate, gates, depth, shots, iteration, loss):
		# store information
		history_train["n"].append(n)
		history_train["d"].append(d)
		history_train["num_cliques"].append(num_cliques)
		history_train["max_clique"].append(max_clique)
		history_train["success_rate"].append(success_rate)
		history_train["gates"].append(gates)
		history_train["depth"].append(depth)
		history_train["shots"].append(shots)
		history_train["iteration"].append(iteration)
		history_train["loss"].append(loss)
		# and forward information to users callback
		publisher.callback(n, d, num_cliques, max_clique, success_rate, gates, depth, shots, iteration, loss)


	if not train_mode:
		result = run(
			backend,
			serialized_inputs["graphs"],
			serialized_inputs["thetas"],
			serialized_inputs["gammas"],
			serialized_inputs["betas"],
			serialized_inputs["repetitions"],
			serialized_inputs["shots"],
			serialized_inputs["layout"],
			store_history_and_forward,
			serialized_inputs["measurement_error_mitigation"],
			serialized_inputs["optimization_level"]
		)

		serialized_result = {
			"Fidelity_mean": np.mean(history['fidelity']),
			"Fidelity_sdev": np.std(history['fidelity']),
			"KL_mean": np.mean(history['KL']),
			"KL_sdev": np.std(history['KL']),
			"SR_mean": np.mean(history['success_rate']),
			"SR_sdev": np.std(history['success_rate']),
			"Depth_mean": np.mean(history['depth']),
			"Depth_sdev": np.std(history['depth']),
			"all_results": history,
			"inputs": serialized_inputs
		}
	else:
		result = train(
			backend,
			serialized_inputs["graphs"][0],
			serialized_inputs["mu"],
			serialized_inputs["data"],
			serialized_inputs["shots"],
			serialized_inputs["iterations"],
			serialized_inputs["layout"],
			store_history_and_forward_train,
			serialized_inputs["measurement_error_mitigation"],
			serialized_inputs["optimization_level"]
		)

		serialized_result = {
			"all_results": history_train,
			"inputs": serialized_inputs
		}

	return serialized_result

def run_mitigated(circuits,shots,backend,error_mitigation_cls,optimization_level,seed_transpiler,initial_layout):
	if isinstance(circuits, list):
		circuits = circuits.copy()
	else:
		circuits = [circuits]

	compile_config = {
		"initial_layout": initial_layout,
		"seed_transpiler": seed_transpiler,
		"optimization_level": optimization_level,
	}

	if type(backend) == QasmSimulator:
		qjob_config = (
			{"timeout": None}
		)
	else:
		qjob_config = (
			{"timeout": None, "wait": 5.0}
		)

	skip_qobj_validation = True
	max_job_retries = int(1e18)
	noise_config = {}
	callback = None

	COMPLETE_MEAS_FITTER = 0
	TENSORED_MEAS_FITTER = 1

	run_config = RunConfig(shots=shots)
	qobj = assemble(circuits,shots=shots)
	time_taken = 0
	meas_type = COMPLETE_MEAS_FITTER
	if error_mitigation_cls == TensoredMeasFitter:
		meas_type = TENSORED_MEAS_FITTER

	qubit_index, qubit_mappings = (
		get_measured_qubits_from_qobj(qobj)
	)

	mit_pattern = [[i] for i in range(len(qubit_index))]

	qubit_index_str = "_".join([str(x) for x in qubit_index]) + "_{}".format(shots)
	meas_error_mitigation_fitter = None

	build_cals_matrix = meas_error_mitigation_fitter is None

	cal_circuits = None
	prepended_calibration_circuits: int = 0
	if build_cals_matrix:
		temp_run_config = copy.deepcopy(run_config)
		
		(
			cals_qobj,
			state_labels,
			circuit_labels,
		) = build_measurement_error_mitigation_qobj(
			initial_layout, #qubit_index,
			error_mitigation_cls,
			backend,
			{},
			compile_config,
			temp_run_config,
			mit_pattern=mit_pattern,
		)

		# insert the calibration circuit into main qobj if the shots are the same
		qobj.experiments[0:0] = cals_qobj.experiments
		result = run_qobj(
			qobj,
			backend,
			qjob_config,
			None,
			noise_config,
			skip_qobj_validation,
			callback,
			max_job_retries,
		)
		time_taken += result.time_taken
		cals_result = result

		if meas_type == COMPLETE_MEAS_FITTER:
			meas_error_mitigation_fitter = error_mitigation_cls(
			cals_result, state_labels, qubit_list=qubit_index, circlabel=circuit_labels
			)
		elif meas_type == TENSORED_MEAS_FITTER:
			meas_error_mitigation_fitter = error_mitigation_cls(
			cals_result, mit_pattern=state_labels, circlabel=circuit_labels
			)
	
	if meas_error_mitigation_fitter is not None:
		if (
			hasattr(run_config, "parameterizations")
			and len(run_config.parameterizations) > 0
			and len(run_config.parameterizations[0]) > 0
			and len(run_config.parameterizations[0][0]) > 0
		):
			num_circuit_templates = len(run_config.parameterizations)
			num_param_variations = len(run_config.parameterizations[0][0])
			num_circuits = num_circuit_templates * num_param_variations
		else:
			input_circuits = circuits[prepended_calibration_circuits:]
			num_circuits = len(input_circuits)
		skip_num_circuits = len(result.results) - num_circuits
		#  remove the calibration counts from result object to assure the length of
		#  ExperimentalResult is equal length to input circuits
		result.results = result.results[skip_num_circuits:]
		tmp_result = copy.deepcopy(result)
		for qubit_index_str, c_idx in qubit_mappings.items():
			curr_qubit_index = [int(x) for x in qubit_index_str.split("_")]
			tmp_result.results = [result.results[i] for i in c_idx]
			
			if curr_qubit_index == qubit_index:
				tmp_fitter = meas_error_mitigation_fitter
			elif meas_type == COMPLETE_MEAS_FITTER:
				tmp_fitter = meas_error_mitigation_fitter.subset_fitter(curr_qubit_index)
			else:
				tmp_fitter = meas_error_mitigation_fitter
	
			tmp_result = tmp_fitter.filter.apply(
				tmp_result, "least_squares"
			)
			for i, n in enumerate(c_idx):
				# convert counts to integer and remove 0 values
				tmp_result.results[i].data.counts = {
					k: round(v)
					for k, v in tmp_result.results[i].data.counts.items()
					if round(v) != 0
				}
				result.results[n] = tmp_result.results[i]
	return result

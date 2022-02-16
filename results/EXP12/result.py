from pathlib import Path
import numpy as np
import os.path
import json
import itertools
from qiskit import IBMQ
from qiskit.opflow import I, X, Z
from scipy.linalg import expm

#options = {
#	'backend_name': 'ibmq_qasm_simulator'
#}
#
#with open('account.json', 'r') as accountfile:
#	account = json.load(accountfile)
#
#IBMQ.enable_account(account['token'], account['url'])
#
#account['project'] = 'member'
#
#provider = IBMQ.get_provider(
#	hub=account['hub'],
#	group=account['group'],
#	project=account['project']
#)
#
#with open('program_id.json', 'r') as idfile:
#	res = json.load(idfile)
#
#

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


def sufficient_statistic(C,y,n):
	result = 1

	plus  = [v for i,v in enumerate(C) if not y[i]] # 0
	minus = [v for i,v in enumerate(C) if     y[i]] # 1)

	for i in range(n):
		f = I
		if i in minus:
			f = (I-Z)/2
		elif i in plus:
			f = (I+Z)/2

		result = result^f

	return result
		
def Hamiltonian(CL,theta,n):
	H = 0
	i = 0
	for C in CL:
		for y in list(itertools.product([0, 1], repeat=len(C))):
			H += -theta[i] * sufficient_statistic(C,y,n)
			i += 1
	return H

results = {}

def graphListToTuple(G):
	temp = []
	for C in G:
		temp.append(tuple(C))
	return tuple(temp)
	
def graphTupleToList(G):
	temp = []
	for C in G:
		temp.append(list(C))
	return temp

class Result:
	def __init__(self,backend,result,idx):
		reps = result['inputs']['repetitions']
		self.backend = backend
		self.mit = result['inputs']['measurement_error_mitigation']
		self.n = result['all_results']['n'][idx*reps]
		self.num_cliques = result['all_results']['num_cliques'][idx*reps]
		self.dim = result['all_results']['d'][idx*reps]
		self.shots = result['all_results']['shots'][idx*reps]
		self.F = result['all_results']['fidelity'][idx*reps:(idx+1)*reps]  # Fidelity
		self.KL = result['all_results']['KL'][idx*reps:(idx+1)*reps] # Kullback-Leibler div.
		self.SR = result['all_results']['success_rate'][idx*reps:(idx+1)*reps] # Success rate
		self.G = result['all_results']['gates'][idx*reps:(idx+1)*reps]  # Numbers of gates
		self.D = result['all_results']['depth'][idx*reps:(idx+1)*reps]  # Depths
		self.W = result['all_results']['theta'][idx*reps:(idx+1)*reps]  # Weights
		self.C = result['all_results']['counts'][idx*reps:(idx+1)*reps]  # Counts
		
pathlist = Path('.').glob('**/*info*.json')
for path in pathlist:
	# because path is object not string
	info_file_name = str(path)
	result_json_name = info_file_name.replace('info','result')
	result_txt_name = result_json_name.replace('json','txt')
	has_info = os.path.isfile(info_file_name)
	has_json = os.path.isfile(result_json_name)
	hsa_txt  = os.path.isfile(result_txt_name)
	
	if has_info and has_json:
		info_f = open(info_file_name,'r')
		info = json.load(info_f)
		backend = info['backend']
		info_f.close()
		
		for G in info['params']['graphs']:
			g = graphListToTuple(G)
			if results.get(g) is None:
				results[g] = []

		result_f = open(result_json_name,'r')
		result = json.load(result_f)
		result_f.close()
		
		for i,G in enumerate(result['inputs']['graphs']):
			g = graphListToTuple(G)
			r = Result(info['backend'],result,i)
			results[g].append(r)

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

for G in results.keys():
	COUNTS = None
	FIDS = {}
	KLS = {}
	SRS = {}
	for i in range(10):
		FIDS[i] = []
		KLS[i] = []
		SRS[i] = []
	g = graphTupleToList(G)
	print(g)
	first = True
	n = 0
	cl = 0
	w = None
	S = 0
	for jj,r in enumerate(results[G]):
		if r.backend != 'ibmq_qasm_simulator':
			S += r.shots
			# aggregate counts
			if first:
				COUNTS = r.C
				n = r.n
				cl = r.num_cliques
				w = r.W
				#print(n,cl,w)
			else:
				for j,CC in enumerate(r.C): 
					for c in CC.keys():
						if COUNTS[j].get(c) is None:
							COUNTS[j][c] = CC[c]
						else:
							COUNTS[j][c] += CC[c]
			for j,f in enumerate(r.F):
				FIDS[j].append(f) 
			for j,k in enumerate(r.KL):
				KLS[j].append(k) 
			for j,s in enumerate(r.SR):
				SRS[j].append(s) 
			first = False
			#for j,CC in enumerate(r.C): 
			#	P,_ = extract_probs(CC,n,cl)
			#	print('P'+str(j),P)
			#runtime_inputs = {
			#	"graphs": [graphTupleToList(G)]*10,
			#	"repetitions": 1,
			#	"shots": 100000,
			#	"thetas": r.W,
			#	"measurement_error_mitigation": 0,
			#	"optimization_level": 3
			#}
			#job = provider.runtime.run(
			#	program_id=res['program_id'],
			#	options=options,
			#	inputs=runtime_inputs
			#)
			#print(job.job_id(),job.status())
			#break
	FF = []
	KK = []
	SS = []
	for i,c in enumerate(COUNTS):
		P,zz = extract_probs(c,n,cl)
		
		H = Hamiltonian(g,w[i],n)
		RHO = expm(-H.to_matrix())
		z = np.trace(RHO)
		Q = np.diag(RHO)/z
		
		SS.append(zz/S)
		FF.append(fidelity(P,Q))
		KK.append(KL(P,Q))
	#print(np.max(FF),np.min(KK),np.mean(SS))
	FF = []
	KK = []
	SS = []
	for i in range(10):
		FF.append(np.max(FIDS[i])) # best over all backends for each run
		KK.append(np.min(KLS[i])) # best over all backends for each run
		SS.append(np.max(SRS[i])) # best over all backends for each run
	print(np.max(FF),np.median(FF),np.min(FF))
	print(np.max(KK),np.median(KK),np.min(KK))
	print(np.max(SS),np.median(SS),np.min(SS))

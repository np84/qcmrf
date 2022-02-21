from pathlib import Path
import numpy as np
import os.path
import json
import itertools

N = 100000

def extract_probs(R,n,a=0):
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
	
def int2np(x,n):
	b = np.binary_repr(x, width=n)
	r = np.zeros(n)
	for i,c in enumerate(b):
		if c == '1':
			r[i] = 1
	return r

def np2str(x):
	s = ''
	for v in x:
		s += str(int(v))
	return s
	
def nphot2str(x,n):
	a = np.argmax(x)
	return int2str(a,n)

def int2str(a,n):
	y = int2np(a,n)
	return np2str(y)

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

import argparse

parser = argparse.ArgumentParser(description='Evaluate QCMRF results.')
parser.add_argument('method',
                    default='qcmrf',
                    const='qcmrf',
                    nargs='?',
                    choices=['qcmrf', 'qcmrfsim', 'gibbs', 'pam'],
                    help='The sampler. (default: %(default)s)')
args = parser.parse_args()

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

results = {}

################################################################################

# load json files into memory

pathlist = Path('.').glob('./*info*.json')
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

IDX = {
((0, 1, 2, 3),):8,
((0, 1), (1, 2), (2, 3), (0, 3)):5,
((0, 1), (1, 2), (2, 3), (3, 4)):4,
((0, 1, 2),):6,
((0, 1, 2), (2, 3, 4)):7,
((0, 1), (1, 2), (2, 3)):3,
((0,),):0,
((0, 1), (1, 2)):2,
((0, 1),):1,
((0, 1, 2), (2, 3, 4), (4, 5, 6)):9
}

del results[((0, 1, 2), (2, 3, 4), (4, 5, 6))] # only one simulation run this graph
del IDX[((0, 1, 2), (2, 3, 4), (4, 5, 6))]

RES = np.array([[0.0]*len(IDX)]* 3).tolist() # reserve mem for results
			
SUM_CHECK = {} # store weight sum as a "hash" to ensure that models use same weights
W = {} # store weights

################################################################################

# extract 10 weight vectors for each graph from the json dicts
# this is used later when we compute the corresponding results
# with Gibbs-sampling and PAM

MRFs = {'graphs':[],'thetas':[]}

for idx,G in enumerate(results.keys()):

	g = graphTupleToList(G)
	
	MRFs['graphs'].append(g)
		
	WS = {}
	l = 10
	for i in range(l):
		WS[i] = []
		
	TH = [None] * 10
			
	for jj,r in enumerate(results[G]):
		if (r.backend == 'ibmq_montreal'): # any backend is ok. 
			for j,w in enumerate(r.W):
				WS[j].append(w)
				if SUM_CHECK.get(idx) is None:
					SUM_CHECK[idx] = {}
					W[idx] = {}
				if SUM_CHECK[idx].get(j) is None:
					SUM_CHECK[idx][j] = np.sum(w)
					W[idx][j] = w
					TH[j] = w
					
	MRFs['thetas'].append(TH)

with open('models.json', 'w', encoding='utf-8') as f:
    json.dump(MRFs, f, ensure_ascii=False, indent=4)

################################################################################

# extract qcmrf results from json dicts

if True:

	for idx,G in enumerate(results.keys()):
	
		g = graphTupleToList(G)

		FIDS = {}
		KLS  = {}
		SRS  = {}
		
		l = 10
		
		for i in range(l):
			FIDS[i] = []
			KLS[i]  = []
			SRS[i]  = []

		for jj,r in enumerate(results[G]):
			if (r.backend == 'ibmq_qasm_simulator' and args.method != 'qcmrf') or (r.backend != 'ibmq_qasm_simulator' and args.method == 'qcmrf'):
				for j,f in enumerate(r.F):
					if SUM_CHECK[idx][j] == np.sum(r.W[j]):
						FIDS[j].append(f)
				for j,k in enumerate(r.KL):
					if SUM_CHECK[idx][j] == np.sum(r.W[j]):
						KLS[j].append(k) 
				for j,s in enumerate(r.SR):
					if SUM_CHECK[idx][j] == np.sum(r.W[j]):
						SRS[j].append(s)

		if len(FIDS[0]) == 0:
			break

		FF = []
		KK = []
		SS = []
		
		if args.method == 'qcmrf':
			for i in range(l):
				if len(FIDS[i]) > 0:
					jdx = np.argmax(FIDS[i])
					FF.append(FIDS[i][jdx]) # best over all backends for each run
					KK.append(KLS[i][jdx]) # best over all backends for each run
					SS.append(SRS[i][jdx]) # best over all backends for each run

			v1 = (np.median(FF))
			v2 = (np.median(KK))
			v3 = int((np.median(SS))*N)
			
		else:
			for i in range(l):
				if len(FIDS[i]) > 0:
					FF += FIDS[i]
					KK += KLS[i]
					SS += SRS[i]

			v1 = (np.median(FF))
			v2 = (np.median(KK))
			v3 = int((np.median(SS))*N)
			
		RES[0][IDX[G]] = v1
		RES[1][IDX[G]] = v2
		RES[2][IDX[G]] = v3
		#print(RES[0][IDX[G]],RES[1][IDX[G]],RES[2][IDX[G]])

#print(RES)

################################################################################

# the following method computes unnomalized log-probs for all 2**n states
def compute_all_scores(g,w,n):
	SCORES = np.zeros(2**n)
				
	for x in range(2**n):
		X = int2np(x,n)
					
		score = 0
		offset = 0
		for C in g:
			s = ''
			for v in C:
				if X[v] < 1:
					s += '1'
				else:
					s += '0'
				y = int(s,2)
			score += w[offset+y]
			offset += 2**len(C)
		SCORES[x] = score
	
	return SCORES
	
################################################################################

if args.method == 'gibbs' or args.method == 'pam':
	RES[0] = [0]*len(IDX) # cleanup results
	RES[1] = [0]*len(IDX) # cleanup results
	for idx,G in enumerate(results.keys()):

		N = 100_000 # Number of raw samples

		g = graphTupleToList(G)
		print('G =',g,'N =',N)
		FF = []
		KK = []
		
		for jdx in W[idx]:
		
			n = np.max(g)+1 # number of variables
			w = W[idx][jdx] # model parameters
				
			SCORES = compute_all_scores(g,w,n) # unnomalized log-probs
			Q = np.exp(SCORES)
			Q /= np.sum(Q) # groundtruth probs
			counts = {}
			
################################################################################
				
			if args.method == 'pam':
			
				for x in range(N):
				
					local = np.copy(w)
				
					S = 3 # hyper parameter
					b = 1

					for ii in range(len(w)):
						noise = 0;
						for s in range(S):
							scale = len(G) / (1.0+s);
							noise += scale * np.random.gamma(1.0/len(G),1.0) - np.log(S*1.0)

						noise *= (b/len(G));
						local[ii] += noise;
						
					P_SCORES = compute_all_scores(g,local,n)

					x_star = np.argmax(P_SCORES)
					s = int2str(x_star,n)
					if counts.get(s) is None:
						counts[s] = 1
					else:
						counts[s] += 1
						
################################################################################

			elif args.method == 'gibbs':
				for x in range(int(N/(100*n))):
					X = np.random.binomial(n=1,p=0.5,size=n) # restart chain, random init
					for j in range(100): # effetively drop 100*n samples in the loop
						for v in range(n): # resample each variable n times (counts as bunred gibbs sample)
							xs = np2str(X)
							p = Q[int(xs,2)] # P(V) = P(v, V\{v})
							Xtemp = np.copy(X)							
							if xs[v] == '0':
								Xtemp[v] = 1
							else:
								Xtemp[v] = 0
							q = p + Q[int(np2str(Xtemp),2)] # P(V\{v})
							p = p/q # P(v | V\{v}) = P(v | N(v)) due to Markov prop.
							if X[v] < 1: # v=0
								p = 1-p # p = P(v=1 | V\{v})
							v_new = np.random.binomial(n=1,p=p,size=1)
							X[v] = v_new[0]
					s = np2str(X)
					if counts.get(s) is None:
						counts[s] = 1
					else:
						counts[s] += 1
						
################################################################################

			P,_ = extract_probs(counts,n)
				
			f = np.real(fidelity(P,Q))
			k = np.real(KL(Q,P))
					
			FF.append(f)
			KK.append(k)
				
		if len(FF)>0:
			RES[0][IDX[G]] = (np.median(FF))
			RES[1][IDX[G]] = (np.median(KK))

for i,row in enumerate(RES):
	if i == 0:
		prefix = '&$F$&'
	if i == 1:
		prefix = '&KL&'
	if i == 2:
		prefix = '&$\\tilde{\delta}^*N$&'
		if not args.method.startswith('qcmrf'):
			break
	print(prefix,end='',flush=True)
	for val in row:
		if i < 2:
			if val < 0:
				print('---&',end='',flush=True)
			else:
				print(("%.3f" % val)+'&',end='',flush=True)
		else:
			print(str(val)+'&',end='',flush=True)
	print('\\\\')

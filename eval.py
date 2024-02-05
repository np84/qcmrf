from QCMRF import QCMRF, extract_probs, fidelity as F, KL
import numpy as np
np.random.seed(1984)
import json
from scipy.stats import halfnorm, norm
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options, Sampler
from login_ibm import service
from qiskit import Aer
import errno
import os
import argparse
from prettytable import PrettyTable

import kiopto_native as px

parser = argparse.ArgumentParser(prog='QCMRF result evaluation.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--results', type=str, default='result_ehningen.json', help='Result file as downloaded from backend.')
parser.add_argument('--scale', type=str, default='0.1', help='Variance of parameter prior.')
parser.add_argument('--mode', type=str, default='file', help='file or gibbs or pam.')

args = parser.parse_args()

##############################################################################

REPS = 10

GRAPHS = [[[0]],[[0,1]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[3,4]],[[0,1,2]],[[0,1,2],[2,3,4]],[[0,1,2,3]]]
THETAS = {}

for j,C in enumerate(GRAPHS):
	n = max(np.array(C).flatten()) + 1
	b = px.backend(C,np.array([2]*n),inference='exact')
	d = len( px.weights(b) )
	
	for i in range(REPS):
		theta = -halfnorm.rvs(loc = 0, scale = float(args.scale), size=d)
		if THETAS.get(j) is None:
			THETAS[j] = []
		THETAS[j].append( theta.tolist() )

R = {'GRAPHS':GRAPHS, 'THETAS':THETAS}
f = open('./models_'+args.scale+'.json', 'w')
f.write( json.dumps(R, indent=4) )
f.close()

##############################################################################

results_fname = './res_' + args.scale + '/' + args.results

if os.path.isfile(results_fname):
	f = open(results_fname, 'r')
	results_file = json.load(f)
	f.close()
	try:
		dists = results_file['quasi_dists']
		norm = 1
	except:
		dists = results_file
		norm = 10_000 # num of shots!
else:
	raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), results_fname)

##############################################################################

idx = 0

TAB = [['graph','fidelity','max fidelity','success rate']]

for j,C in enumerate(GRAPHS):
	n = max(np.array(C).flatten()) + 1
	Y = 2*np.ones(n).astype(int)
	N = 2**n

	best_idx = None
	best_F = 0
	best_delta = 0

	L_F = []
	L_delta = []

	for i in range(REPS):
		theta = THETAS[j][i]
		b = px.backend(C,Y)
		
		backend_w = px.weights(b)
		backend_w[:] = theta[:]
		lnZ = px.infer(b,task='partition')
		p = np.zeros(N)
		q = np.zeros(N)
		for xid in range(N):
			log_psi = px.logpot(b,xid)
			p[xid] = np.exp(log_psi-lnZ)

		if args.mode == 'gibbs':
			S = px.sample(b)
			burn = 10
			_S = S[::burn][1:]
			for s in _S:
				k = ''.join(str(cc) for cc in s)
				kid = int(k,2)
				q[kid] += 1
			Z = q.sum()
			norm = 10_000
			
		elif args.mode == 'pam':
			S = px.sample(b,pam=True)
			for s in S:
				k = ''.join(str(cc) for cc in s)
				kid = int(k,2)
				q[kid] += 1
			Z = q.sum()
			norm = 10_000

		else:
			Q = dists[idx]
			Z = 0
			for k in Q.keys():
				kid = int(k,2)
				if kid < N:
					q[kid] = Q[k]
					Z += Q[k]
		q /= Z

		mF = min(F(p,q),1)
		mF = max(mF,0)
		L_F += [mF]
		L_delta += [Z/norm]

		if mF > best_F:
			best_F = mF
			best_idx = i
			best_delta = Z

		idx += 1
	smF = np.array(L_F).mean() 
	zmF = np.array(L_F).std()
	
	smd = np.array(L_delta).mean()
	zmd = np.array(L_delta).std()

	ROW = [str(C),'{:.3f}'.format(smF)+' ±''{:.3f}'.format(zmF),'{:.3f}'.format(best_F),'{:.3f}'.format(smd)+' ±''{:.3f}'.format(zmd)]
	TAB.append(ROW)
tab = PrettyTable(TAB[0])
tab.add_rows(TAB[1:])
print(tab)

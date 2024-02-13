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

import matplotlib.pyplot as plt

import kiopto_native as px

scales = [0.1,0.25,0.5]

parser = argparse.ArgumentParser(prog='Whisker plot for QCMRF success rate.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--backend', type=str, default='simulation', help='The backend.')

args = parser.parse_args()

L_A = []
L_B = []
L_C = []

WH = {}

##############################################################################

for scale in scales:

	WH[scale] = []

	REPS = 10

	GRAPHS = [[[0]],[[0,1]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[3,4]],[[0,1,2]],[[0,1,2],[2,3,4]],[[0,1,2,3]]]
	THETAS = {}

	for j,C in enumerate(GRAPHS):
		n = max(np.array(C).flatten()) + 1
		b = px.backend(C,np.array([2]*n),inference='exact')
		d = len( px.weights(b) )
		
		for i in range(REPS):
			theta = -halfnorm.rvs(loc = 0, scale = scale, size=d)
			if THETAS.get(j) is None:
				THETAS[j] = []
			THETAS[j].append( theta.tolist() )

	R = {'GRAPHS':GRAPHS, 'THETAS':THETAS}
	f = open('./models_'+str(scale)+'.json', 'w')
	f.write( json.dumps(R, indent=4) )
	f.close()

	##############################################################################

	results_fname = './res_' + str(scale) + '/' + 'result_' + args.backend + '.json' 

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
			
			w_nrm = np.linalg.norm(px.weights(b), ord=2)
			
			if j == 1:
				L_A.append((w_nrm,mF))
				L_B.append((w_nrm,Z/norm))
				L_C.append((w_nrm,scale))
				
				WH[scale].append(Z/norm)

			if mF > best_F:
				best_F = mF
				best_idx = i
				best_delta = Z

			idx += 1

A = np.array(L_A)
B = np.array(L_B)
C = np.array(L_C)

plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{amsmath}
    \usepackage{amssymb}
    \usepackage{bm}
'''

plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

#plt.subplots_adjust(wspace=0.5, hspace=0.5)

axes[0].scatter(B[:,0],B[:,1])
axes[0].set_xlabel('Parameter norm $\|\\bm{\\theta}\|_2$')
axes[0].set_ylabel('Empirical success rate $\hat{\delta}$')

axes[1].boxplot([WH[k] for k in WH.keys()])
axes[1].set_xlabel('Scale $\sigma$')
#axes[1].set_ylabel('Estimated success rate $\hat{\delta}$')
axes[1].set_xticklabels([str(s) for s in scales])

plt.suptitle('\\texttt{'+args.backend+'}')

out_fname = './success_' + args.backend + '.pdf' 

plt.savefig(out_fname)

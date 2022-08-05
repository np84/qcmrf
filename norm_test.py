import qiskit
import numpy as np
import pxpy as px
import itertools
import matplotlib.pyplot as plt
import tqdm

from scipy.linalg import expm
from qiskit.execute_function import execute
from qiskit import BasicAer
from QCMRF import QCMRF, extract_probs
from qiskit.opflow import I, X, Y, Z, MatrixOp
from qiskit.quantum_info import Operator

M = 1000
d = 4
D = np.zeros(M*d).reshape(M,d)

def entropy(P,n):
	ent = 0
	for i,x in enumerate(list(itertools.product([0, 1], repeat=n))):
		p = P[i]
		ent -= p * np.log(p)
	return ent

for I in tqdm.tqdm(range(M), ascii=None):
	G = [[0,1],[1,2],[2,3],[0,3]]
	
	w = np.random.uniform(low=0, high=5.0, size=len(G)*4)
	w -= np.max(w)
	beta = 1

	C = QCMRF(cliques=G, theta=w, beta=beta, with_measurements=False, with_barriers=False)
	#print('gammma       =',C.gamma,np.linalg.norm(C.gamma))
	R = np.array(Operator(C))[:,0]
	p = np.abs(R[:2**C.num_nodes])**2 # P(X,A=0)
	a = p.sum()                       # P(A=0)
	P = p/a                           # P(X | A=0)

	#print('P_S(A=0) =',a)
	#print()

	G = px.create_graph(np.array(G))
	m = px.create_model(w,G,2)
	
	A = 0
	Q = np.zeros(len(G)*4)
	for y in list(itertools.product([0, 1], repeat=C.num_nodes)):
		A += np.exp(m.score(np.array(y)))
	A = np.log(A)
	
	Q,_ = m.infer()
	
	#X.append(ent)
	D[I,0] = -np.min(w)
	D[I,1] = a
	D[I,2] = A
	D[I,3] = entropy(P,C.num_nodes)
	#D[I,4] = entropy(Q[0:4],2)
	#D[I,5] = entropy(Q[4:8],2)
	#D[I,6] = entropy(Q[8:12],2)

#plt.scatter(X, Y)
#plt.show()
np.savetxt("./acc_stats_4.csv", D, fmt='%.5e', delimiter=",")

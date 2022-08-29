import numpy as np
from scipy.linalg import expm
from QCMRF import QCMRF
from qiskit.opflow import Zero
from qiskit.quantum_info import Operator

G = [[0,1],[1,2],[2,3]]
w = np.random.uniform(low=0, high=0.1, size=len(G)*4)
w -= np.max(w) # now, w is in [-w_max,0]^d
beta = 1

C = QCMRF(cliques=G, theta=w, beta=beta, with_measurements=False, with_barriers=False, qubit_hack=True)

print('NUM QUBITS',C.num_qubits,sep='\t')
print('NUM NODES',C.num_nodes,sep='\t')

U = np.array(Operator(C))
z = (Zero^C.num_qubits).to_matrix()

SUCCESS_PROB = ( np.abs(( U @ z )[:2**C.num_vertices] )**2 ).sum()
print('SUCCESS_PROB',SUCCESS_PROB,sep='\t\t')

EXPH = expm(-C.Hamiltonian().to_matrix())
ZM = np.trace(EXPH)
print('PARTITION',ZM,sep='\t\t')
print('PARTITION/SUCCESS_PROB',ZM/SUCCESS_PROB,sep='\t')

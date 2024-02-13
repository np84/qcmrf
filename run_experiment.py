from QCMRF import QCMRF, extract_probs, fidelity as F, KL
import numpy as np
np.random.seed(1984)

import json

import kiopto_native as px

from scipy.stats import halfnorm, norm

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Options, Sampler
from login_ibm import service
from qiskit import Aer

SHOTS = 10000
REPS = 10
SCALE = 0.5

GRAPHS = [[[0]],[[0,1]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[3,4]],[[0,1,2]],[[0,1,2],[2,3,4]],[[0,1,2,3]]]
THETAS = {}

for j,C in enumerate(GRAPHS):
	n = max(np.array(C).flatten()) + 1
	N = 2**n
	b = px.backend(C,np.array([2]*n))
	d = len( px.weights(b) )
	
	for i in range(REPS):
		theta = -halfnorm.rvs(loc = 0, scale = SCALE, size=d)
		if THETAS.get(j) is None:
			THETAS[j] = []
		THETAS[j].append( theta.tolist() )

R = {'GRAPHS':GRAPHS, 'THETAS':THETAS}
f = open('./models_'+str(SCALE)+'.json', 'w')
f.write( json.dumps(R, indent=4) )
f.close()

##############################################################################

CIRCS = []

for j,C in enumerate(GRAPHS):
	for i in range(REPS):
		theta = THETAS[j][i]
		qcmrf = QCMRF(C, theta, with_measurements=True)
		CIRCS.append(qcmrf)

##############################################################################

T = transpile(CIRCS, basis_gates=['cx', 'id', 'rz', 'sx', 'x'])

simulator = Aer.get_backend('qasm_simulator')

result = simulator.run(T, shots=SHOTS).result()
counts = result.get_counts()

f = open('./result_simulation_'+str(SCALE)+'.json', 'w')
f.write( json.dumps(counts, indent=4) )
f.close()

exit()

options = Options()
options.resilience_level = 1
options.optimization_level = 3
options.execution.shots = SHOTS

service = service()
b = service.backend('ibm_torino')
print(b)

# add try-catch recovery around Session

with Session(backend=b):
	sampler = Sampler(options=options)
	result = sampler.run(CIRCS).result()

	f = open('result_torino_'+str(SCALE)+'.json', 'w')
	f.write( json.dumps(result.quasi_dists, indent=4) )
	f.close()

#aux = mrf.num_qubits - n
#
#q,delta = extract_probs(counts, n, aux)
#
#print( F(p,q), KL(p,q), delta, Z/N )

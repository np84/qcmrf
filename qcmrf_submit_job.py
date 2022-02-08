import numpy as np
from qiskit import IBMQ

options = {
	'backend_name': 'ibmq_montreal'
}

runtime_inputs = {
		#"graphs": [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[0,3]],[[0,1,2,3]]],
		"graphs": [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]]],
		"thetas": None,
		"gammas": None,
		"betas": None,
		"repetitions": 10,
		"shots": 32000,
		"layout": None,
		"measurement_error_mitigation": True,
		"optimization_level": 3
	}

print(runtime_inputs)

IBMQ.load_account()

provider = IBMQ.get_provider(
	hub='ibm-q-fraunhofer',
	group='fhg-all',
	project='ticket'
)

job = provider.runtime.run(
	program_id='qcmrf-GRqbDkgv8l',
	options=options,
	inputs=runtime_inputs
)

# Job id
print(job.job_id())
# See job status
print(job.status())

# Get results
result = job.result()

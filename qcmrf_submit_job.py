import numpy as np
import json
from qiskit import IBMQ
import argparse

parser = argparse.ArgumentParser(description='Enqueue QCMRF experiments on qiskit runtime systems.')
parser.add_argument('backend', metavar='backend', type=str, nargs=1, help='Qiskit backend.')
parser.add_argument('--layout', nargs="+", type=int)
args = parser.parse_args()

options = {
	'backend_name': args.backend[0]
}

print("LAYOUT",args.layout)

# 0 1 4 7 6 2 3 5 # auckland
# 21 23 24 25 26 22 15 17 # cairo
# 7 10 12 13 15 18 14 11 # mumbai

runtime_inputs = {
		#"graphs": [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[0,3]],[[0,1,2,3]]],
		"graphs": [[[0,1],[1,2],[2,3]]],
		"thetas": None,
		"gammas": None,
		"betas": None,
		"repetitions": 10,
		"shots": 32000,
		"layout": args.layout,
		"measurement_error_mitigation": True,
		"optimization_level": 3
	}

IBMQ.load_account()

provider = IBMQ.get_provider(
	hub='ibm-q-fraunhofer',
	group='fhg-all',
	project='ticket'
)

with open('program_id.json', 'r') as idfile:
	res = json.load(idfile)

print(res)

job = provider.runtime.run(
	program_id=res['program_id'],
	options=options,
	inputs=runtime_inputs,
	callback=print
)

print(job.job_id(),job.status())

# Get results
result = job.result()

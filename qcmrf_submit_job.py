import numpy as np
import json
from qiskit import IBMQ
import argparse

parser = argparse.ArgumentParser(description='Enqueue QCMRF experiments on qiskit runtime systems.')
parser.add_argument('backend', metavar='backend', type=str, nargs=1, help='Qiskit backend.')
parser.add_argument('--layout', nargs="+", type=int, help='List of qubits')
parser.add_argument('--auto_layout', dest='autolayout', action='store_true')
args = parser.parse_args()

backend = args.backend[0]

options = {
	'backend_name': backend
}

if args.layout is not None:
	layout = args.layout
elif args.autolayout:
	layouts = {'ibm_auckland':[0,1,4,7,6,2,3,5,8,11,9],'ibmq_mumbai':[7,10,12,13,15,18,14,11,8,16,19],'ibm_cairo':[21,23,24,25,26,22,19,20,18,15,17,12],'ibm_hanoi':[6,7,4,1,2,3,5,8,11,9,10],'ibmq_ehningen':[10,12,13,14,16,19,20,22]}
	layout = layouts[backend]
else:
	layout = None

print("LAYOUT FOR",backend,':',layout)

runtime_inputs = {
		#"graphs": [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[0,3]],[[0,1,2,3]]],
		#"graphs": [[[0,1],[1,2],[2,3],[3,4]],[[0,1],[1,2],[2,3],[3,4],[4,5]],[[0,1,2]],[[0,1,2],[2,3,4]],[[0,1,2],[2,3,4],[4,5,6]]],
		"graphs": [[[0]],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]]],
		"thetas": None,
		"gammas": None,
		"betas": None,
		"repetitions": 10,
		"shots": 48000,
		"layout": layout,
		"measurement_error_mitigation": 0,
		"optimization_level": 3
	}

with open('account.json', 'r') as accountfile:
	account = json.load(accountfile)

IBMQ.enable_account(account['token'], account['url'])

if backend == 'ibmq_qasm_simulator':
	account['project'] = 'member'

provider = IBMQ.get_provider(
	hub=account['hub'],
	group=account['group'],
	project=account['project']
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
#result = job.result()

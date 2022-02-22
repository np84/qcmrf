import numpy as np
import json
from qiskit import IBMQ
import argparse

parser = argparse.ArgumentParser(description='Enqueue QCMRF experiments on qiskit runtime systems.')
parser.add_argument('backend', metavar='backend', type=str, nargs=1, help='Qiskit backend.')
parser.add_argument('--layout', nargs="+", type=int, help='List of qubits')
parser.add_argument('--auto_layout', dest='autolayout', action='store_true')
parser.add_argument('--train', dest='train', action='store_true')
args = parser.parse_args()

backend = args.backend[0]

options = {
	'backend_name': backend
}

if args.layout is not None:
	layout = args.layout
elif args.autolayout:
	layouts = {'ibm_auckland':[0,1,4,7,6,2,3,5,8,11,9],'ibmq_mumbai':[7,10,12,13,15,18,14,11,8,16,19],'ibm_cairo':[21,23,24,25,26,22,19,20,18,15,17,12],'ibm_hanoi':[6,7,4,1,2,3,5,8,11,9,10],'ibmq_ehningen':[10,12,13,14,16,19,20,22],'ibmq_montreal':[3,5,8,9,11,14,16,19,13,20,22,12],'ibm_washington':[51,50,49,48,47,35,46,45,44,43,42],'ibmq_toronto':[23,24,25,22,19,20,16,14,11,8,5],'ibmq_bogota':list(range(5))}
	layout = layouts[backend]
else:
	layout = None

print("LAYOUT FOR",backend,':',layout)


with open('account.json', 'r') as accountfile:
	account = json.load(accountfile)

IBMQ.enable_account(account['token'], account['url'])

if backend == 'ibmq_qasm_simulator' or backend == 'ibmq_bogota':
	account['project'] = 'member'

provider = IBMQ.get_provider(
	hub=account['hub'],
	group=account['group'],
	project=account['project']
)

with open('program_id.json', 'r') as idfile:
	res = json.load(idfile)

print(res)

with open('models.json') as modelfile:
	models = json.load(modelfile)

#print('#GRAPHS =',len(models['graphs']))
#for ii,G in enumerate(models['graphs']):
#	print(G, len(models['thetas'][ii]))
	
TASKS = [8]

if not args.train:
	for T in TASKS:
		G = models['graphs'][T]
		W = models['thetas'][T]
		print(G,len(W))
		
		nqubits = np.max(G) + 1 + len(G)
		print('nqubits', nqubits)
		
		runtime_inputs = {
				"graphs": G * len(W),
				"repetitions": 1,
				"shots": 100000,
				"layout": layout[:nqubits],
				"thetas": W,
				"measurement_error_mitigation": 1,
				"optimization_level": 3
			}

		if backend == 'ibmq_qasm_simulator':
			runtime_inputs['measurement_error_mitigation'] = 0
			
		elif backend == 'ibmq_montreal' or backend == 'ibmq_toronto':
			runtime_inputs['shots'] = 32000
		
		elif backend == 'ibmq_bogota':
			runtime_inputs['shots'] = 20000

		job = provider.runtime.run(
			program_id=res['program_id'],
			options=options,
			inputs=runtime_inputs,
			callback=print
		)

		print(job.job_id(),job.status())
		
else:
	runtime_inputs = {
		"graphs": [[[0,1],[1,2]]],
		"mu": [0.3,0,0,0.7,0.1,0.2,0.1,0.6],
		"data": ['000','001','001','110','111','111','111','111','111','111'],
		"iterations": 50,
		"train": True,
		"adam": True,
		"layout": layout,
		"shots": 100000,
		"measurement_error_mitigation": 1,
	}

	if backend == 'ibmq_qasm_simulator':
		runtime_inputs['measurement_error_mitigation'] = 0
		
	elif backend == 'ibmq_montreal' or backend == 'ibmq_toronto':
		runtime_inputs['shots'] = 32000
		
	elif backend == 'ibmq_bogota':
		runtime_inputs['shots'] = 20000

	job = provider.runtime.run(
		program_id=res['program_id'],
		options=options,
		inputs=runtime_inputs,
		callback=print
	)

	print(job.job_id(),job.status())

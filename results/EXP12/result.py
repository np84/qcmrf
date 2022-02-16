from pathlib import Path
import numpy as np
import os.path
import json
import itertools

import argparse

parser = argparse.ArgumentParser(description='Evaluate QCMRF results.')
parser.add_argument('--sim', dest='sim', action='store_true')
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

pathlist = Path('.').glob('**/*info*.json')
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


for G in results.keys():

	g = graphTupleToList(G)
	print(g)
	
	FIDS = {}
	KLS  = {}
	SRS  = {}
	
	l = 10
	
	for i in range(l):
		FIDS[i] = []
		KLS[i]  = []
		SRS[i]  = []

	for jj,r in enumerate(results[G]):
		if (r.backend == 'ibmq_qasm_simulator' and args.sim) or (r.backend != 'ibmq_qasm_simulator' and not args.sim):
			for j,f in enumerate(r.F):
				#print(jj,j,r.backend,r.mit)
				FIDS[j].append(f) 
			for j,k in enumerate(r.KL):
				KLS[j].append(k) 
			for j,s in enumerate(r.SR):
				SRS[j].append(s) 
			first = False

	if len(FIDS[0]) == 0:
		break

	FF = []
	KK = []
	SS = []
	
	for i in range(l):
		if len(FIDS[i]) > 0:
			FF.append(np.max(FIDS[i])) # best over all backends for each run
			KK.append(np.min(KLS[i])) # best over all backends for each run
			SS.append(np.max(SRS[i])) # best over all backends for each run

	print("%.3f" % (np.median(FF)))
	print("%.3f" % (np.median(KK)))
	print("%.3f" % (np.median(SS)))

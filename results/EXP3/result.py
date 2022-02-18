from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook
import os.path
import json
import itertools

N = 100000

ite = range(100)

fig, ax = plt.subplots()
ax.set(xlabel='Iteration', ylabel='neg. avg. log-likelihood', title='QCMRF Training')
ax.grid()

L = []
S = []
NAM = []

pathlist = Path('.').glob('./*info*.json')
for path in pathlist:
	# because path is object not string
	info_file_name = str(path)
	result_json_name = info_file_name.replace('info','result')
	result_txt_name = result_json_name.replace('json','txt')
	has_info = os.path.isfile(info_file_name)
	has_json = os.path.isfile(result_json_name)
	has_txt  = os.path.isfile(result_txt_name)
	
	if has_info:
		info_f = open(info_file_name,'r')
		info = json.load(info_f)
		info_f.close()

		if has_json:
			result_f = open(result_json_name,'r')
			result = json.load(result_f)
			result_f.close()
		elif has_txt:
			result_f = open(result_txt_name,'r')
			result = json.load(result_f)
			result_f.close()
			
		name = info['backend']
		loss = result['all_results']['loss']
		rate = result['all_results']['success_rate']

		NAM.append(name)
		S.append(np.array(rate))
		L.append(np.array(loss))

M = 21

fig, ax = plt.subplots()
ax.set_aspect(16/2)
ax.set(xlabel='Iteration', ylabel='neg. avg. log-likelihood', title='')
#ax.grid()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.plot(ite[:M], L[0][:M], label='Simulation', linewidth=2)
ax.plot(ite[:M], ((L[1]+L[2])/2)[:M], label='Hardware', linewidth=2)
plt.legend()
fig.savefig("train_ll.pdf", bbox_inches='tight')
plt.show()

print(S)

fig, ax = plt.subplots()
ax.set_aspect(16/2)
ax.set(xlabel='Iteration', ylabel='Success rate', title='')
#ax.grid()
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.plot(ite[:M], S[0][:M], label='Simulation', linewidth=2)
ax.plot(ite[:M], ((S[1]+S[2])/2)[:M], label='Hardware', linewidth=2)
#plt.legend()
fig.savefig("train_sr.pdf", bbox_inches='tight')
plt.show()

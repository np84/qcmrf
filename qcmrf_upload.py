import os
from qiskit import IBMQ
import json

with open('account.json', 'r') as accountfile:
	account = json.load(accountfile)

IBMQ.enable_account(account.token, account.url)

provider = IBMQ.get_provider(
	hub=account.hub,
	group=account.group,
	project=account.project
)

qcmrf_data = os.path.join(os.getcwd(), "./qcmrf_runtime.py")
qcmrf_json = os.path.join(os.getcwd(), "./qcmrf_runtime.json")

program_id = provider.runtime.upload_program(
	data=qcmrf_data,
	metadata=qcmrf_json
)

ID = {'program_id':program_id}
res = json.dumps(ID)

print(res)

with open('program_id.json', 'w') as outfile:
	outfile.write(res)

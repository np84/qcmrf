import os
import json
from qiskit import IBMQ

with open('account.json', 'r') as accountfile:
	account = json.load(accountfile)

IBMQ.enable_account(account['token'], account['url'])

provider = IBMQ.get_provider(
	hub=account['hub'],
	group=account['group'],
	project=account['project']
)

with open('program_id.json', 'r') as idfile:
	res = json.load(idfile)

print(res)

provider.runtime.delete_program(res['program_id'])

import os
import json
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(project='ticket')  # Substitute with your provider.

with open('program_id.json', 'r') as idfile:
	res = json.load(idfile)

print(res)

provider.runtime.delete_program(res['program_id'])

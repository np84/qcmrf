import os
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(project='ticket')  # Substitute with your provider.

qcmrf_data = os.path.join(os.getcwd(), "./qcmrf_runtime.py")
qcmrf_json = os.path.join(os.getcwd(), "./qcmrf_runtime.json")

program_id = provider.runtime.upload_program(
    data=qcmrf_data,
    metadata=qcmrf_json
)
print(program_id)

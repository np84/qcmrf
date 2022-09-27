from qiskit import Aer
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
from qiskit.providers.ibmq.runtime import UserMessenger

import json
import qcmrf_runtime as qcmrf

def test():
	backend = Aer.get_backend('qasm_simulator')
	inputs = {
		"graphs": [[[0,1],[1,2]]],
		"mu": [0.3,0,0,0.7,0.1,0.2,0.1,0.6],
		"data": ['000','001','001','110','111','111','111','111','111','111'],
		"iterations": 32,
		"train": True,
		"adam": True,
		"measurement_error_mitigation": 0,
		"shots": 64000,
		"layout": [0,1,2,3,4,5]
	}
	user_messenger = UserMessenger()
	serialized_inputs = json.dumps(inputs, cls=RuntimeEncoder)
	deserialized_inputs = json.loads(serialized_inputs, cls=RuntimeDecoder)
	qcmrf.main(backend, user_messenger, **deserialized_inputs)

test()

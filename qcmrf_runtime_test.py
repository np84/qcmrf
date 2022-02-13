from qiskit import Aer
from qiskit.providers.ibmq.runtime.utils import RuntimeEncoder, RuntimeDecoder
from qiskit.providers.ibmq.runtime import UserMessenger

import json
import qcmrf_runtime as qcmrf

def test():
	backend = Aer.get_backend('qasm_simulator')
	inputs = {
		"graphs": [[[0,1],[1,2]]],
		"repetitions": 3,
		"shots": 10000
	}
	user_messenger = UserMessenger()
	serialized_inputs = json.dumps(inputs, cls=RuntimeEncoder)
	deserialized_inputs = json.loads(serialized_inputs, cls=RuntimeDecoder)
	qcmrf.main(backend, user_messenger, **deserialized_inputs)

test()

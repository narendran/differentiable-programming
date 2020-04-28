# Run this test with `python -m unittest jaxnn.jaxnn_test`

import unittest
import jax.numpy as np
from numpy import testing as np_testing

from jaxnn import Neuron

class TestNeuron(unittest.TestCase):

    def test_default_init(self):
        n = Neuron(3)
        np_testing.assert_array_equal(n.weights, np.array([1., 1., 1.]))

        self.assertEqual(n.bias, 0)

    def test_forward_pass(self):
        inputs = np.array([10., 20., 30.])
        n = Neuron(len(inputs))

        np_testing.assert_array_equal(n.forward(inputs), np.array(60.))

if __name__ == '__main__':
    unittest.main()

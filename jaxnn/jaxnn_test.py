# Run this test with `python -m unittest jaxnn.jaxnn_test`

import unittest
import jax.numpy as np
from numpy import testing as np_testing

from jaxnn import Neuron

class TestNeuron(unittest.TestCase):

    def test_default_init(self):
        n = Neuron(3)

        expected_weights = np.array([1., 1., 1.])
        expected_bias = 0

        np_testing.assert_array_equal(n.weights, expected_weights)
        self.assertEqual(n.bias, expected_bias)

    def test_forward_pass(self):
        inputs = np.array([10., 20., 30.])
        n = Neuron(len(inputs))

        expected_output = np.array(60.)

        np_testing.assert_array_equal(n.forward(inputs), expected_output)

    def test_backward_pass_one_loss(self):
        inputs = np.array([10., 20., 30.])
        n = Neuron(len(inputs))

        n.weights = np.array([3., 4., 5.])
        n.forward(inputs)

        backward_loss_gradients = np.array([2.0])
        n.learning_rate = 0.1

        expected_output = np.array([6., 8., 10.])
        expected_weights = np.array([2.4, 3.2, 4.])

        np_testing.assert_array_equal(n.backward(backward_loss_gradients), expected_output)
        np_testing.assert_array_equal(n.weights, expected_weights)

    def test_training_flow_second_input(self):
        inputs = np.array([10., 2., 5.])
        n = Neuron(len(inputs))
        n.weights = np.array([1., 1., -1.])
        n.learning_rate = 0.1

        for i in range(20):
            output = float(n.forward(inputs))
            print("Output: ", output)

            loss = np.array([output - inputs[1]]) # The learnt function is a selector, which selects second input.
            print("Loss: ", loss)

            n.backward(loss)
            print("Weights: ", n.weights)

if __name__ == '__main__':
    unittest.main()
